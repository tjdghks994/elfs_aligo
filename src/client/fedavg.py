from collections import OrderedDict
from copy import deepcopy
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset

# 베이지안 최적화를 위한 임포트 (scikit-optimize)
try:
    from skopt import gp_minimize
    from skopt.space import Real
except ImportError:
    print("Warning: scikit-optimize not installed. Bayesian Optimization will be unavailable.")
    gp_minimize = None

from data.utils.datasets import BaseDataset
from src.utils.functional import evaluate_model, get_optimal_cuda_device
from src.utils.metrics import Metrics
from src.utils.models import DecoupledModel
from src.utils.aligo_utils import (
    FocalLoss,
    DiceLoss,
    LDAMLoss,
    BalancedSoftmaxLoss,
    calculate_normalized_entropy, 
    calculate_interpolation_weight,
    )




class FedAvgClient:
    def __init__(
        self,
        model: DecoupledModel,
        optimizer_cls: type[torch.optim.Optimizer],
        lr_scheduler_cls: type[torch.optim.lr_scheduler.LRScheduler],
        args: DictConfig,
        dataset: BaseDataset,
        data_indices: list,
        device: torch.device | None,
        return_diff: bool,
    ):
        self.client_id: int = None
        self.args = args
        if device is None:
            self.device = get_optimal_cuda_device(use_cuda=self.args.common.use_cuda)
        else:
            self.device = device
        self.dataset = dataset
        self.model = model.to(self.device)
        self.regular_model_params: OrderedDict[str, torch.Tensor]
        self.personal_params_name: list[str] = []
        self.regular_params_name = list(key for key, _ in self.model.named_parameters())
        if self.args.common.buffers == "local":
            self.personal_params_name.extend(
                [name for name, _ in self.model.named_buffers()]
            )
        elif self.args.common.buffers == "drop":
            self.init_buffers = deepcopy(OrderedDict(self.model.named_buffers()))

        self.optimizer = optimizer_cls(params=self.model.parameters())
        self.init_optimizer_state = deepcopy(self.optimizer.state_dict())

        self.lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None
        self.init_lr_scheduler_state: dict = None
        self.lr_scheduler_cls = None
        if lr_scheduler_cls is not None:
            self.lr_scheduler_cls = lr_scheduler_cls
            self.lr_scheduler = self.lr_scheduler_cls(optimizer=self.optimizer)
            self.init_lr_scheduler_state = deepcopy(self.lr_scheduler.state_dict())

        # [{"train": [...], "val": [...], "test": [...]}, ...]
        self.data_indices = data_indices
        # Please don't bother with the [0], which is only for avoiding raising runtime error by setting Subset(indices=[]) with `DataLoader(shuffle=True)`
        self.trainset = Subset(self.dataset, indices=[0])
        self.valset = Subset(self.dataset, indices=[])
        self.testset = Subset(self.dataset, indices=[])
        self.trainloader = DataLoader(
            self.trainset, batch_size=self.args.common.batch_size, shuffle=True
        )
        self.valloader = DataLoader(self.valset, batch_size=self.args.common.batch_size)
        self.testloader = DataLoader(
            self.testset, batch_size=self.args.common.batch_size
        )
        self.testing = False

        self.local_epoch = self.args.common.local_epoch
        # L_Base (Cross-Entropy) 초기화
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

        # --- ELFS with ALI-GO 초기화 시작 ---
        self.beta = 1.0 # 기본 보간 가중치 (L_Base만 사용)

        if self.args.aligo.use_aligo:
            self._initialize_aligo()

        # --- ELFS with ALI-GO 초기화 끝 ---

        self.eval_results = {}

        self.return_diff = return_diff

    def _initialize_aligo(self):
        # 1. L_Imb 초기화 (Focal Loss 예시, args를 통해 gamma 설정 가능)
        if self.args.aligo.loss == "focal":
            gamma = self.args.aligo.focal_gamma
            self.criterion_imb = FocalLoss(gamma=gamma).to(self.device)
        elif self.args.aligo.loss == "dice":
            self.criterion_imb = DiceLoss().to(self.device)
        
        # 2. 최적화된 파라미터(Theta*) 로드 (Algorithm 1, Prerequisite)
        # 서버 설정을 통해 args로 전달됨
        self.theta1 = self.args.aligo.aligo_theta1
        self.theta0 = self.args.aligo.aligo_theta0

        if self.theta1 is None or self.theta0 is None:
            raise ValueError(f"Client {self.client_id}: ALI-GO enabled but theta parameters (aligo_theta1/0) are missing in args.")

        # 초기 모델 상태 저장 (BO 시뮬레이션 재시작을 위함 - 매우 중요)
        self.initial_model_state = deepcopy(self.model.state_dict())

        # set theta1, theta0
        # self.run_bayesian_optimization()
        # print(f"Client {self.client_id} BO Result: Theta1={self.theta1}, Theta0={self.theta0}")

        # set beta for ALI-GO
        # 3. 로컬 엔트로피 계산 (Algorithm 1, Line 15-16)
        if not self.dataset.classes:
            raise ValueError("ALI-GO requires 'classes' in dataset.")
        
        self.entropy = list()
        self.cls_num_list = list()

        for idx in range(len(self.data_indices)):
            all_labels = self._get_all_labels_robust(idx)
            normalized_entropy = calculate_normalized_entropy(all_labels, len(self.dataset.classes))
            # print(f"client id: {self.client_id}, H_i: {normalized_entropy}")

            # 4. 보간 가중치 계산 (Algorithm 1, Line 18)
            # Theta와 Entropy가 고정값이므로 Beta도 한 번만 계산하여 최적화
            
            self.entropy.append(normalized_entropy)

            # for LDAM Loss, Balanced Softmax
            self.cls_num_list.append(np.bincount(all_labels, minlength=len(self.dataset.classes)))

        # print(self.entropy)
        print(f"CE {len([e for e in self.entropy if e >= self.args.aligo.elfs_threshold])}")
        print(f"FO {len([e for e in self.entropy if e < self.args.aligo.elfs_threshold])}")


    def _get_all_labels_robust(self, idx):
        """데이터 로더의 데이터셋에서 모든 레이블을 효율적이고 안정적으로 추출합니다."""
        # dataset = self.trainloader.dataset
        
        # 효율적인 접근 시도 (데이터셋 구현체에 따라 다름)
        # print(f"trainset: {self.trainloader.dataset}")
        # if hasattr(self.trainloader.dataset.dataset, 'targets') or hasattr(self.valloader.dataset.dataset, 'targets'):
        #     labels = self.trainloader.dataset.dataset.targets + self.valloader.dataset.dataset.targets
        # else:
        #     # 폴백: 데이터 로더 순회 (느릴 수 있음)
        #     print(f"Warning: Iterating DataLoader for Client {self.client_id} to get labels.")

        self.trainset.indices = self.data_indices[idx]["train"]
        self.valset.indices = self.data_indices[idx]["val"]
        self.testset.indices = self.data_indices[idx]["test"]

        labels = []
        for _, target in self.trainloader:
            labels.extend(target.cpu().tolist())
        for _, target in self.valloader:
            labels.extend(target.cpu().tolist())
        
        return np.array(labels)
    

    def calculate_loss(self, outputs, targets):
        """
        손실을 계산합니다. (Algorithm 1, Line 20)
        이 메소드는 FedProx와 같은 다른 알고리즘에서 super()로 호출되어 쉽게 확장될 수 있습니다.
        """
        if self.args.aligo.algo == "aligo" and self.args.aligo.use_aligo:
            # Adaptive Loss Interpolation (ALI)

            # L_i = beta * L_Base + (1 - beta) * L_Imb
            # beta = self.entropy[self.client_id]["beta"]
            loss_base = self.criterion(outputs, targets)
            loss_imb = self.criterion_imb(outputs, targets)
            loss = self.beta * loss_base + (1-self.beta) * loss_imb
            # loss = self.entropy[self.client_id] * loss_base + (1-self.entropy[self.client_id]) * loss_imb
        elif self.args.aligo.algo == "elfs" and self.args.aligo.use_aligo:
            if self.entropy[self.client_id] >= self.args.aligo.elfs_threshold:
                # print(f"[id {self.client_id}] use CE because h_hat {self.entropy[self.client_id]}")
                # Cross Entropy
                loss = self.criterion(outputs, targets)
            else:
                # Focal Loss
                # print(f"[id {self.client_id}] use FO  because h_hat {self.entropy[self.client_id]}")
                loss = self.criterion_imb(outputs, targets)
        elif self.args.aligo.algo == "tripod" and self.args.aligo.use_aligo:
            if self.entropy[self.client_id] > 0.6:
                # print(f"[id {self.client_id}] use CE because h_hat {self.entropy[self.client_id]}")
                # Cross Entropy
                loss = self.criterion(outputs, targets)
            elif 0.3 < self.entropy[self.client_id] <= 0.6:
                # Balanced Softmaxloss
                cnl = self.cls_num_list[self.client_id]
                self.criterion_bs = BalancedSoftmaxLoss(cls_num_list=cnl).to(self.device)
                loss = self.criterion_bs(outputs, targets)
            else:
                # Focal Loss
                # print(f"[id {self.client_id}] use FO  because h_hat {self.entropy[self.client_id]}")
                loss = self.criterion_imb(outputs, targets)
        elif self.args.aligo.algo == "prob" and self.args.aligo.use_aligo:
            h_avg = np.mean(self.entropy)
            if torch.rand(1).item() < h_avg:
                loss = self.criterion(outputs, targets)
            else:
                cnl = self.cls_num_list[self.client_id]
                self.criterion_bs = BalancedSoftmaxLoss(cls_num_list=cnl).to(self.device)
                loss = self.criterion_bs(outputs, targets)
        else:
            # 표준 FedAvg 손실
            loss = self.criterion(outputs, targets)
            
        return loss

    def load_data_indices(self):
        """This function is for loading data indices for No.`self.client_id`
        client."""
        self.trainset.indices = self.data_indices[self.client_id]["train"]
        self.valset.indices = self.data_indices[self.client_id]["val"]
        self.testset.indices = self.data_indices[self.client_id]["test"]

    def train_with_eval(self):
        """Wraps `fit()` with `evaluate()` and collect model evaluation
        results.

        A model evaluation results dict: {
                `before`: {...}
                `after`: {...}
                `message`: "..."
            }
            `before` means pre-local-training.
            `after` means post-local-training
        """
        eval_results = {
            "before": {"train": Metrics(), "val": Metrics(), "test": Metrics()},
            "after": {"train": Metrics(), "val": Metrics(), "test": Metrics()},
        }
        eval_results["before"] = self.evaluate()

        if self.args.aligo.use_aligo:
            h_hat = self.entropy[self.client_id]
            self.beta = calculate_interpolation_weight(h_hat, self.theta1, self.theta0)
        
            # for LDAM, Balanced Softmax
            cnl = self.cls_num_list[self.client_id]
            if self.args.aligo.loss == "ldam":
                self.criterion_imb = LDAMLoss(cls_num_list=cnl).to(self.device)
            elif self.args.aligo.loss == "bs":
                self.criterion_imb = BalancedSoftmaxLoss(cls_num_list=cnl).to(self.device)

        if self.local_epoch > 0:
            self.fit()
            eval_results["after"] = self.evaluate()

        eval_msg = []
        for split, color, flag, subset in [
            ["train", "yellow", self.args.common.test.client.train, self.trainset],
            ["val", "green", self.args.common.test.client.val, self.valset],
            ["test", "cyan", self.args.common.test.client.test, self.testset],
        ]:
            if len(subset) > 0 and flag:
                eval_msg.append(
                    f"client [{self.client_id}]\t"
                    f"[{color}]({split}set)[/{color}]\t"
                    f"[red]loss: {eval_results['before'][split].loss:.4f} -> "
                    f"{eval_results['after'][split].loss:.4f}\t[/red]"
                    f"[blue]accuracy: {eval_results['before'][split].accuracy:.2f}% -> {eval_results['after'][split].accuracy:.2f}%[/blue]"
                )

        eval_results["message"] = eval_msg
        self.eval_results = eval_results

    def set_parameters(self, package: dict[str, Any]):
        self.client_id = package["client_id"]
        self.local_epoch = package["local_epoch"]
        self.load_data_indices()

        self.theta1, self.theta0 = package["theta1"], package["theta0"]

        if (
            package["optimizer_state"]
            and not self.args.common.reset_optimizer_on_global_epoch
        ):
            self.optimizer.load_state_dict(package["optimizer_state"])
        else:
            self.optimizer.load_state_dict(self.init_optimizer_state)

        if self.lr_scheduler is not None:
            if package["lr_scheduler_state"]:
                self.lr_scheduler.load_state_dict(package["lr_scheduler_state"])
            else:
                self.lr_scheduler.load_state_dict(self.init_lr_scheduler_state)

        self.model.load_state_dict(package["regular_model_params"], strict=False)
        self.model.load_state_dict(package["personal_model_params"], strict=False)
        if self.args.common.buffers == "drop":
            self.model.load_state_dict(self.init_buffers, strict=False)

        if self.return_diff:
            model_params = self.model.state_dict()
            self.regular_model_params = OrderedDict(
                (key, model_params[key].clone().cpu())
                for key in self.regular_params_name
            )

    def train(self, server_package: dict[str, Any]) -> dict:
        self.set_parameters(server_package)
        self.train_with_eval()
        client_package = self.package()
        return client_package

    def package(self):
        """Package data that client needs to transmit to the server. You can
        override this function and add more parameters.

        Returns:
            A dict: {
                `weight`: Client weight. Defaults to the size of client training set.
                `regular_model_params`: Client model parameters that will join parameter aggregation.
                `model_params_diff`: The parameter difference between the client trained and the global. `diff = global - trained`.
                `eval_results`: Client model evaluation results.
                `personal_model_params`: Client model parameters that absent to parameter aggregation.
                `optimzier_state`: Client optimizer's state dict.
                `lr_scheduler_state`: Client learning rate scheduler's state dict.
            }
        """
        model_params = self.model.state_dict()
        client_package = dict(
            weight=len(self.trainset),
            eval_results=self.eval_results,
            regular_model_params={
                key: model_params[key].clone().cpu() for key in self.regular_params_name
            },
            personal_model_params={
                key: model_params[key].clone().cpu()
                for key in self.personal_params_name
            },
            optimizer_state=deepcopy(self.optimizer.state_dict()),
            lr_scheduler_state=(
                {}
                if self.lr_scheduler is None
                else deepcopy(self.lr_scheduler.state_dict())
            ),
        )
        if self.return_diff:
            client_package["model_params_diff"] = {
                key: param_old - param_new
                for (key, param_new), param_old in zip(
                    client_package["regular_model_params"].items(),
                    self.regular_model_params.values(),
                )
            }
            client_package.pop("regular_model_params")
        return client_package

    def fit(self):
        self.model.train()
        self.dataset.train()
        for _ in range(self.local_epoch):
            for x, y in self.trainloader:
                # When the current batch size is 1, the batchNorm2d modules in the model would raise error.
                # So the latent size 1 data batches are discarded.
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)

                loss = self.calculate_loss(logit, y)
                # loss = self.criterion(logit, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    @torch.no_grad()
    def evaluate(self, model: torch.nn.Module = None) -> dict[str, Metrics]:
        """Evaluating client model.

        Args:
            model: Used model. Defaults to None, which will fallback to `self.model`.

        Returns:
            A evalution results dict: {
                `train`: results on client training set.
                `val`: results on client validation set.
                `test`: results on client test set.
            }
        """
        target_model = self.model if model is None else model
        target_model.eval()
        self.dataset.eval()
        train_metrics = Metrics()
        val_metrics = Metrics()
        test_metrics = Metrics()
        criterion = torch.nn.CrossEntropyLoss(reduction="sum")

        if (
            len(self.testset) > 0
            and (self.testing or self.args.common.client_side_evaluation)
            and self.args.common.test.client.test
        ):
            test_metrics = evaluate_model(
                model=target_model,
                dataloader=self.testloader,
                criterion=criterion,
                device=self.device,
            )

        if (
            len(self.valset) > 0
            and (self.testing or self.args.common.client_side_evaluation)
            and self.args.common.test.client.val
        ):
            val_metrics = evaluate_model(
                model=target_model,
                dataloader=self.valloader,
                criterion=criterion,
                device=self.device,
            )

        if (
            len(self.trainset) > 0
            and (self.testing or self.args.common.client_side_evaluation)
            and self.args.common.test.client.train
        ):
            train_metrics = evaluate_model(
                model=target_model,
                dataloader=self.trainloader,
                criterion=criterion,
                device=self.device,
            )
        return {"train": train_metrics, "val": val_metrics, "test": test_metrics}

    def test(self, server_package: dict[str, Any]) -> dict[str, dict[str, Metrics]]:
        """Test client model. If `finetune_epoch > 0`, `finetune()` will be
        activated.

        Args:
            server_package: Parameter package.

        Returns:
            A model evaluation results dict : {
                `before`: {...}
                `after`: {...}
                `message`: "..."
            }
            `before` means pre-local-training.
            `after` means post-local-training
        """
        self.testing = True
        self.set_parameters(server_package)

        results = {
            "before": {"train": Metrics(), "val": Metrics(), "test": Metrics()},
            "after": {"train": Metrics(), "val": Metrics(), "test": Metrics()},
        }

        results["before"] = self.evaluate()
        if self.args.common.test.client.finetune_epoch > 0:
            frz_params_dict = deepcopy(self.model.state_dict())
            self.finetune()
            results["after"] = self.evaluate()
            self.model.load_state_dict(frz_params_dict)

        self.testing = False
        return results

    def finetune(self):
        """Client model finetuning.

        This function will only be activated in `test()`
        """
        self.model.train()
        self.dataset.train()
        for _ in range(self.args.common.test.client.finetune_epoch):
            for x, y in self.trainloader:
                if len(x) <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                logit = self.model(x)
                loss = self.criterion(logit, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
