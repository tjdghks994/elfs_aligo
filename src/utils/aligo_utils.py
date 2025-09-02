import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- 1. Entropy Calculation (Eq. 1 & 2) ---

def calculate_normalized_entropy(labels, num_classes):
    """
    로컬 데이터셋 레이블의 정규화된 섀넌 엔트로피(H_hat_i)를 계산합니다.
    """
    if len(labels) == 0 or num_classes <= 1:
        return 1.0 # 데이터가 없거나 클래스가 1개인 경우 균등 분포로 간주

    # 1. 클래스 빈도수 계산
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    elif isinstance(labels, list):
        labels = np.array(labels)
        
    counts = np.bincount(labels, minlength=num_classes)
    # print(f"labels: {len(labels)}, num_classes: {num_classes}")
    # print(f"bin count: {counts}")
    
    # 2. 확률 계산 (p_i,k)
    probabilities = counts / len(labels)
    
    # 3. 섀넌 엔트로피 계산 (H_i)
    non_zero_probs = probabilities[probabilities > 0]
    entropy = -np.sum(non_zero_probs * np.log2(non_zero_probs))
    
    # 4. 정규화 (H_i / log2(K))
    max_entropy = np.log2(num_classes)
    normalized_entropy = entropy / max_entropy

    return normalized_entropy

# --- 2. Parameterized Weighting Function (Eq. 4) ---

def calculate_interpolation_weight(normalized_entropy, theta1, theta0):
    """
    파라미터화된 시그모이드 함수를 사용하여 보간 가중치(beta_i)를 계산합니다.
    beta_i = sigma(theta1 * H_hat_i + theta0)
    """
    exponent = theta1 * normalized_entropy + theta0
    # 수치적 안정성을 위한 클리핑
    exponent = np.clip(exponent, -50, 50) 
    weight = 1.0 / (1.0 + np.exp(-exponent))
    return weight

# --- 3. Imbalance-aware Loss (L_Imb 예시: Focal Loss) ---

class FocalLoss(nn.Module):
    """Focal Loss 구현"""
    def __init__(self, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
# --- 2. LDAM Loss (Label-Distribution-Aware Margin Loss) ---

class LDAMLoss(nn.Module):
    """
    LDAM Loss 구현. 클래스 빈도에 반비례하여 마진을 설정합니다.
    Reference: https://arxiv.org/abs/1906.07413
    cls_num_list: 학습 데이터셋의 클래스별 샘플 수 리스트 (필수).
    max_m: 최대 마진 (기본값 0.5).
    s: 스케일링 팩터 (기본값 30).
    """
    def __init__(self, cls_num_list, max_m=0.5, s=30, weight=None, reduction='mean'):
        super(LDAMLoss, self).__init__()
        
        if cls_num_list is None or len(cls_num_list) == 0:
            raise ValueError("LDAMLoss requires cls_num_list.")
            
        cls_num_list = torch.tensor(cls_num_list, dtype=torch.float32)
        
        # 1. 마진 계산: m_j ∝ 1/n_j^(1/4)
        m_list = 1.0 / torch.pow(cls_num_list, 0.25)
        
        # 2. 마진 정규화 (최대값이 max_m이 되도록)
        m_list = m_list * (max_m / torch.max(m_list))
        
        # 마진 리스트를 버퍼로 등록
        self.register_buffer('m_list', m_list)
        
        self.s = s
        # 선택적 가중치 (예: Deferred Re-Weighting (DRW) 스케줄링 사용 시 weight 업데이트 필요)
        if weight is not None:
            if not isinstance(weight, torch.Tensor):
                weight = torch.tensor(weight, dtype=torch.float32)
            self.register_buffer('weight', weight)
        else:
            self.weight = None
            
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: (N, C) logits, targets: (N) labels

        # 1. 타겟 클래스에 해당하는 마진 선택
        # self.m_list[targets]를 사용하여 효율적으로 배치 마진을 가져옴
        batch_margins = self.m_list[targets]
        
        # 2. 마진 적용: logit_y - m_y
        # 정답 클래스의 로짓에서만 마진을 빼야 함.
        
        # 타겟을 one-hot 인코딩으로 변환하여 마스킹 인덱스 생성
        index = F.one_hot(targets, num_classes=inputs.size(1)).bool()
        
        output = inputs.clone()
        # boolean 마스크를 사용하여 정답 로짓에만 마진 적용
        # output[index]는 1차원 텐서이며, batch_margins의 요소들이 순서대로 대응되어 차감됨
        output[index] -= batch_margins

        # 3. 스케일링 적용
        output = output * self.s
        
        # 4. Cross Entropy Loss 계산
        return F.cross_entropy(output, targets, weight=self.weight, reduction=self.reduction)

# --- 3. Balanced Softmax Loss ---

class BalancedSoftmaxLoss(nn.Module):
    """
    Balanced Softmax Loss 구현. Softmax 계산 시 클래스 빈도(Prior)를 통합합니다.
    Reference: https://arxiv.org/abs/2007.07314
    cls_num_list: 학습 데이터셋의 클래스별 샘플 수 리스트 (필수).
    """
    def __init__(self, cls_num_list, reduction='mean'):
        super(BalancedSoftmaxLoss, self).__init__()
        
        if cls_num_list is None or len(cls_num_list) == 0:
            raise ValueError("BalancedSoftmaxLoss requires cls_num_list.")

        cls_num_list = torch.tensor(cls_num_list, dtype=torch.float32)
        
        # 1. 클래스 사전 확률(p_j) 계산
        priors = cls_num_list / cls_num_list.sum()
        
        # 2. 로그 확률 계산 (수치 안정성을 위해 작은 값 추가)
        log_priors = torch.log(priors + 1e-9)
        
        # 로그 확률을 버퍼로 등록
        self.register_buffer('log_priors', log_priors)
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: (N, C) logits, targets: (N) labels
        
        # Balanced Softmax 개념: logit을 log(p_j)만큼 조정하여 샘플링 편향 상쇄
        # f'_j = f_j + log(p_j)
        
        # 브로드캐스팅을 통해 (N, C) 형태의 inputs에 (C) 형태의 log_priors를 더함
        adjusted_logits = inputs + self.log_priors
        
        # 조정된 로짓에 표준 Cross Entropy Loss 적용
        return F.cross_entropy(adjusted_logits, targets, reduction=self.reduction)


# =====================================================================================
# 메트릭 기반 학습 (Metric-based Learning) - 주로 Segmentation 용도
# =====================================================================================

def _prepare_classification_inputs(inputs, targets, activation):
    """
    분류 작업을 위해 입력 Logits에 활성화 함수를 적용하고 타겟을 (N, C) 형태로 통일합니다.
    Input: (N, C) logits
    Target: (N) labels (multi-class) 또는 (N, C) multi-hot (multi-label)
    """
    
    if inputs.dim() != 2:
        # 일반적인 분류 작업은 (N, C) 입력을 가정함
        raise ValueError(f"Classification Loss expects 2D inputs (N, C), but got {inputs.dim()}D.")

    N, C = inputs.shape

    # 1. 활성화 함수 적용 (확률 계산)
    if activation == 'softmax':
        # 멀티클래스 분류
        if C == 1:
             # 이진 분류(C=1)인 경우 Sigmoid 사용이 일반적
             probs = torch.sigmoid(inputs)
        else:
            probs = F.softmax(inputs, dim=1)
    elif activation == 'sigmoid':
        # 멀티레이블 또는 이진 분류
        probs = torch.sigmoid(inputs)
    elif activation == 'none':
        probs = inputs # 활성화 함수가 이미 적용된 경우
    else:
        raise ValueError(f"Unsupported activation: {activation}. Use 'softmax', 'sigmoid', or 'none'.")

    # 2. 타겟 형태 변환 (One-hot 또는 Multi-hot (N, C) 형태로 통일)
    if targets.dim() == 1 or (targets.dim() == 2 and targets.shape[1] == 1):
        # (N) 또는 (N, 1) 형태의 정수 레이블인 경우
        if C > 1:
            # 멀티클래스: One-hot 인코딩으로 변환
            targets_format = F.one_hot(targets.long().squeeze(), num_classes=C).float()
        else:
            # 이진 분류 (C=1): (N, 1) 형태로 변환
            targets_format = targets.float().view(N, 1)
    elif targets.dim() == 2 and targets.shape[1] == C:
        # 이미 (N, C) 형태인 경우 (예: 멀티레이블, 소프트 레이블)
        targets_format = targets.float()
    else:
        raise ValueError(f"Targets shape {targets.shape} is incompatible with inputs shape {inputs.shape}.")
            
    # 입력과 타겟의 최종 형태 확인 (N, C)
    if probs.shape != targets_format.shape:
         # 이 경우는 발생해서는 안 되지만 안전을 위해 추가
         raise ValueError(f"Shape mismatch after preparation: inputs {probs.shape} vs targets {targets_format.shape}")

    return probs, targets_format

# Segmentation Loss를 위한 헬퍼 함수 (N차원 데이터 지원 및 일반화)
def _get_probabilities_and_one_hot_targets(inputs, targets, activation):
    """
    입력 Logits에 활성화 함수를 적용하고, 타겟을 One-hot 인코딩으로 변환한 후 공간 차원을 평탄화합니다.
    """
    N, C = inputs.shape[0], inputs.shape[1]

    # 1. 활성화 함수 적용
    if activation == 'softmax':
        # 멀티클래스 (클래스 간 상호 배타적)
        if C == 1:
             # 클래스가 1개인 경우 softmax는 1이 되므로 sigmoid 사용이 일반적임
             probs = torch.sigmoid(inputs)
        else:
            probs = F.softmax(inputs, dim=1)
    elif activation == 'sigmoid':
        # 멀티레이블 또는 이진 분류
        probs = torch.sigmoid(inputs)
    else:
        probs = inputs # 활성화 함수가 이미 적용된 경우

    # 2. 타겟을 One-hot 인코딩으로 변환
    # 입력이 (N, C, D1, D2...)이고 타겟이 (N, D1, D2...)인 경우 (Segmentation Mask)
    if targets.dim() == inputs.dim() - 1 and C > 1:
        targets_one_hot = F.one_hot(targets.long(), num_classes=C)
        # One-hot 변환 시 채널 차원(C)이 마지막에 생성됨. 예: (N, H, W, C)
        # (N, C, H, W)로 차원 변경 필요.
        
        # 일반화된 차원 변경 로직: 마지막 차원(C)을 두 번째 위치로 이동
        dims = list(range(targets_one_hot.ndim))
        # [0(N), 마지막 차원(C)] + [나머지 공간 차원들]
        dims = [0, dims[-1]] + dims[1:-1]
        targets_one_hot = targets_one_hot.permute(dims).float()
    else:
        # 이미 one-hot 이거나 이진 분류(C=1)인 경우
        targets_one_hot = targets.float()
        # 이진 분류 시 입력과 타겟의 차원 수 맞추기 (예: (N, H, W) -> (N, 1, H, W))
        if targets_one_hot.dim() < inputs.dim():
             targets_one_hot = targets_one_hot.unsqueeze(1)

    # 3. 공간 차원 평탄화 (N, C, Spatial)
    probs = probs.view(N, C, -1)
    targets_one_hot = targets_one_hot.view(N, C, -1)
    
    return probs, targets_one_hot

# --- 4. Dice Loss for Classification (Soft F1-Score Loss) ---

class DiceLoss(nn.Module):
    """
    이미지 분류 작업을 위한 Dice Loss (Soft F1-Score Loss) 구현.
    
    smooth: 수치 안정성을 위한 값 (Default: 1e-6).
    activation: 입력 로짓에 적용할 활성화 함수 ('softmax', 'sigmoid', 'none').
    average: 평균화 방식 ('macro' 또는 'micro'). 불균형 데이터에는 'macro' 권장.
    """
    def __init__(self, smooth=1e-6, activation='softmax', average='macro'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.activation = activation
        self.average = average

    def forward(self, inputs, targets):
        
        probs, targets_format = _prepare_classification_inputs(inputs, targets, self.activation)
        
        # 합산 차원 결정
        if self.average == 'micro':
            # Micro-average: 모든 차원(N, C)에 대해 합산하여 전역 메트릭 계산
            # (N, C) -> 스칼라
            sum_dims = tuple(range(probs.dim()))
        elif self.average == 'macro':
            # Macro-average: 배치 차원(N, dim=0)에 대해서만 합산하여 클래스별 메트릭 계산
            # (N, C) -> (C)
            sum_dims = 0
        else:
            raise ValueError("Average must be 'micro' or 'macro'")

        # 1. TP(Intersection) 및 Cardinality 계산
        # TP (Soft True Positive): Σ(P * G)
        TP = (probs * targets_format).sum(dim=sum_dims)
        
        # Cardinality (크기 합): ΣP + ΣG
        # 참고: 일부 구현에서는 ΣP^2 + ΣG^2를 사용하기도 하지만(Generalized Dice Loss), 
        # 분류에서는 ΣP + ΣG가 F1-Score와 더 직접적으로 연관됩니다.
        cardinality = probs.sum(dim=sum_dims) + targets_format.sum(dim=sum_dims)
        
        # 2. Dice score 계산: (2 * TP + smooth) / (Cardinality + smooth)
        dice_score = (2. * TP + self.smooth) / (cardinality + self.smooth)
        
        # 3. Dice Loss = 1 - Dice Score
        dice_loss = 1 - dice_score

        # 4. 최종 Reduction
        if self.average == 'macro':
            # Macro의 경우 클래스별 Loss (C)가 계산되었으므로 평균을 취함
            return dice_loss.mean()
        else:
            # Micro의 경우 이미 단일 스칼라 값임
            return dice_loss

# --- 5. Tversky Loss for Classification ---

class TverskyLoss(nn.Module):
    """
    이미지 분류 작업을 위한 Tversky Loss 구현. F1-score의 일반화된 형태.
    
    alpha: False Positive(FP)에 대한 페널티 가중치 (Precision 관련).
    beta: False Negative(FN)에 대한 페널티 가중치 (Recall 관련).
    (alpha=0.5, beta=0.5이면 Dice Loss와 동일)
    average: 평균화 방식 ('macro' 또는 'micro').
    """
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6, activation='softmax', average='macro'):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.activation = activation
        self.average = average

    def forward(self, inputs, targets):
        
        probs, targets_format = _prepare_classification_inputs(inputs, targets, self.activation)

        # 합산 차원 결정 (DiceLoss와 동일)
        if self.average == 'micro':
            sum_dims = tuple(range(probs.dim()))
        elif self.average == 'macro':
            sum_dims = 0
        else:
            raise ValueError("Average must be 'micro' or 'macro'")

        # 1. TP, FP, FN 계산
        # TP (Soft True Positives): Σ(P * G)
        TP = (probs * targets_format).sum(dim=sum_dims)
        
        # FP (Soft False Positives): Σ(P * (1-G))
        FP = (probs * (1 - targets_format)).sum(dim=sum_dims)
        
        # FN (Soft False Negatives): Σ((1-P) * G)
        FN = ((1 - probs) * targets_format).sum(dim=sum_dims)
        
        # 2. Tversky 지수 계산: (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
        tversky_index = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        # 3. Tversky Loss = 1 - Tversky 지수
        tversky_loss = 1 - tversky_index

        # 4. 최종 Reduction
        if self.average == 'macro':
            # 클래스별 Loss의 평균
            return tversky_loss.mean()
        else:
            return tversky_loss