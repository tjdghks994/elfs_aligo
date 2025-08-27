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