# kd_research/losses/builder.py
import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from .unified_kd_gauss import PrototypeKDLoss
except ImportError:
    pass

try:
    from .unified_kd_geometric import UnifiedCentersKDLoss
except ImportError:
    pass

class VanillaKDLoss(nn.Module):
    """
    Hinton's Standard Knowledge Distillation Loss.
    """
    def __init__(self, T=4.0, kd_lambda=0.9):
        super().__init__()
        self.T = T
        self.kd_lambda = kd_lambda

    def forward(self, logits_s, logits_t, *args, **kwargs):

        # 1. Standard KD Loss
        log_p_s = F.log_softmax(logits_s / self.T, dim=1)
        p_t = F.softmax(logits_t / self.T, dim=1)
        

        kd_loss = F.kl_div(log_p_s, p_t, reduction='batchmean') * (self.T**2)
        
        # 2. Weughts
        total_loss = kd_loss * self.kd_lambda
        
        # 3. Return Dic
        return {
            "loss_total": total_loss,
            "loss_kd": kd_loss
        }

def build_distill_loss(cfg, dim_s=None, dim_t=None, num_classes=None):

    if hasattr(cfg.distillation, 'method'):
        method = cfg.distillation.method.lower()
    else:
        method = "hinton" 

    if method == "hinton":
        T = getattr(cfg.distillation, 'T', 4.0)
        kd_lambda = getattr(cfg.distillation, 'kd_lambda', 0.9)
        return VanillaKDLoss(T=T, kd_lambda=kd_lambda)
        
    elif method == "unified_gauss":
        from .unified_kd_gauss import PrototypeKDLoss
        return PrototypeKDLoss(
            num_classes=num_classes,
            dim_s=dim_s,
            dim_t=dim_t,
            k=getattr(cfg.distillation, 'proto_k', 16),  #  proto_k
            proj_dim=getattr(cfg.distillation, 'proj_dim', 128),
            init_tau=getattr(cfg.distillation, 'sim_tau', 1.0),
            learnable_tau=getattr(cfg.distillation, 'learnable_tau', True),
            kd_lambda=cfg.distillation.kd_lambda,
            sim_lambda=cfg.distillation.sim_lambda,
            ortho_lambda=getattr(cfg.distillation, 'ortho_lambda', 1e-3),
            balance_lambda=getattr(cfg.distillation, 'balance_lambda', 2e-4),
            T=getattr(cfg.distillation, 'T', 4.0)
        )
    elif method == "unified_geometric":
        return UnifiedCentersKDLoss(
            num_classes=num_classes,
            dim_s=dim_s,
            dim_t=dim_t,
            k=getattr(cfg.distillation, 'center_k', 16),
            proj_dim=getattr(cfg.distillation, 'proj_dim', 128),
            init_tau=getattr(cfg.distillation, 'sim_tau', 1.0),
            kd_lambda=cfg.distillation.kd_lambda,
            sim_lambda=cfg.distillation.sim_lambda,
            T=getattr(cfg.distillation, 'T', 4.0)
        )
    else:
        raise ValueError(f"Unknown method: {method}")


    