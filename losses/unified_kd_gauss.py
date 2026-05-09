# kd_research/losses/unified_kd_gauss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

class PrototypeKDLoss(nn.Module):
    def __init__(self,
                 num_classes: int,
                 dim_s: int,
                 dim_t: int,
                 k: int = 16,
                 proj_dim: int = 128,
                 init_tau: float = 1.0,
                 learnable_tau: bool = True,
                 kd_lambda: float = 1.0,
                 sim_lambda: float = 1.0,
                 ortho_lambda: float = 1e-3,
                 balance_lambda: float = 2e-4,
                 T: float = 4.0):
        super().__init__()
        self.k = k
        self.kd_lambda = kd_lambda
        self.sim_lambda = sim_lambda
        self.ortho_lambda = ortho_lambda
        self.balance_lambda = balance_lambda
        self.T = T

        self.proj_S = nn.Linear(dim_s, proj_dim)
        self.proj_T = nn.Linear(dim_t, proj_dim)
        
        # Learnable Prototypes (Random Gaussian Init)
        self.K = nn.Parameter(torch.randn(k, proj_dim) * 0.02)

        if learnable_tau:
            self.tau = nn.Parameter(torch.tensor(float(init_tau)))
        else:
            self.register_buffer("tau", torch.tensor(float(init_tau)))

    def _row_norm(self, x):
        return F.normalize(x, dim=1)

    def get_PS_PT(self, W_s, W_t):
        if isinstance(self.tau, torch.Tensor):
            tau_val = torch.clamp(self.tau, 1e-3, 10.0)
            tau_ps = tau_val
            tau_pt = tau_val.detach()
        else:
            tau_ps = self.tau
            tau_pt = self.tau

        W_s_proj = self._row_norm(self.proj_S(W_s))
        K_norm   = self._row_norm(self.K)
        sim_s    = W_s_proj @ K_norm.t()
        P_S      = F.softmax(sim_s / tau_ps, dim=1)

        with torch.no_grad():
            W_t_proj = self._row_norm(self.proj_T(W_t))
            sim_t    = W_t_proj @ K_norm.t()
            P_T      = F.softmax(sim_t / tau_pt, dim=1)
            
        return P_S, P_T

    def forward(self, 
                logits_s: torch.Tensor, 
                logits_t: torch.Tensor, 
                W_s: torch.Tensor, 
                W_t: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        eps = 1e-9
        P_S, P_T = self.get_PS_PT(W_s, W_t)
        sim_loss = torch.sum(P_S * (torch.log(P_S + eps) - torch.log(P_T + eps)), dim=1).mean()

        log_p_s = F.log_softmax(logits_s / self.T, dim=1)
        p_t     = F.softmax(logits_t / self.T, dim=1)
        kd_loss = F.kl_div(log_p_s, p_t, reduction='batchmean') * (self.T**2)

        Kn = self._row_norm(self.K)
        I  = torch.eye(self.k, device=logits_s.device)
        ortho_loss = ((Kn @ Kn.t() - I) ** 2).mean()
        
        col_mean = P_T.mean(dim=0)
        balance_loss = ((col_mean - 1.0 / self.k) ** 2).mean()

        total_loss = (self.kd_lambda * kd_loss + 
                      self.sim_lambda * sim_loss + 
                      self.ortho_lambda * ortho_loss + 
                      self.balance_lambda * balance_loss)

        return {
            "loss_total": total_loss,
            "loss_kd": kd_loss,
            "loss_sim": sim_loss,
            "loss_ortho": ortho_loss,
            "loss_balance": balance_loss,
            "K_matrix": self.K
        }