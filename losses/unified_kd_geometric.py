# kd_research/losses/unified_kd_geometric.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class UnifiedCentersKDLoss(nn.Module):
    def __init__(self,
                 num_classes: int,
                 dim_s: int,
                 dim_t: int,
                 k: int = 16,
                 proj_dim: int = 128,
                 init_tau: float = 1.0,
                 kd_lambda: float = 1.0,
                 sim_lambda: float = 1.0,
                 T: float = 4.0):
        super().__init__()
        self.k = k
        self.kd_lambda = kd_lambda
        self.sim_lambda = sim_lambda
        self.T = T

        self.proj_S = nn.Linear(dim_s, proj_dim)
        self.proj_T = nn.Linear(dim_t, proj_dim)
        
        # ETF Initialization
        etf_k = self._generate_etf(k, proj_dim)
        self.register_buffer("K", etf_k)

        self.register_buffer("tau", torch.tensor(float(init_tau)))

    def _generate_etf(self, k: int, d: int) -> torch.Tensor:
        identity = torch.eye(k)
        one_k = torch.ones(k, k) / k
        etf_base = identity - one_k
        u, s, v = torch.svd(etf_base)
        etf_vecs = u @ torch.diag(s)
        
        if d > k:
            padding = torch.zeros(k, d - k)
            etf_vecs = torch.cat([etf_vecs, padding], dim=1)
        elif d < k:
            etf_vecs = etf_vecs[:, :d]
        return F.normalize(etf_vecs, p=2, dim=1)

    def _row_norm(self, x):
        return F.normalize(x, dim=1)

    def get_PS_PT(self, W_s, W_t):
        W_s_proj = self._row_norm(self.proj_S(W_s))
        sim_s = W_s_proj @ self.K.t()
        P_S = F.softmax(sim_s / self.tau, dim=1)

        with torch.no_grad():
            W_t_proj = self._row_norm(self.proj_T(W_t))
            sim_t = W_t_proj @ self.K.t()
            P_T = F.softmax(sim_t / self.tau, dim=1)
            
        return P_S, P_T

    def forward(self, 
                logits_s: torch.Tensor, 
                logits_t: torch.Tensor, 
                W_s: torch.Tensor, 
                W_t: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        eps = 1e-9
        P_S, P_T = self.get_PS_PT(W_s, W_t)
        loss_sim = torch.sum(P_S * (torch.log(P_S + eps) - torch.log(P_T + eps)), dim=1).mean()

        log_p_s = F.log_softmax(logits_s / self.T, dim=1)
        p_t = F.softmax(logits_t / self.T, dim=1)
        loss_kd = F.kl_div(log_p_s, p_t, reduction='batchmean') * (self.T**2)

        total_loss = self.kd_lambda * loss_kd + self.sim_lambda * loss_sim

        return {
            "loss_total": total_loss,
            "loss_kd": loss_kd,
            "loss_sim": loss_sim,
            "K_matrix": self.K
        }