import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# SCTMM
class SemanticAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.lambda_sem = nn.Parameter(torch.ones(1))

        assert self.head_dim * num_heads == embed_dim, \
            f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}"

    def forward(self, T_cat, w_sem):
        if next(self.parameters()).device != T_cat.device:
            self.to(T_cat.device)

        b, l, d = T_cat.shape
        Q = self.q_proj(T_cat).reshape(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(T_cat).reshape(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(T_cat).reshape(b, l, self.num_heads, self.head_dim).transpose(1, 2)

        A_score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        semantic_bias = w_sem.unsqueeze(1).unsqueeze(2)
        semantic_bias = semantic_bias * w_sem.unsqueeze(1).unsqueeze(3)
        A_score = A_score + self.lambda_sem * semantic_bias

        A_attn = F.softmax(A_score, dim=-1)
        A_attn = self.dropout(A_attn)
        T_attn = torch.matmul(A_attn, V).transpose(1, 2).reshape(b, l, d)
        T_attn = self.out_proj(T_attn)
        T_attn = self.norm(T_cat + self.dropout(T_attn))
        return T_attn


class SCTMM(nn.Module):
    def __init__(self, L, C, num_heads=8, hidden_dim_scale=2, dropout=0.1):
        super().__init__()
        self.L = L
        self.C = C
        self.num_heads = num_heads
        self.hidden_dim_scale = hidden_dim_scale
        self.dropout = dropout

        embed_dim = 3 * C
        hidden_dim = hidden_dim_scale * embed_dim

        self.Psi = nn.ModuleList([
            nn.Linear(C, embed_dim),
            nn.Linear(C, embed_dim),
            nn.Linear(C, embed_dim)
        ])

        self.T_center = nn.Parameter(torch.randn(1, 1, embed_dim))
        nn.init.xavier_uniform_(self.T_center)

        self.semantic_attention = SemanticAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        self.reconstruction_mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim)
        )

        self.omega = nn.Parameter(torch.ones(3))

    def calculate_semantic_matching(self, projected_tokens):
        b, _, _ = projected_tokens[0].shape
        T_center = self.T_center.expand(b, -1, -1).to(projected_tokens[0].device)

        matching_scores = []
        for T_m in projected_tokens:
            T_m_norm = F.normalize(T_m, dim=-1)
            T_center_norm = F.normalize(T_center, dim=-1)
            score = torch.matmul(T_m_norm, T_center_norm.transpose(-2, -1)).squeeze(-1)
            matching_scores.append(score)

        w_sem = torch.stack(matching_scores, dim=-1).mean(dim=-1)
        w_sem = (w_sem - w_sem.min(dim=-1, keepdim=True)[0]) / (
            w_sem.max(dim=-1, keepdim=True)[0] -
            w_sem.min(dim=-1, keepdim=True)[0] + 1e-8
        )
        return w_sem

    def forward(self, T_s, T_r, T_a):
        assert T_s.shape == T_r.shape == T_a.shape, "The three modality token shapes must be identical"
        b, L, C = T_s.shape
        assert L == self.L and C == self.C, \
            f"Input shape does not match initialization: init(L={self.L}, C={self.C}), current(L={L}, C={C})"

        device = T_s.device
        if next(self.parameters()).device != device:
            self.to(device)

        projected_tokens = [proj(T_m) for proj, T_m in zip(self.Psi, [T_s, T_r, T_a])]
        w_sem = self.calculate_semantic_matching(projected_tokens)

        omega = F.softmax(self.omega, dim=0).to(device)
        T_agg = (
            omega[0] * projected_tokens[0] +
            omega[1] * projected_tokens[1] +
            omega[2] * projected_tokens[2]
        )

        T_cat = torch.cat([T_s, T_r, T_a], dim=-1)
        T_attn = self.semantic_attention(T_cat, w_sem)
        T_g = self.reconstruction_mlp(torch.cat([T_agg, T_attn], dim=-1))
        return T_g


# DCSFM
class ChannelEstimation(nn.Module):
    def __init__(self, C, R=8):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.M_R = nn.Sequential(
            nn.Linear(C, C // R),
            nn.LayerNorm(C // R),
            nn.GELU(),
            nn.Linear(C // R, C),
            nn.Sigmoid()
        )

    def forward(self, F_sp_hat_m):
        b, c, _, _ = F_sp_hat_m.shape
        alpha_m = self.M_R(self.gap(F_sp_hat_m).view(b, c)).view(b, c, 1, 1)
        return alpha_m


class CrossModalSupplement(nn.Module):
    def __init__(self, C, R=8):
        super().__init__()
        self.spatial_fusion = nn.Sequential(
            nn.Conv2d(3 * C, C, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, C),
            nn.GELU(),
            nn.Conv2d(C, C, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.supp_transform = nn.Sequential(
            nn.Conv2d(2 * C, C, kernel_size=1, bias=False),
            nn.GroupNorm(4, C),
            nn.GELU()
        )

        self.out_proj = nn.Conv2d(C, C, kernel_size=1)
        self.norm = nn.GroupNorm(8, C)

    def forward(self, F_lo_m, F_hi_p, F_hi_q):
        F_hi_pq = torch.cat([F_hi_p, F_hi_q], dim=1)
        F_supp_source = self.supp_transform(F_hi_pq)

        F_cat = torch.cat([F_lo_m, F_hi_p, F_hi_q], dim=1)
        gamma_m = self.spatial_fusion(F_cat)
        F_supp_m = F_lo_m + gamma_m * F_supp_source

        F_supp_m = self.out_proj(F_supp_m)
        F_supp_m = self.norm(F_supp_m + F_lo_m)
        return F_supp_m


class AdaptiveFusion(nn.Module):
    def __init__(self, C, R=8):
        super().__init__()
        self.spatial_weight = nn.Sequential(
            nn.Conv2d(3 * C, 3, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

        self.channel_weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(3 * C, 3 * C // R, 1),
            nn.ReLU(),
            nn.Conv2d(3 * C // R, 3 * C, 1),
            nn.Sigmoid()
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(3 * C, 3 * C, kernel_size=3, padding=1, bias=False, groups=3),
            nn.Conv2d(3 * C, 3 * C, kernel_size=1, bias=False),
            nn.GroupNorm(8, 3 * C),
            nn.GELU()
        )

    def forward(self, F_tilde_s, F_tilde_r, F_tilde_a):
        F_cat = torch.cat([F_tilde_s, F_tilde_r, F_tilde_a], dim=1)

        spatial_weight = self.spatial_weight(F_cat)
        w_s_spa, w_r_spa, w_a_spa = (
            spatial_weight[:, 0:1],
            spatial_weight[:, 1:2],
            spatial_weight[:, 2:3]
        )

        channel_weight = self.channel_weight(F_cat)
        C = F_tilde_s.shape[1]
        w_s_ch = channel_weight[:, :C]
        w_r_ch = channel_weight[:, C:2 * C]
        w_a_ch = channel_weight[:, 2 * C:]

        F_s = F_tilde_s * w_s_spa * w_s_ch
        F_r = F_tilde_r * w_r_spa * w_r_ch
        F_a = F_tilde_a * w_a_spa * w_a_ch

        F_weighted = torch.cat([F_s, F_r, F_a], dim=1)
        F_g = self.fusion(F_weighted)
        F_g = F_g + F_cat
        return F_g


class DCSFM(nn.Module):
    def __init__(self, C, R=8):
        super().__init__()
        self.channel_estimation_s = ChannelEstimation(C, R)
        self.channel_estimation_r = ChannelEstimation(C, R)
        self.channel_estimation_a = ChannelEstimation(C, R)

        self.cross_modal_supplement_s = CrossModalSupplement(C, R)
        self.cross_modal_supplement_r = CrossModalSupplement(C, R)
        self.cross_modal_supplement_a = CrossModalSupplement(C, R)

        self.adaptive_fusion = AdaptiveFusion(C, R)

    def forward(self, F_sp_hat_s, F_sp_hat_r, F_sp_hat_a):
        alpha_s = self.channel_estimation_s(F_sp_hat_s)
        alpha_r = self.channel_estimation_r(F_sp_hat_r)
        alpha_a = self.channel_estimation_a(F_sp_hat_a)

        F_hi_s = F_sp_hat_s * alpha_s
        F_lo_s = F_sp_hat_s * (1 - alpha_s)
        F_hi_r = F_sp_hat_r * alpha_r
        F_lo_r = F_sp_hat_r * (1 - alpha_r)
        F_hi_a = F_sp_hat_a * alpha_a
        F_lo_a = F_sp_hat_a * (1 - alpha_a)

        F_supp_s = self.cross_modal_supplement_s(F_lo_s, F_hi_r, F_hi_a) + F_hi_s
        F_supp_r = self.cross_modal_supplement_r(F_lo_r, F_hi_s, F_hi_a) + F_hi_r
        F_supp_a = self.cross_modal_supplement_a(F_lo_a, F_hi_s, F_hi_r) + F_hi_a

        F_g = self.adaptive_fusion(F_supp_s, F_supp_r, F_supp_a)
        return F_g
