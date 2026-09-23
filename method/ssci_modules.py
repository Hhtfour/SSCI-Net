"""Interaction modules for SSCI-Net.

Names follow the manuscript terminology. The active computations are kept from
``dd(2).py``. Unused alternative fusion designs have been removed.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """Legacy layer retained only for compatibility with existing checkpoints."""

    def __init__(self, in_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.d_k = in_channels // 8

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, height * width)

        energy = torch.bmm(query, key)
        energy = energy / torch.sqrt(
            torch.tensor(self.d_k, dtype=torch.float32, device=x.device)
        )

        attention = torch.softmax(energy, dim=-1)
        value = self.value_conv(x).view(batch_size, -1, height * width)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        out = self.gamma * out + x
        return out


class SemanticEnhancedAttention(nn.Module):
    """Semantic-bias attention used by the SCTMM."""

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

        # Learnable semantic-bias scale (lambda in the manuscript).
        self.semantic_scale = nn.Parameter(torch.ones(1))

        assert self.head_dim * num_heads == embed_dim, (
            f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}"
        )

    def forward(self, x, semantic_matching_weights):
        if next(self.parameters()).device != x.device:
            self.to(x.device)

        batch_size, token_len, embed_dim = x.shape

        q = self.q_proj(x).reshape(
            batch_size, token_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(x).reshape(
            batch_size, token_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(x).reshape(
            batch_size, token_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        semantic_weights_expanded = semantic_matching_weights.unsqueeze(1).unsqueeze(2)
        semantic_weights_expanded = (
            semantic_weights_expanded
            * semantic_matching_weights.unsqueeze(1).unsqueeze(3)
        )
        attn_scores = (
            attn_scores + self.semantic_scale * semantic_weights_expanded
        )

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_out = torch.matmul(attn_weights, v).transpose(1, 2).reshape(
            batch_size, token_len, embed_dim
        )
        attn_out = self.out_proj(attn_out)
        attn_out = self.norm(x + self.dropout(attn_out))
        return attn_out


class SCTMM(nn.Module):
    """Semantic Center Token Matching Module (SCTMM)."""

    def __init__(self, l, n, num_heads=8, hidden_dim_scale=2, dropout=0.1):
        super().__init__()
        self.l = l
        self.n = n
        self.num_heads = num_heads
        self.hidden_dim_scale = hidden_dim_scale
        self.dropout = dropout

        three_n = 3 * n
        hidden_dim = hidden_dim_scale * three_n

        # Modality-specific projections: C -> 3C.
        self.modal_projs = nn.ModuleList([
            nn.Linear(n, three_n),
            nn.Linear(n, three_n),
            nn.Linear(n, three_n),
        ])

        # Learnable global semantic center.
        self.global_semantic_center = nn.Parameter(torch.randn(1, 1, three_n))
        nn.init.xavier_uniform_(self.global_semantic_center)

        self.semantic_attn = SemanticEnhancedAttention(
            embed_dim=three_n,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Deep feature reconstruction MLP.
        self.adaptive_fusion = nn.Sequential(
            nn.Linear(2 * three_n, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, three_n),
        )

        # Learnable modality weights.
        self.fusion_weights = nn.Parameter(torch.ones(3))

    def calculate_semantic_matching(self, projected_modals):
        batch_size, _, _ = projected_modals[0].shape
        global_center = self.global_semantic_center.expand(batch_size, -1, -1).to(
            projected_modals[0].device
        )

        matching_scores = []
        for modal_feat in projected_modals:
            modal_feat_norm = F.normalize(modal_feat, dim=-1)
            center_norm = F.normalize(global_center, dim=-1)
            score = torch.matmul(
                modal_feat_norm, center_norm.transpose(-2, -1)
            ).squeeze(-1)
            matching_scores.append(score)

        semantic_matching_weights = torch.stack(matching_scores, dim=-1).mean(dim=-1)
        semantic_matching_weights = (
            semantic_matching_weights
            - semantic_matching_weights.min(dim=-1, keepdim=True)[0]
        ) / (
            semantic_matching_weights.max(dim=-1, keepdim=True)[0]
            - semantic_matching_weights.min(dim=-1, keepdim=True)[0]
            + 1e-8
        )
        return semantic_matching_weights

    def forward(self, sar_tokens, amsr2_tokens, auxiliary_tokens):
        assert sar_tokens.shape == amsr2_tokens.shape == auxiliary_tokens.shape, (
            "The three modality token tensors must have identical shapes"
        )
        batch_size, token_len, channel_dim = sar_tokens.shape
        assert token_len == self.l and channel_dim == self.n, (
            f"Input shape does not match initialization: "
            f"init(l={self.l}, n={self.n}), current(l={token_len}, n={channel_dim})"
        )

        device = sar_tokens.device
        if next(self.parameters()).device != device:
            self.to(device)

        # The order is SAR, AMSR2, auxiliary, matching the SSCI-Net forward path.
        modal_tokens = [sar_tokens, amsr2_tokens, auxiliary_tokens]
        projected_modals = [
            proj(modal)
            for proj, modal in zip(self.modal_projs, modal_tokens)
        ]

        semantic_matching_weights = self.calculate_semantic_matching(projected_modals)

        fusion_weights = F.softmax(self.fusion_weights, dim=0).to(device)
        aggregated_modal = (
            fusion_weights[0] * projected_modals[0]
            + fusion_weights[1] * projected_modals[1]
            + fusion_weights[2] * projected_modals[2]
        )

        original_cat = torch.cat(modal_tokens, dim=-1)
        semantic_attn_out = self.semantic_attn(
            original_cat, semantic_matching_weights
        )

        combined = torch.cat([aggregated_modal, semantic_attn_out], dim=-1)
        final_out = self.adaptive_fusion(combined)
        return final_out


class ChannelWeightEstimation(nn.Module):
    """Channel-estimation submodule of DCSFM."""

    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.GELU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch_size, channels = x.shape[:2]
        global_feat = self.gap(x).view(batch_size, channels)
        channel_weight = self.mlp(global_feat)
        return channel_weight


class CrossModalSupplement(nn.Module):
    """Cross-modal supplement submodule of DCSFM."""

    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(3 * in_channels, in_channels // reduction),
            nn.GELU(),
            nn.Linear(in_channels // reduction, 2 * in_channels),
        )
        self.res_proj = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, target_low, supp1_high, supp2_high):
        batch_size, channels = target_low.shape[:2]

        target_low_gap = self.gap(target_low).view(batch_size, channels)
        supp1_high_gap = self.gap(supp1_high).view(batch_size, channels)
        supp2_high_gap = self.gap(supp2_high).view(batch_size, channels)

        mlp_input = torch.cat(
            [target_low_gap, supp1_high_gap, supp2_high_gap], dim=1
        )
        gamma = self.mlp(mlp_input).view(batch_size, 2, channels)
        gamma1, gamma2 = gamma[:, 0, :], gamma[:, 1, :]

        target_low_supplemented = (
            target_low
            + gamma1.unsqueeze(-1).unsqueeze(-1) * supp1_high
            + gamma2.unsqueeze(-1).unsqueeze(-1) * supp2_high
        )

        residual = self.res_proj(target_low)
        target_low_supplemented = target_low_supplemented.permute(0, 2, 3, 1)
        target_low_supplemented = self.norm(target_low_supplemented)
        target_low_supplemented = target_low_supplemented.permute(0, 3, 1, 2)
        target_low_supplemented = target_low_supplemented + residual
        return target_low_supplemented


class AdaptiveModalFusion(nn.Module):
    """Adaptive fusion submodule of DCSFM."""

    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.GELU(),
            nn.Linear(in_channels // reduction, 1),
        )
        self.final_res_proj = nn.Conv2d(
            3 * in_channels,
            3 * in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.final_norm = nn.LayerNorm(3 * in_channels)

    def forward(self, x1, x2, x3):
        batch_size = x1.shape[0]

        w1 = torch.sigmoid(
            self.mlp(self.gap(x1).view(batch_size, -1))
        ).view(batch_size, 1, 1, 1)
        w2 = torch.sigmoid(
            self.mlp(self.gap(x2).view(batch_size, -1))
        ).view(batch_size, 1, 1, 1)
        w3 = torch.sigmoid(
            self.mlp(self.gap(x3).view(batch_size, -1))
        ).view(batch_size, 1, 1, 1)

        weighted_fused = torch.cat([x1 * w1, x2 * w2, x3 * w3], dim=1)

        raw_concat = torch.cat([x1, x2, x3], dim=1)
        residual = self.final_res_proj(raw_concat)

        final_fused = weighted_fused + residual
        final_fused = final_fused.permute(0, 2, 3, 1)
        final_fused = self.final_norm(final_fused)
        final_fused = final_fused.permute(0, 3, 1, 2)
        return final_fused


class DCSFM(nn.Module):
    """Dynamic Cross-Modal Supplement Fusion Module (DCSFM).

    Note: legacy internal attribute names are intentionally retained so that
    existing checkpoints keep the same parameter keys. In the original code,
    the first DCSFM argument is the SAR branch even though several registered
    attribute names contain ``amsr``. Local variable names below reflect the
    actual branch semantics without changing the computation or state-dict keys.
    """

    def __init__(self, in_channels, reduction=8):
        super().__init__()

        # Legacy registration order/keys retained for checkpoint compatibility.
        self.cwe_amsr = ChannelWeightEstimation(in_channels, reduction)
        self.cwe_sar = ChannelWeightEstimation(in_channels, reduction)
        self.cwe_aul = ChannelWeightEstimation(in_channels, reduction)

        self.cms_amsr = CrossModalSupplement(in_channels, reduction)
        self.cms_sar = CrossModalSupplement(in_channels, reduction)
        self.cms_aul = CrossModalSupplement(in_channels, reduction)

        self.refine_res_proj_amsr = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.refine_res_proj_sar = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.refine_res_proj_aul = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.refine_norm = nn.LayerNorm(in_channels)

        self.amf = AdaptiveModalFusion(in_channels, reduction)

    def forward(self, sar_feature, amsr2_feature, auxiliary_feature):
        # Stage 1: channel estimation and soft high/low decomposition.
        # These module accesses preserve the original learned branch ordering.
        alpha_s = self.cwe_amsr(sar_feature)
        alpha_r = self.cwe_sar(amsr2_feature)
        alpha_a = self.cwe_aul(auxiliary_feature)

        sar_high = sar_feature * alpha_s.unsqueeze(-1).unsqueeze(-1)
        sar_low = sar_feature * (1 - alpha_s).unsqueeze(-1).unsqueeze(-1)
        amsr2_high = amsr2_feature * alpha_r.unsqueeze(-1).unsqueeze(-1)
        amsr2_low = amsr2_feature * (1 - alpha_r).unsqueeze(-1).unsqueeze(-1)
        auxiliary_high = auxiliary_feature * alpha_a.unsqueeze(-1).unsqueeze(-1)
        auxiliary_low = auxiliary_feature * (1 - alpha_a).unsqueeze(-1).unsqueeze(-1)

        # Stage 2: cross-modal supplementation.
        sar_low_supp = self.cms_amsr(sar_low, amsr2_high, auxiliary_high)
        amsr2_low_supp = self.cms_sar(amsr2_low, sar_high, auxiliary_high)
        auxiliary_low_supp = self.cms_aul(auxiliary_low, sar_high, amsr2_high)

        # Stage 3: branch reconstruction with residual information.
        sar_refined = sar_high + sar_low_supp
        amsr2_refined = amsr2_high + amsr2_low_supp
        auxiliary_refined = auxiliary_high + auxiliary_low_supp

        sar_residual = self.refine_res_proj_amsr(sar_feature) + sar_feature
        amsr2_residual = self.refine_res_proj_sar(amsr2_feature) + amsr2_feature
        auxiliary_residual = (
            self.refine_res_proj_aul(auxiliary_feature) + auxiliary_feature
        )

        sar_refined = sar_refined + sar_residual
        sar_refined = sar_refined.permute(0, 2, 3, 1)
        sar_refined = self.refine_norm(sar_refined)
        sar_refined = sar_refined.permute(0, 3, 1, 2)

        amsr2_refined = amsr2_refined + amsr2_residual
        amsr2_refined = amsr2_refined.permute(0, 2, 3, 1)
        amsr2_refined = self.refine_norm(amsr2_refined)
        amsr2_refined = amsr2_refined.permute(0, 3, 1, 2)

        auxiliary_refined = auxiliary_refined + auxiliary_residual
        auxiliary_refined = auxiliary_refined.permute(0, 2, 3, 1)
        auxiliary_refined = self.refine_norm(auxiliary_refined)
        auxiliary_refined = auxiliary_refined.permute(0, 3, 1, 2)

        # Stage 4: adaptive modality fusion.
        fused_feat = self.amf(sar_refined, amsr2_refined, auxiliary_refined)
        return fused_feat


# Backward-compatible class names used by the original experiment scripts.
CrossModalFusion_new = SCTMM
MultiModalDynamicFusion = DCSFM
