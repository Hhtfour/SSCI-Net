import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ssci_modules import SCTMM, DCSFM


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TwoLayerFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


class TwoLayerConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.relu(x + identity)
        return x


class ResNet18Backbone(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256, stride=1),
            ResidualBlock(256, 256),
        )
        self.layer4 = nn.Sequential(
            ResidualBlock(256, 512, stride=1),
            ResidualBlock(512, 512),
        )


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth=1, heads=8, mlp_dim=None, dropout=0.0):
        super().__init__()
        if mlp_dim is None:
            mlp_dim = 2 * dim
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, x):
        return self.encoder(x)


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_dim=None, dropout=0.0):
        super().__init__()
        if mlp_dim is None:
            mlp_dim = 2 * dim

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, Q_spa, T_g):
        Q_norm = self.norm_q(Q_spa)
        T_norm = self.norm_kv(T_g)
        attn_out, _ = self.attn(Q_norm, T_norm, T_norm, need_weights=False)
        Q_spa = Q_spa + attn_out
        Q_spa = Q_spa + self.ffn(self.norm_ffn(Q_spa))
        return Q_spa


class TransformerFusion(nn.Module):
    def __init__(self, dim, depth=1, heads=8, mlp_dim=None, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(
                dim=dim,
                heads=heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

    def forward(self, Q_spa, T_g):
        for block in self.blocks:
            Q_spa = block(Q_spa, T_g)
        return Q_spa


# Multimodal Feature Extractor
class MultimodalFeatureExtractor(nn.Module):
    def __init__(
        self,
        input_nc,
        backbone_in_channels=24,
        resnet_stages_num=5,
        if_upsample_2x=True,
    ):
        super().__init__()
        self.resnet = ResNet18Backbone(backbone_in_channels)
        self.upsamplex2 = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False
        )
        self.resnet_stages_num = resnet_stages_num
        self.if_upsample_2x = if_upsample_2x

        if resnet_stages_num == 5:
            layers = 512
        elif resnet_stages_num == 4:
            layers = 256
        elif resnet_stages_num == 3:
            layers = 128
        else:
            raise NotImplementedError

        self.conv_pred = nn.Conv2d(
            layers, input_nc * 3, kernel_size=3, padding=1
        )

    def forward_single(self, X_cat):
        X_cat = self.resnet.conv1(X_cat)
        X_cat = self.resnet.bn1(X_cat)
        X_cat = self.resnet.relu(X_cat)
        X_cat = self.resnet.maxpool(X_cat)

        X_4 = self.resnet.layer1(X_cat)
        X_8 = self.resnet.layer2(X_4)

        if self.resnet_stages_num > 3:
            X_8 = self.resnet.layer3(X_8)

        if self.resnet_stages_num == 5:
            X_8 = self.resnet.layer4(X_8)
        elif self.resnet_stages_num > 5:
            raise NotImplementedError

        if self.if_upsample_2x:
            X_sp_cat = self.upsamplex2(X_8)
        else:
            X_sp_cat = X_8

        X_sp_cat = self.conv_pred(X_sp_cat)
        return X_sp_cat


# SSCI-Net
class SSCINet(MultimodalFeatureExtractor):
    def __init__(
        self,
        input_nc=16,
        output_nc=2,
        with_pos='learned',
        pretrained=False,
        resnet_stages_num=5,
        token_len=10,
        enc_depth=1,
        dec_depth=1,
        dim_head=64,
        decoder_dim_head=64,
        tokenizer=True,
        if_upsample_2x=True,
        pool_mode='max',
        pool_size=2,
        backbone='resnet18',
        decoder_softmax=True,
        with_decoder_pos=None,
        input_size=256,
    ):
        del output_nc, dim_head, decoder_dim_head, pool_mode, backbone, decoder_softmax

        backbone_in_channels = 3 if pretrained else 24
        super().__init__(
            input_nc=input_nc,
            backbone_in_channels=backbone_in_channels,
            resnet_stages_num=resnet_stages_num,
            if_upsample_2x=if_upsample_2x,
        )

        self.token_len = token_len
        self.semantic_attention_map = nn.Conv2d(
            input_nc, self.token_len, kernel_size=1, padding=0, bias=False
        )

        if not tokenizer:
            self.token_len = pool_size * pool_size

        self.with_pos = with_pos
        if with_pos == 'learned':
            self.pos_embedding = nn.Parameter(
                torch.randn(1, self.token_len * 3, input_nc)
            )

        self.with_decoder_pos = with_decoder_pos
        feature_size = input_size // 4
        if with_decoder_pos == 'learned':
            self.pos_embedding_decoder = nn.Parameter(
                torch.randn(1, input_nc * 3, feature_size, feature_size)
            )

        self.transformer_encoder = TransformerEncoder(
            dim=input_nc,
            depth=enc_depth,
            heads=8,
            mlp_dim=input_nc * 4,
            dropout=0.0,
        )

        self.transformer_fusion = TransformerFusion(
            dim=input_nc * 3,
            depth=dec_depth,
            heads=8,
            mlp_dim=input_nc * 6,
            dropout=0.0,
        )

        self.classifier_SIC = TwoLayerConv2d(
            in_channels=input_nc * 3, out_channels=12
        )
        self.classifier_SOD = TwoLayerConv2d(
            in_channels=input_nc * 3, out_channels=7
        )
        self.classifier_FLOE = TwoLayerConv2d(
            in_channels=input_nc * 3, out_channels=8
        )

        self.embedding_s = DoubleConv(2, input_nc, input_nc // 2)
        self.embedding_r = DoubleConv(14, input_nc, input_nc // 2)
        self.embedding_a = DoubleConv(8, input_nc, input_nc // 2)
        self.shared_encoder = TwoLayerFeatureExtractor(
            input_nc, input_nc, input_nc * 2
        )
        self.down1 = Down(input_nc, input_nc * 2)
        self.down2 = Down(input_nc * 2, input_nc)

        # SCIM
        self.phi_vis = nn.Conv2d(input_nc * 2, input_nc * 4, 1)
        self.phi_sem = nn.Linear(input_nc, input_nc * 4)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=input_nc * 4,
            num_heads=8,
            batch_first=True,
            dropout=0.1,
        )
        self.phi_comp = nn.Sequential(
            nn.Conv2d(input_nc * 4, input_nc, 1),
            nn.BatchNorm2d(input_nc),
            nn.GELU(),
        )

        # Transformer Fusion
        self.E_spa = nn.Sequential(
            nn.Conv2d(
                input_nc * 3,
                input_nc * 3 // 2,
                kernel_size=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(input_nc * 3 // 2),
            nn.GELU(),
            nn.Conv2d(
                input_nc * 3 // 2,
                1,
                kernel_size=1,
                padding=0,
                bias=False,
            ),
            nn.Sigmoid(),
        )

        # SCTMM
        self.sctmm = SCTMM(L=self.token_len, C=input_nc)

        # DCSFM
        self.dcsfm = DCSFM(C=input_nc, R=8)

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=input_nc * 6,
                out_channels=input_nc * 3,
                kernel_size=2,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm2d(input_nc * 3),
            nn.GELU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=input_nc * 6,
                out_channels=input_nc * 3,
                kernel_size=2,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm2d(input_nc * 3),
            nn.GELU(),
        )

        self.max_i = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True
        )
        self.pretrained_input = pretrained
        self.input_projection = ConvBlock(
            24, 3, kernel_size=3, padding=1
        )

    def _semantic_token_generator(self, F_enc_m):
        b, c, _, _ = F_enc_m.shape
        A_m = self.semantic_attention_map(F_enc_m)
        A_m = A_m.view(b, self.token_len, -1).contiguous()
        A_m = torch.softmax(A_m, dim=-1)
        F_enc_flat = F_enc_m.view(b, c, -1).contiguous()
        T_init_m = torch.einsum('bln,bcn->blc', A_m, F_enc_flat)
        return T_init_m

    def _transformer_encoder(self, T_cat):
        if self.with_pos == 'learned':
            T_cat = T_cat + self.pos_embedding
        T_cat = self.transformer_encoder(T_cat)
        return T_cat

    def _transformer_fusion(self, F_spa, T_g):
        b, _, h, w = F_spa.shape

        if self.with_decoder_pos == 'learned':
            if self.pos_embedding_decoder.shape[-2:] != (h, w):
                pos = F.interpolate(
                    self.pos_embedding_decoder,
                    size=(h, w),
                    mode='bilinear',
                    align_corners=False,
                )
            else:
                pos = self.pos_embedding_decoder
            F_spa = F_spa + pos

        Q_spa = rearrange(F_spa, 'b c h w -> b (h w) c')
        F_tf = self.transformer_fusion(Q_spa, T_g)
        F_tf = rearrange(F_tf, 'b (h w) c -> b c h w', h=h, w=w)
        return F_tf

    # SCIM
    def _scim(self, F_enc_m, F_sp_m, T_m):
        B, C, H, W = F_enc_m.shape

        F_vis_m = torch.cat([F_enc_m, F_sp_m], dim=1)
        F_vis_m = self.phi_vis(F_vis_m)
        V_m = F_vis_m.flatten(2).transpose(1, 2)

        S_m = self.phi_sem(T_m)

        V_hat_m, _ = self.cross_attention(
            query=V_m, key=S_m, value=S_m, need_weights=False
        )
        S_hat_m, _ = self.cross_attention(
            query=S_m, key=V_hat_m, value=V_hat_m, need_weights=False
        )

        w_ch_m = S_hat_m.mean(dim=1)
        w_ch_m = F.softmax(w_ch_m, dim=-1).view(B, C * 4, 1, 1)

        V_hat_m = V_hat_m.transpose(1, 2).reshape(B, C * 4, H, W)
        F_weighted_m = V_hat_m * w_ch_m
        F_sp_hat_m = self.phi_comp(F_weighted_m)
        return F_sp_hat_m

    def forward(self, X_cat):
        if X_cat.ndim != 4 or X_cat.shape[1] != 24:
            raise ValueError(
                f'Expected input shape [B, 24, H, W], got {tuple(X_cat.shape)}'
            )

        X_raw = X_cat
        if self.pretrained_input:
            X_cat = self.input_projection(X_cat)

        X_sp_cat = self.forward_single(X_cat)

        X_s = X_raw[:, 0:2, :, :]
        X_r = X_raw[:, 4:18, :, :]
        X_a = torch.cat(
            [
                X_raw[:, 2:4, :, :],
                X_raw[:, 18:24, :, :],
            ],
            dim=1,
        )

        F_enc_s = self.shared_encoder(self.embedding_s(X_s))
        F_enc_r = self.shared_encoder(self.embedding_r(X_r))
        F_enc_a = self.shared_encoder(self.embedding_a(X_a))

        F_enc_s = self.down2(self.down1(F_enc_s))
        F_enc_r = self.down2(self.down1(F_enc_r))
        F_enc_a = self.down2(self.down1(F_enc_a))

        if X_sp_cat.shape[-2:] != F_enc_s.shape[-2:]:
            X_sp_cat = F.interpolate(
                X_sp_cat,
                size=F_enc_s.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )

        T_init_s = self._semantic_token_generator(F_enc_s)
        T_init_r = self._semantic_token_generator(F_enc_r)
        T_init_a = self._semantic_token_generator(F_enc_a)

        T_cat = torch.cat(
            [T_init_s, T_init_r, T_init_a], dim=1
        )
        T_cat = self._transformer_encoder(T_cat)
        T_s, T_r, T_a = T_cat.chunk(3, dim=1)

        F_sp_s, F_sp_r, F_sp_a = X_sp_cat.chunk(3, dim=1)

        F_sp_hat_s = self._scim(F_enc_s, F_sp_s, T_s)
        F_sp_hat_r = self._scim(F_enc_r, F_sp_r, T_r)
        F_sp_hat_a = self._scim(F_enc_a, F_sp_a, T_a)

        F_g = self.dcsfm(
            F_sp_hat_s, F_sp_hat_r, F_sp_hat_a
        )

        T_cat = torch.cat([T_s, T_r, T_a], dim=2)
        T_g = self.sctmm(T_s, T_r, T_a) + T_cat

        A_spa = self.E_spa(F_g)
        F_spa = A_spa * F_g + (1 - A_spa) * X_sp_cat

        F_tf = self._transformer_fusion(F_spa, T_g)

        if not self.if_upsample_2x:
            F_tf = self.upsamplex2(F_tf)

        F_tf = self.up1(torch.cat((F_tf, F_spa), dim=1))
        F_out = self.up2(
            torch.cat((F_tf, self.max_i(F_spa)), dim=1)
        )

        sic = self.classifier_SIC(F_out)
        sod = self.classifier_SOD(F_out)
        floe = self.classifier_FLOE(F_out)

        return {
            'SIC': sic,
            'SOD': sod,
            'FLOE': floe,
        }


if __name__ == '__main__':
    torch.manual_seed(42)

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    model = SSCINet(
        input_nc=16,
        token_len=10,
        with_pos='learned',
        pretrained=False,
        input_size=256,
    ).to(device)

    X = torch.randn(1, 24, 64, 64, device=device)

    model.eval()
    with torch.no_grad():
        outputs = model(X)

    print('Input :', tuple(X.shape))
    print('SIC   :', tuple(outputs['SIC'].shape))
    print('SOD   :', tuple(outputs['SOD'].shape))
    print('FLOE  :', tuple(outputs['FLOE'].shape))
