"""SSCI-Net model implementation.

Paper-facing method/module names are used where possible. Registered legacy
submodule names are retained when changing them would break existing checkpoint
state-dict keys. Unused alternative top-level experiment utilities were removed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from resnet import *
from help_funcs import (
    Transformer, TransformerDecoder, TwoLayerConv2d,
    TwoLayerFeatureExtractor, DoubleConv, Down, Final
)
from ssci_modules import SCTMM, DCSFM, SelfAttention

class MCM1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MCM1,self).__init__()
        self.Branch1x1=nn.Conv2d(in_channels,out_channels,kernel_size=1)


        self.Branch3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.Branch3x3=nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)


        self.Branch5x5_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.Branch5x5=nn.Conv2d(out_channels,out_channels,kernel_size=5,padding=2)

        self.Branchmax1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.bn=nn.BatchNorm2d(out_channels*4,eps=0.001)


    def forward(self, x):
        branch1x1=self.Branch1x1(x)

        branch2_1=self.Branch3x3_1(x)
        branch2_2=self.Branch3x3(branch2_1)

        branch3_1=self.Branch5x5_1(x)
        branch3_2=self.Branch5x5(branch3_1)

        branchpool4_1=F.max_pool2d(x,kernel_size=3,stride=1,padding=1)
        branchpool4_2=self.Branchmax1x1(branchpool4_1)

        outputs=[branch1x1,branch2_2,branch3_2,branchpool4_2]
        # x=(branch1x1+branch2_2+branch3_2+branchpool4_2)
        x= torch.cat(outputs,1)
        x=self.bn(x)
        return F.relu(x,inplace=True)
class ConvBlock(nn.Module):
    """Conv-BN-ReLU block used for the pretrained 24-to-3 input projection."""

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3,
                 stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, padding=padding,
            kernel_size=kernel_size, stride=stride
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x

class ResNet(torch.nn.Module):
    def __init__(self, input_nc, output_nc,Flag,
                 resnet_stages_num=5, backbone='resnet18',
                 output_sigmoid=False, if_upsample_2x=True,):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ResNet, self).__init__()
        expand = 1
        if backbone == 'resnet18':
            self.resnet = resnet18(pretrained=Flag,
                                   replace_stride_with_dilation=[False, True, True])
        elif backbone == 'resnet34':
            self.resnet = resnet34(pretrained=Flag,
                                   replace_stride_with_dilation=[False, True, True])
        elif backbone == 'resnet50':
            self.resnet = resnet50(pretrained=Flag,
                                   replace_stride_with_dilation=[False, True, True])
            expand = 4
            print("50")
        else:
            raise NotImplementedError
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2)
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')

        self.classifier = TwoLayerConv2d(in_channels=32, out_channels=output_nc)

        self.resnet_stages_num = resnet_stages_num

        self.if_upsample_2x = if_upsample_2x
        if self.resnet_stages_num == 5:
            layers = 512 * expand
        elif self.resnet_stages_num == 4:
            layers = 256 * expand
        elif self.resnet_stages_num == 3:
            layers = 128 * expand
        else:
            raise NotImplementedError
        self.conv_pred = nn.Conv2d(layers, input_nc*3, kernel_size=3, padding=1)

        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x1 = self.forward_single(x1)
        x2 = self.forward_single(x2)
        x = torch.abs(x1 - x2)
        if not self.if_upsample_2x:
            x = self.upsamplex2(x)
        x = self.upsamplex4(x)
        x = self.classifier(x)

        if self.output_sigmoid:
            x = self.sigmoid(x)
        return x

    def forward_single(self, x):
        # resnet layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x_4 = self.resnet.layer1(x)  # 1/4, in=64, out=64
        x_8 = self.resnet.layer2(x_4)  # 1/8, in=64, out=128

        if self.resnet_stages_num > 3:
            x_8 = self.resnet.layer3(x_8)  # 1/8, in=128, out=256

        if self.resnet_stages_num == 5:
            x_8 = self.resnet.layer4(x_8)  # 1/32, in=256, out=512
        elif self.resnet_stages_num > 5:
            raise NotImplementedError

        if self.if_upsample_2x:
            x = self.upsamplex2(x_8)
        else:
            x = x_8
        # output layers
        x = self.conv_pred(x)
        return x
class SSCINet(ResNet):
    """
    Semantic-Spatial Collaborative Interaction Network (SSCI-Net).
    """

    def __init__(self, input_nc, output_nc, with_pos,pre,resnet_stages_num=5,
                 token_len=4, token_trans=True,
                 enc_depth=1, dec_depth=1,
                 dim_head=64, decoder_dim_head=64,
                 tokenizer=True, if_upsample_2x=True,
                 pool_mode='max', pool_size=2,
                 backbone='resnet18',
                 decoder_softmax=True, with_decoder_pos=None,
                 with_decoder=True,):
        super(SSCINet, self).__init__(input_nc, output_nc,pre, backbone=backbone,
                                               resnet_stages_num=resnet_stages_num,
                                               if_upsample_2x=if_upsample_2x,
                                               )
        self.token_len = token_len
        self.conv_a = nn.Conv2d(input_nc, self.token_len, kernel_size=1,
                                padding=0, bias=False)
        self.tokenizer = tokenizer
        self.exct_sar = DoubleConv(2, input_nc, input_nc//2)
        self.exct_amsr = DoubleConv(14, input_nc, input_nc//2)
        self.exct_aul = DoubleConv(8, input_nc, input_nc//2)
        self.model_com=TwoLayerFeatureExtractor(input_nc,input_nc,input_nc*2)
        self.down1 = Down(input_nc, input_nc*2)
        self.down2 = Down(input_nc*2, input_nc)
        if not self.tokenizer:
            #  if not use tokenzier，then downsample the feature map into a certain size
            self.pooling_size = pool_size
            self.pool_mode = pool_mode
            self.token_len = self.pooling_size * self.pooling_size
        self.token_trans = token_trans
        self.with_decoder = with_decoder
        dim = input_nc*2
        mlp_dim = 2 * dim
        self.with_pos = with_pos
        if with_pos == 'learned':
            self.pos_embedding = nn.Parameter(torch.randn(1, self.token_len * 3, input_nc))
        decoder_pos_size = 256 // 4
        self.with_decoder_pos = with_decoder_pos
        if self.with_decoder_pos == 'learned':
            self.pos_embedding_decoder = nn.Parameter(torch.randn(1, 32,
                                                                  decoder_pos_size,
                                                                  decoder_pos_size))
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.dim_head = dim_head
        self.decoder_dim_head = decoder_dim_head
        self.transformer = Transformer(dim=dim // 2, depth=self.enc_depth, heads=8,
                                       dim_head=self.dim_head,
                                       mlp_dim=mlp_dim, dropout=0)
        self.transformer_decoder = TransformerDecoder(dim=input_nc*3, depth=self.dec_depth,
                                                      heads=8, dim_head=self.decoder_dim_head, mlp_dim=mlp_dim,
                                                      dropout=0,
                                                      softmax=decoder_softmax)
        self.classifier_SIC = TwoLayerConv2d(in_channels=input_nc*3, out_channels=12)#dim
        self.classifier_SOD = TwoLayerConv2d(in_channels=input_nc*3, out_channels=7)
        self.classifier_FLOE = TwoLayerConv2d(in_channels=input_nc*3, out_channels=8)

        self.mac = MCM1(input_nc//4*3, input_nc//4*3)
        self.vis_merge_proj = nn.Conv2d(input_nc*2, input_nc*4, 1)  # 合并视觉特征+通道投影（8+16→64）
        self.sem_proj = nn.Linear(input_nc, input_nc*4)  # 语义Token维度投影（16→64）
        # 多头交叉注意力（双向交互核心，head数=8，维度=64，轻量高效）
        self.cross_attn = nn.MultiheadAttention(embed_dim=input_nc*4, num_heads=8, batch_first=True, dropout=0.1)

        self._upsample_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=input_nc*3, out_channels=dim, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=input_nc*6, out_channels=input_nc*3, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(input_nc*3),
            nn.GELU()  # 非线性保留信息
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=input_nc*6, out_channels=input_nc*3, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(input_nc*3),
            nn.GELU()  # 非线性保留信息
        )


        self._upsample_block2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=dim, out_channels=dim, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.channel_compress = nn.Sequential(
            nn.Conv2d(input_nc*4, input_nc, 1),  # 核心压缩
            nn.BatchNorm2d(input_nc),  # 归一化稳定
            nn.GELU()  # 非线性保留信息
        )
        self.conv_de = nn.Conv2d(input_nc*3, input_nc*3//2, kernel_size=1,
                                 padding=0, bias=False)

        self.adaptive_fusion = nn.Sequential(
            # 1×1 卷积：压缩通道 + 统一特征空间（24→16，减少计算）
            nn.Conv2d(input_nc*3, input_nc*3//2, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(input_nc*3//2),
            nn.GELU(),  # 非线性激活，保留细节
            # 生成自适应权重（16→1，单通道权重图）
            nn.Conv2d(input_nc*3//2, 1, kernel_size=1, padding=0, bias=False),
            nn.Sigmoid()  # 权重归一化到 [0,1]
        )
        self.residual_proj = nn.Linear(input_nc*3,  input_nc*3//2, bias=False)
        self.final=Final(dim, dim)
        self.atte=SelfAttention(input_nc*3)

        self.token_fusion = SCTMM(l=self.token_len, n=input_nc)
        self.f_fusion = DCSFM(in_channels=input_nc, reduction=8)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.max_i=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.pre=pre
        self.co_pre=ConvBlock(24,3)
    def _semantic_token_generator(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)

        return tokens

    def _forward_reshape_tokens(self, x):
        # b,c,h,w = x.shape
        if self.pool_mode == 'max':
            x = F.adaptive_max_pool2d(x, [self.pooling_size, self.pooling_size])
        elif self.pool_mode == 'ave':
            x = F.adaptive_avg_pool2d(x, [self.pooling_size, self.pooling_size])
        else:
            x = x
        tokens = rearrange(x, 'b c h w -> b (h w) c')
        return tokens

    def _forward_transformer(self, x):
        if self.with_pos:
            x += self.pos_embedding
        x = self.transformer(x)
        return x

    def _transformer_fusion(self, x, m):
        b, c, h, w = x.shape
        if self.with_decoder_pos == 'fix':
            x = x + self.pos_embedding_decoder
        elif self.with_decoder_pos == 'learned':
            x = x + self.pos_embedding_decoder
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.transformer_decoder(x, m)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h)
        return x

    def _forward_simple_decoder(self, x, m):
        b, c, h, w = x.shape
        b, l, c = m.shape
        m = m.expand([h, w, b, l, c])
        m = rearrange(m, 'h w b l c -> l b c h w')
        m = m.sum(0)
        x = x + m
        return x

    def semantic_guided_fusion(self, x, x_resnet, global_tok):
        """语义引导的残差融合"""
        b, c, h, w = x.shape

        # 1. 从global_tok生成残差修正项
        # 将global_tok转换为与特征图通道数匹配
        residual_guide = self.residual_proj(global_tok)
        residual_guide = residual_guide.transpose(2, 1)  # (1, 4, 48) -> (1, 24, 4)
        residual_guide = residual_guide.mean(dim=2, keepdim=True)  # (1, 24, 1)
        residual_guide = residual_guide.unsqueeze(-1)  # (1, 24, 1, 1)

        # 2. 基础融合（原方案）
        base_weight = self.adaptive_fusion(x)
        x_base = base_weight * x + (1 - base_weight) * x_resnet

        # 3. 语义引导的残差修正
        # 计算x和x_resnet的差异，用global_tok指导修正
        diff = x - x_resnet  # 特征差异
        residual_mask = torch.sigmoid(residual_guide)  # 语义指导的修正强度

        # 4. 最终融合：基础融合 + 语义指导的残差修正
        x_fused = x_base + residual_mask * diff

        return x_fused

    def _scim(self, fea, ori_fea, token):
        # 输入：fea(1,16,64,64)、ori_fea(1,8,64,64)、token(1,4,16)
        B,C, H, W = fea.shape

        #######################################
        # 步骤1：极简预处理（维度统一+形态对齐）
        #######################################
        # 1.1 视觉特征：合并→投影→展平为序列（适配注意力）
        vis_feat = torch.cat([fea, ori_fea], dim=1)  # (1,24,64,64) → 合并两个视觉特征
        vis_feat = self.vis_merge_proj(vis_feat)  # (1,64,64,64) → 投影到64维（与语义统一）
        vis_seq = vis_feat.flatten(2).transpose(1, 2)  # (1, 64*64=4096, 64) → 2D→序列（B, Lv, D）

        # 1.2 语义Token：维度投影（保持序列形态）
        sem_seq = self.sem_proj(token)  # (1,4,16) → (1,4,64)（B, Ls=4, D）

        #######################################
        # 步骤2：双向注意力交互（核心高级逻辑）
        #######################################
        # 语义引导视觉：用语义Token作为Key/Value，视觉序列作为Query → 视觉聚焦语义相关区域
        vis_enhanced, _ = self.cross_attn(query=vis_seq, key=sem_seq, value=sem_seq)  # (1,4096,64)
        # 视觉增强语义：用增强后的视觉作为Key/Value，语义Token作为Query → 语义吸收视觉细节
        sem_enhanced, _ = self.cross_attn(query=sem_seq, key=vis_enhanced, value=vis_enhanced)  # (1,4,64)

        #######################################
        # 步骤3：自适应加权融合（简便且精准）
        #######################################
        # 语义全局池化→生成像素级注意力权重（语义引导视觉权重分配）
        sem_weight = sem_enhanced.mean(dim=1).unsqueeze(1).unsqueeze(1)  # (1,1,1,64) → 全局语义特征
        sem_weight = F.softmax(sem_weight, dim=-1)  # 归一化权重（0-1）

        # 视觉特征重构+语义加权（逐通道自适应融合）
        vis_enhanced = vis_enhanced.transpose(1, 2).reshape(B, C*4, H, W)  # (1,4096,64) → (1,64,64,64)
        fused_feat = vis_enhanced * sem_weight.transpose(3, 1)  # (1,64,64,64) → 语义加权视觉特征
        fused_feat_8ch = self.channel_compress(fused_feat)
        return fused_feat_8ch  # 输出融合后特征：(1,64,64,64)

    @property
    def sctmm(self):
        """Paper-facing alias; registered name is kept for checkpoint compatibility."""
        return self.token_fusion

    @property
    def dcsfm(self):
        """Paper-facing alias; registered name is kept for checkpoint compatibility."""
        return self.f_fusion

    def forward(self, x1):
        # ------------------------------------------------------------------
        # 1. Initial Feature Extraction
        # ------------------------------------------------------------------
        x_raw = x1
        backbone_input = x1
        if self.pre:
            backbone_input = self.co_pre(backbone_input)
        x_sp_cat = self.forward_single(backbone_input)

        # Input modality partition: SAR / AMSR2 / auxiliary.
        x_s = x_raw[:, 0:2, :, :]
        x_r = x_raw[:, 4:18, :, :]
        x_a = torch.cat([
            x_raw[:, 2:4, :, :],
            x_raw[:, 18:24, :, :]
        ], dim=1)

        # Modality-specific embedding + shared encoder.
        f_enc_s = self.model_com(self.exct_sar(x_s))
        f_enc_r = self.model_com(self.exct_amsr(x_r))
        f_enc_a = self.model_com(self.exct_aul(x_a))

        f_enc_s = self.down2(self.down1(f_enc_s))
        f_enc_r = self.down2(self.down1(f_enc_r))
        f_enc_a = self.down2(self.down1(f_enc_a))

        # Semantic token generation and Transformer encoding.
        t_init_s = self._semantic_token_generator(f_enc_s)
        t_init_r = self._semantic_token_generator(f_enc_r)
        t_init_a = self._semantic_token_generator(f_enc_a)
        tokens = torch.cat([t_init_s, t_init_r, t_init_a], dim=1)
        tokens = self._forward_transformer(tokens)
        t_s, t_r, t_a = tokens.chunk(3, dim=1)

        # Three modality-associated latent spatial groups.
        f_sp_s, f_sp_r, f_sp_a = x_sp_cat.chunk(3, dim=1)

        # ------------------------------------------------------------------
        # 2. TTFFM: SCIM followed by DCSFM
        # ------------------------------------------------------------------
        f_sp_hat_s = self._scim(f_enc_s, f_sp_s, t_s)
        f_sp_hat_r = self._scim(f_enc_r, f_sp_r, t_r)
        f_sp_hat_a = self._scim(f_enc_a, f_sp_a, t_a)
        f_g = self.dcsfm(f_sp_hat_s, f_sp_hat_r, f_sp_hat_a)

        # Parallel SCTMM semantic branch.
        t_cat = torch.cat([t_s, t_r, t_a], dim=2)
        t_g = self.sctmm(t_s, t_r, t_a) + t_cat

        # ------------------------------------------------------------------
        # 3. Transformer Fusion
        # ------------------------------------------------------------------
        a_spa = self.adaptive_fusion(f_g)
        f_spa = a_spa * f_g + (1 - a_spa) * x_sp_cat
        f_tf = self._transformer_fusion(f_spa, t_g)

        if not self.if_upsample_2x:
            f_tf = self.upsamplex2(f_tf)
        f_tf = self.up1(torch.cat((f_tf, f_spa), dim=1))
        f_out = self.up2(torch.cat((f_tf, self.max_i(f_spa)), dim=1))

        # ------------------------------------------------------------------
        # 4. Multi-Task Prediction
        # ------------------------------------------------------------------
        sic = self.classifier_SIC(f_out)
        sod = self.classifier_SOD(f_out)
        floe = self.classifier_FLOE(f_out)
        return {'SIC': sic, 'SOD': sod, 'FLOE': floe}
# #
# net = SSCINet(input_nc=16, output_nc=2, token_len=10, resnet_stages_num=4,
#                             with_pos='learned',pre=True)

