import torch
import pytorch_lightning as pl
import torch.nn as nn
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import itertools
import time
import random
from src.model.CNN_Attention import cnn_PatchTST, cnn_MLLA, Conv_att_simple_mlp
from src.model.Channel_MLP import Channel_mlp_CNN
from src.model.PatchTSTsingle import PatchTST_single_backbone
from src.model.MLLA_new import channel_MLLA
from src.model.Ablation_Transformer import TemporalTransformer
from src.model.vector_quantizer import VectorQuantizer
from src.model.granularity_gnn import MultiGranularityGNN
from src.loss.loss import SimCLRLoss
from src.loss.CDA_loss import CDALoss
from src.utils import report_vram

class MultiModel_PL(pl.LightningModule):
    def __init__(self, cfg=None) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_fea = False
        self.channel_projection_matrix = [[None] * len(self.cfg.data_cfg_list)][0]
        self.channel_interpolate = np.load('channel_interpolate.npy').astype(int)
        self.uni_channelname = self.cfg.model.MLLA.uni_channels
        
        # load channel project matrix
        if(cfg.model.encoder == 'cnn'):
            self.cnn_encoder = Conv_att_simple_mlp(cfg.model.cnn.n_timeFilters,
                                               cfg.model.cnn.timeFilterLen,
                                               cfg.model.cnn.n_msFilters,
                                               cfg.model.cnn.msFilter_timeLen,
                                               cfg.model.cnn.n_channs,
                                               cfg.model.cnn.dilation_array,
                                               cfg.model.cnn.seg_att, 
                                               cfg.model.cnn.avgPoolLen,
                                               cfg.model.cnn.timeSmootherLen,
                                               cfg.model.cnn.multiFact,
                                               cfg.model.cnn.stratified, 
                                               cfg.model.cnn.activ,
                                               cfg.model.cnn.temp,
                                               cfg.model.cnn.saveFea,
                                               cfg.model.cnn.has_att,
                                               cfg.model.cnn.extract_mode,
                                               cfg.model.cnn.global_att,
                                               c_mlps = [Channel_mlp_CNN(cfg_i.n_channs, cfg.model.cnn.n_channs) for cfg_i in cfg.data_cfg_list])

        if(cfg.model.encoder == 'TST_single'):
            self.c_mlps = [Channel_mlp_CNN(cfg_i.n_channs, cfg.model.TST_single.cnn.n_channs) for cfg_i in cfg.data_cfg_list]
            self.patchTST = PatchTST_single_backbone(c_in=1,
                                              context_window=cfg.data_0.timeLen * cfg.data_0.fs,
                                              patch_len=cfg.model.TST_single.patch_len,
                                              stride=cfg.model.TST_single.patch_stride,
                                              d_model=cfg.model.TST_single.cnn.n_timeFilters,
                                              n_heads=cfg.model.TST_single.n_heads)
            self.cnn_encoder = cnn_PatchTST(cfg.model.TST_single.cnn.n_timeFilters,
                                               cfg.model.TST_single.cnn.timeFilterLen,
                                               cfg.model.TST_single.cnn.n_msFilters,
                                               cfg.model.TST_single.cnn.msFilter_timeLen,
                                               cfg.model.TST_single.cnn.n_channs,
                                               cfg.model.TST_single.cnn.dilation_array,
                                               cfg.model.TST_single.cnn.seg_att, 
                                               cfg.model.TST_single.cnn.avgPoolLen,
                                               cfg.model.TST_single.cnn.timeSmootherLen,
                                               cfg.model.TST_single.cnn.multiFact,
                                               cfg.model.TST_single.cnn.stratified, 
                                               cfg.model.TST_single.cnn.activ,
                                               cfg.model.TST_single.cnn.temp,
                                               cfg.model.TST_single.cnn.saveFea,
                                               cfg.model.TST_single.cnn.has_att,
                                               cfg.model.TST_single.cnn.extract_mode,
                                               cfg.model.TST_single.cnn.global_att)
        if(cfg.model.encoder == 'MLLA'):
            # self.c_mlps = nn.ModuleList([Channel_mlp_CNN(cfg_i.n_channs, cfg.model.MLLA.cnn.n_channs) for cfg_i in cfg.data_cfg_list])
            self.uni_mlp = Channel_mlp_CNN(len(self.uni_channelname), cfg.model.MLLA.cnn.n_channs)
            self.MLLA = channel_MLLA(
                context_window=cfg.data_0.timeLen * cfg.data_0.fs,
                patch_size=cfg.model.MLLA.patch_size,
                hidden_dim=cfg.model.MLLA.hidden_dim,
                out_dim=cfg.model.MLLA.out_dim,
                depth=cfg.model.MLLA.depth,
                patch_stride=cfg.model.MLLA.patch_stride,
                n_heads=cfg.model.MLLA.n_heads)
            # Multi-granularity spatial GNN
            self.granularity_gnn = self._build_granularity_gnn(cfg)
            self.cnn_encoder = cnn_MLLA(cfg.model.MLLA.cnn.n_timeFilters,
                                               cfg.model.MLLA.cnn.timeFilterLen,
                                               cfg.model.MLLA.cnn.n_msFilters,
                                               cfg.model.MLLA.cnn.msFilter_timeLen,
                                               cfg.model.MLLA.cnn.n_channs,
                                               cfg.model.MLLA.cnn.dilation_array,
                                               cfg.model.MLLA.cnn.seg_att, 
                                               cfg.model.MLLA.cnn.avgPoolLen,
                                               cfg.model.MLLA.cnn.timeSmootherLen,
                                               cfg.model.MLLA.cnn.multiFact,
                                               cfg.model.MLLA.cnn.stratified, 
                                               cfg.model.MLLA.cnn.activ,
                                               cfg.model.MLLA.cnn.temp,
                                               cfg.model.MLLA.cnn.saveFea,
                                               cfg.model.MLLA.cnn.has_att,
                                               cfg.model.MLLA.cnn.extract_mode,
                                               cfg.model.MLLA.cnn.global_att)
        if(cfg.model.encoder == 'Transformer'):
            self.transformer_encoder = TemporalTransformer(n_chann=len(self.uni_channelname),
                                                          dim=cfg.model.Transformer.dim,
                                                          dim_out=cfg.model.Transformer.out_dim,
                                                          n_heads=cfg.model.Transformer.n_heads)
            self.uni_mlp = Channel_mlp_CNN(len(self.uni_channelname), cfg.model.MLLA.cnn.n_channs)
            # Multi-granularity spatial GNN
            self.granularity_gnn = self._build_granularity_gnn(cfg)
            self.cnn_encoder = cnn_MLLA(cfg.model.MLLA.cnn.n_timeFilters,
                                               cfg.model.MLLA.cnn.timeFilterLen,
                                               cfg.model.MLLA.cnn.n_msFilters,
                                               cfg.model.MLLA.cnn.msFilter_timeLen,
                                               cfg.model.MLLA.cnn.n_channs,
                                               cfg.model.MLLA.cnn.dilation_array,
                                               cfg.model.MLLA.cnn.seg_att, 
                                               cfg.model.MLLA.cnn.avgPoolLen,
                                               cfg.model.MLLA.cnn.timeSmootherLen,
                                               cfg.model.MLLA.cnn.multiFact,
                                               cfg.model.MLLA.cnn.stratified, 
                                               cfg.model.MLLA.cnn.activ,
                                               cfg.model.MLLA.cnn.temp,
                                               cfg.model.MLLA.cnn.saveFea,
                                               cfg.model.MLLA.cnn.has_att,
                                               cfg.model.MLLA.cnn.extract_mode,
                                               cfg.model.MLLA.cnn.global_att)
        self.clisa_loss = SimCLRLoss(cfg.train.loss.temp)
        self.cda_loss = CDALoss(cfg)
        self.channel_projection_matrix = [[None] * len(self.cfg.data_cfg_list)][0]
        # VQ placeholders (created lazily)
        self.vq = None
        self.vq_proj = None
        self.vq_postproj = None
        self._last_vq_loss = 0.0
        # propagate cnn config options (enable region conv / scales) to cnn_encoder if available
        if hasattr(self, 'cnn_encoder') and self.cnn_encoder is not None:
            try:
                cnn_cfg = getattr(self.cfg.model, 'cnn', {})
                if isinstance(cnn_cfg, dict):
                    use_region = cnn_cfg.get('use_region_conv', False)
                    n_scales = cnn_cfg.get('n_scales', None)
                else:
                    use_region = getattr(self.cfg.model.cnn, 'use_region_conv', False)
                    n_scales = getattr(self.cfg.model.cnn, 'n_scales', None)
                if use_region:
                    setattr(self.cnn_encoder, 'use_region_conv', True)
                if n_scales is not None:
                    setattr(self.cnn_encoder, 'n_scales', int(n_scales))
            except Exception:
                pass
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.train.lr, weight_decay=self.cfg.train.wd)
        return {'optimizer': optimizer}
    
    def _build_granularity_gnn(self, cfg):
        """Build multi-granularity GNN with 7 functional brain regions."""
        use_gnn = cfg.model.get('use_granularity_gnn', False)
        if not use_gnn:
            return None
        n_channels = len(self.uni_channelname)
        # 7 functional brain regions based on 10-20 layout
        region_groups = [
            [0, 1, 2],                                                          # Prefrontal (FP)
            [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],                             # Frontal (AF+F)
            [15, 16, 17, 18, 19, 20, 21],                                       # Fronto-central (FC)
            [24, 25, 26, 27, 28, 29, 30],                                       # Central (C)
            [14, 22, 23, 31, 32, 40],                                           # Temporal (FT+T+TP)
            [33, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49],  # Parietal (CP+P)
            [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],                           # Occipital (PO+O)
        ]
        gnn_cfg = cfg.model.get('granularity_gnn', {})
        return MultiGranularityGNN(
            n_channels=n_channels,
            feat_dim=cfg.model.MLLA.out_dim,
            region_groups=region_groups,
            hidden_dim=gnn_cfg.get('hidden_dim', 64),
            n_layers=gnn_cfg.get('n_layers', 1),
            channel_names=self.uni_channelname,
            dist_sigma=gnn_cfg.get('dist_sigma', 0.3),
        )
    
    def forward(self, x, dataset=0, returnMLLAout=False):
        if(self.cfg.model.encoder == 'cnn'):
            if self.save_fea:
                self.cnn_encoder.saveFea = True
            x = self.cnn_encoder(x, dataset)
            return x
        if(self.cfg.model.encoder == 'TST_single'):
            x = x.squeeze(1)
            x = self.patchTST(x)
            x = torch.permute(x, (0, 2, 1, 3))
            x = self.c_mlps[dataset](x)
            if self.save_fea:
                self.cnn_encoder.saveFea = True
            x = self.cnn_encoder(x)
            return x
        if(self.cfg.model.encoder == 'MLLA'):
            x = self.MLLA(x)
            if(returnMLLAout):
                mllaout = x
            x = torch.permute(x, (0, 3, 1, 2))
            # Multi-granularity spatial GNN
            if self.granularity_gnn is not None:
                x_mean = x.mean(dim=-1).permute(0, 2, 1)  # [B, C, D]
                x_gnn = self.granularity_gnn(x_mean)       # [B, C, D]
                x_gnn = x_gnn.permute(0, 2, 1).unsqueeze(-1)  # [B, D, C, 1]
                x = x + x_gnn
            # x = self.c_mlps[dataset](x)
            x = self.uni_mlp(x)
            fea_cov = x
            # optional vector quantization on feature cov
            if getattr(self.cfg.model, 'use_vq', False):
                if self.vq is None:
                    # fea_cov shape is [B, dim, n_chann, T] -> we quantize across channel dim (dim)
                    C = fea_cov.shape[1] if fea_cov.dim()==4 else fea_cov.shape[1]
                    vq_cfg = getattr(self.cfg.model, 'vq', {}) if hasattr(self.cfg.model, 'vq') else {}
                    emb_dim = int(vq_cfg.get('emb_dim', C)) if isinstance(vq_cfg, dict) else getattr(self.cfg.model, 'vq_emb_dim', C)
                    self.vq = VectorQuantizer(num_embeddings=int(vq_cfg.get('n_emb', 512)), embedding_dim=emb_dim, commitment_cost=float(vq_cfg.get('commit', 0.25)))
                    # create projection layers if needed
                    if emb_dim != C:
                        if fea_cov.dim() == 4:
                            self.vq_proj = nn.Conv2d(C, emb_dim, kernel_size=1)
                            self.vq_postproj = nn.Conv2d(emb_dim, C, kernel_size=1)
                        else:
                            self.vq_proj = nn.Conv1d(C, emb_dim, kernel_size=1)
                            self.vq_postproj = nn.Conv1d(emb_dim, C, kernel_size=1)
                    # move newly created modules to current feature device so parameters are on correct device
                    try:
                        dev = fea_cov.device
                        self.vq = self.vq.to(dev)
                        if self.vq_proj is not None:
                            self.vq_proj = self.vq_proj.to(dev)
                        if self.vq_postproj is not None:
                            self.vq_postproj = self.vq_postproj.to(dev)
                    except Exception:
                        pass
                # project if needed
                vq_in = self.vq_proj(fea_cov) if self.vq_proj is not None else fea_cov
                quantized, vq_loss = self.vq(vq_in)
                if self.vq_postproj is not None:
                    quantized = self.vq_postproj(quantized)
                fea_cov = quantized
                self._last_vq_loss = vq_loss
            else:
                self._last_vq_loss = 0.0
            if self.save_fea:
                self.cnn_encoder.saveFea = True
            x = self.cnn_encoder(x)
            if(returnMLLAout):
                return x, fea_cov, mllaout
            return x, fea_cov
        if(self.cfg.model.encoder == 'Transformer'):
            x = self.transformer_encoder(x)
            # Multi-granularity spatial GNN
            if self.granularity_gnn is not None:
                x_mean = x.mean(dim=-1).permute(0, 2, 1)  # [B, C, D]
                x_gnn = self.granularity_gnn(x_mean)
                x_gnn = x_gnn.permute(0, 2, 1).unsqueeze(-1)
                x = x + x_gnn
            x = self.uni_mlp(x)
            fea_cov = x
            # optional VQ
            if getattr(self.cfg.model, 'use_vq', False):
                if self.vq is None:
                    C = fea_cov.shape[1] if fea_cov.dim()==4 else fea_cov.shape[1]
                    vq_cfg = getattr(self.cfg.model, 'vq', {}) if hasattr(self.cfg.model, 'vq') else {}
                    emb_dim = int(vq_cfg.get('emb_dim', C)) if isinstance(vq_cfg, dict) else getattr(self.cfg.model, 'vq_emb_dim', C)
                    self.vq = VectorQuantizer(num_embeddings=int(vq_cfg.get('n_emb', 512)), embedding_dim=emb_dim, commitment_cost=float(vq_cfg.get('commit', 0.25)))
                    if emb_dim != C:
                        if fea_cov.dim() == 4:
                            self.vq_proj = nn.Conv2d(C, emb_dim, kernel_size=1)
                            self.vq_postproj = nn.Conv2d(emb_dim, C, kernel_size=1)
                        else:
                            self.vq_proj = nn.Conv1d(C, emb_dim, kernel_size=1)
                            self.vq_postproj = nn.Conv1d(emb_dim, C, kernel_size=1)
                    # move newly created modules to current feature device
                    try:
                        dev = fea_cov.device
                        self.vq = self.vq.to(dev)
                        if self.vq_proj is not None:
                            self.vq_proj = self.vq_proj.to(dev)
                        if self.vq_postproj is not None:
                            self.vq_postproj = self.vq_postproj.to(dev)
                    except Exception:
                        pass
                vq_in = self.vq_proj(fea_cov) if self.vq_proj is not None else fea_cov
                quantized, vq_loss = self.vq(vq_in)
                if self.vq_postproj is not None:
                    quantized = self.vq_postproj(quantized)
                fea_cov = quantized
                self._last_vq_loss = vq_loss
            else:
                self._last_vq_loss = 0.0
            if self.save_fea:
                self.cnn_encoder.saveFea = True
            x = self.cnn_encoder(x)
            return x, fea_cov

    
    def _augment_eeg(self, x):
        """Contrastive data augmentation: channel dropout + Gaussian noise.
        x: [N, 1, C, T] — batch of EEG samples.
        Only applied during training (self.training == True).
        """
        aug_cfg = self.cfg.train.get('augmentation', {})
        if not aug_cfg.get('enable', False):
            return x
        if not self.training:
            return x

        channel_drop = aug_cfg.get('channel_drop', 0.1)
        noise_std = aug_cfg.get('noise_std', 0.02)
        scale_range = aug_cfg.get('scale_range', None)

        # Channel dropout: randomly zero entire channels
        if channel_drop > 0:
            mask = torch.rand(x.shape[0], 1, x.shape[2], 1, device=x.device) > channel_drop
            x = x * mask.float()

        # Gaussian noise
        if noise_std > 0:
            noise = torch.randn_like(x) * noise_std
            x = x + noise

        # Amplitude scaling
        if scale_range is not None and len(scale_range) == 2:
            scale = torch.FloatTensor(x.shape[0], 1, x.shape[2], 1).uniform_(*scale_range).to(x.device)
            x = x * scale

        return x

    def training_step(self, batch, batch_idx):
        loss = 0
        x_list, y_list = batch
        n_dataset = len(x_list)
        # x_list = [x_i[0] for x_i in x_list]  # 提取数据
        x_list = [self.channel_project(x_list[i][0], self.cfg.data_cfg_list[i].channels) for i in range(n_dataset)]  # 提取数据
        x_list = [self._augment_eeg(x_i) for x_i in x_list]
        fea_clisa = []
        fea_cov = []

        for dataset in range(n_dataset):
            fea_clisa_i, fea_cov_i = self.forward(x_list[dataset], dataset)
            fea_clisa.append(fea_clisa_i)
            fea_cov.append(fea_cov_i)

        # 计算损失
        if self.cfg.train.loss.clisa_loss:
            for dataset, fea_clisa_i in enumerate(fea_clisa):
                clisa_loss_i, logits_i, labels_i, (acc_1, acc_5) = self.clisa_loss(fea_clisa_i)
                loss += clisa_loss_i

            # 记录日志
                self.log_dict({
                    f'loss_clisa_{self.cfg.data_cfg_list[dataset].dataset_name}/train': clisa_loss_i,
                    # f'acc1_{self.cfg.data_cfg_list[dataset].dataset_name}/train': acc_1,
                    # f'acc5_{self.cfg.data_cfg_list[dataset].dataset_name}/train': acc_5,
                }, on_step=False, on_epoch=True, prog_bar=True)
        if self.cfg.train.loss.CDA_loss:
            cda_loss = self.cda_loss(fea_cov) * self.cfg.train.loss.CDA_factor
            loss += cda_loss
            self.log_dict({
                f'loss_cda/train': cda_loss,
            }, on_step=False, on_epoch=True, prog_bar=True)

        # add VQ loss if present and configured
        vq_w = getattr(self.cfg.train, 'vq_weight', 0.0)
        if vq_w and getattr(self, '_last_vq_loss', 0.0) is not None:
            vq_loss_val = self._last_vq_loss
            try:
                loss = loss + vq_loss_val * float(vq_w)
            except Exception:
                # fallback if vq_loss_val is float
                loss = loss + float(vq_loss_val) * float(vq_w)
            self.log_dict({
                'loss_vq/train': vq_loss_val,
                'loss_vq/weight': float(vq_w)
            }, on_step=False, on_epoch=True, prog_bar=False)

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = 0
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        # 用来临时指定predict时用谁的mlp。-1即为未训练的随机mlp。（原本是作为微调基底）
        x = self.channel_project(x, self.cfg.data_val.channels)
        fea_clisa_i, fea_cov_i, mllaout = self.forward(x, 0, returnMLLAout=True)
        return fea_clisa_i, fea_cov_i, mllaout
    
    def channel_project(self, data, cha_source):
        # np.save('./visualize/original_eeg', data.cpu())
        device = data.device
        batch_size, _, n_channel_source, n_timepoint = data.shape
        n_channel_standard = len(self.uni_channelname)
        
        # 创建输入通道名称映射表（统一大写处理）
        source_ch_map = {name.upper(): idx for idx, name in enumerate(cha_source)}
        
        # 初始化结果张量（使用零值作为默认填充）
        result = torch.zeros((batch_size, 1, n_channel_standard, n_timepoint),
                            device=device,
                            dtype=data.dtype)
        
        # 遍历所有标准通道
        for std_idx, std_name in enumerate(self.uni_channelname):
            std_name_upper = std_name.upper()
            
            # Case 1: 直接存在对应通道
            if std_name_upper in source_ch_map:
                src_idx = source_ch_map[std_name_upper]
                result[:, :, std_idx] = data[:, :, src_idx]
                continue
                
            # Case 2: 需要插值的情况
            # 获取预存的最近邻索引（标准通道坐标系）
            neighbor_std_indices = self.channel_interpolate[std_idx]
            
            # 寻找实际存在的最近邻通道（输入数据坐标系）
            valid_src_indices = []
            for neighbor_std_idx in neighbor_std_indices:
                neighbor_std_name = self.uni_channelname[neighbor_std_idx.item()].upper()
                if neighbor_std_name in source_ch_map:
                    valid_src_indices.append(source_ch_map[neighbor_std_name])
                    if len(valid_src_indices) == 3:  # 最多取3个
                        break
            
            # 插值处理（根据找到的有效通道数量）
            if len(valid_src_indices) > 0:
                # 提取有效通道数据 [batch, 1, M, time]
                neighbor_data = data[:, :, valid_src_indices, :]
                
                # 简单平均插值（可替换为加权平均）
                interpolated = neighbor_data.mean(dim=2)  # [batch, 1, time]
                result[:, :, std_idx] = interpolated
            else:
                # 处理无可用通道情况（可选方案）
                # 方案1：保留零值 方案2：警告 方案3：抛出异常
                print(f"Channel {std_name} has no available neighbors, filled with zeros")
        # print(result.shape)
        # np.save('./visualize/projected_eeg_emo', result.cpu())
        return result