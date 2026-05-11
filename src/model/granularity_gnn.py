import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GraphConv(nn.Module):
    """Graph convolution with optional dynamic attention: H' = σ(α ⊙ A @ H @ W)"""
    def __init__(self, in_dim, out_dim, use_attention=False):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.use_attention = use_attention
        if use_attention:
            self.att = nn.Linear(in_dim * 2, 1)

    def forward(self, x, adj):
        # x: [B, N, D_in], adj: [N, N] or [B, N, N]
        support = self.W(x)  # [B, N, D_out]
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(x.shape[0], -1, -1)

        if self.use_attention:
            B, N, _ = adj.shape
            # Compute pairwise attention: concat(x_i, x_j) -> scalar
            xi = x.unsqueeze(2).expand(-1, -1, N, -1)   # [B, N, N, D_in]
            xj = x.unsqueeze(1).expand(-1, N, -1, -1)   # [B, N, N, D_in]
            att_in = torch.cat([xi, xj], dim=-1)          # [B, N, N, 2*D_in]
            att_w = self.att(att_in).squeeze(-1)           # [B, N, N]
            att_w = torch.softmax(att_w, dim=-1)
            adj = adj * att_w                              # weighted adjacency

        out = torch.bmm(adj, support)  # [B, N, D_out]
        out = self.norm(out)
        return F.relu(out)


class MultiGranularityGNN(nn.Module):
    """Multi-granularity spatial modeling with learnable distance-based adjacency.

    Three levels:
    1. Global: all channels with learnable adjacency (+ GAT attention)
    2. Intra-region: channels within each functional brain region
    3. Inter-region: brain regions as super-nodes

    Adjacency initialized from 10-20 spatial distances, then fine-tuned.
    """
    # Approximate 10-20 2D coords (normalized [-1,1]) for 60 channels
    # y: FP→O (front→back), x: left→right
    _CHANNEL_COORDS = {
        # Prefrontal
        'FP1': (-0.4,  0.9), 'FPZ': ( 0.0,  0.9), 'FP2': ( 0.4,  0.9),
        # Anterior frontal
        'AF3': (-0.3,  0.7), 'AF4': ( 0.3,  0.7),
        # Frontal
        'F7':  (-0.8,  0.6), 'F5':  (-0.55, 0.6), 'F3':  (-0.35, 0.6),
        'F1':  (-0.15, 0.6), 'FZ':  ( 0.0,  0.6), 'F2':  ( 0.15, 0.6),
        'F4':  ( 0.35, 0.6), 'F6':  ( 0.55, 0.6), 'F8':  ( 0.8,  0.6),
        # Fronto-temporal / Fronto-central
        'FT7': (-0.85, 0.4), 'FC5': (-0.6,  0.4), 'FC3': (-0.35, 0.4),
        'FC1': (-0.15, 0.4), 'FCZ': ( 0.0,  0.4), 'FC2': ( 0.15, 0.4),
        'FC4': ( 0.35, 0.4), 'FC6': ( 0.6,  0.4), 'FT8': ( 0.85, 0.4),
        # Temporal / Central
        'T7':  (-0.95, 0.15), 'C5':  (-0.6,  0.15), 'C3':  (-0.35, 0.15),
        'C1':  (-0.15, 0.15), 'CZ':  ( 0.0,  0.15), 'C2':  ( 0.15, 0.15),
        'C4':  ( 0.35, 0.15), 'C6':  ( 0.6,  0.15), 'T8':  ( 0.95, 0.15),
        # Temporal-parietal / Centro-parietal
        'TP7': (-0.85,-0.1), 'CP5': (-0.6, -0.1), 'CP3': (-0.35,-0.1),
        'CP1': (-0.15,-0.1), 'CPZ': ( 0.0, -0.1), 'CP2': ( 0.15,-0.1),
        'CP4': ( 0.35,-0.1), 'CP6': ( 0.6, -0.1), 'TP8': ( 0.85,-0.1),
        # Parietal
        'P7':  (-0.8, -0.35), 'P5':  (-0.55,-0.35), 'P3':  (-0.35,-0.35),
        'P1':  (-0.15,-0.35), 'PZ':  ( 0.0, -0.35), 'P2':  ( 0.15,-0.35),
        'P4':  ( 0.35,-0.35), 'P6':  ( 0.55,-0.35), 'P8':  ( 0.8, -0.35),
        # Parieto-occipital
        'PO7': (-0.7, -0.55), 'PO5': (-0.45,-0.55), 'PO3': (-0.25,-0.55),
        'POZ': ( 0.0, -0.55), 'PO4': ( 0.25,-0.55), 'PO6': ( 0.45,-0.55),
        'PO8': ( 0.7, -0.55),
        # Occipital
        'O1':  (-0.3, -0.75), 'OZ':  ( 0.0, -0.75), 'O2':  ( 0.3, -0.75),
    }

    def __init__(self, n_channels, feat_dim, region_groups, hidden_dim=None,
                 n_layers=1, channel_names=None, dist_sigma=0.3):
        super().__init__()
        self.n_channels = n_channels
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim or feat_dim
        self.region_groups = region_groups
        self.n_regions = len(region_groups)
        self.dist_sigma = dist_sigma

        # Build spatial coordinates and distance-based adjacency prior
        coords = self._build_coords(channel_names)
        dist_adj = self._build_dist_adj(coords)

        # Learnable adjacency parameters (initialized from distance prior)
        self.global_adj_raw = nn.Parameter(self._adj_to_logits(dist_adj))
        self.register_buffer('region_mask', self._build_region_mask())
        self.register_buffer('inter_region_prior', self._build_inter_region_prior())

        # Learnable inter-region adjacency
        self.inter_adj_raw = nn.Parameter(
            self._adj_to_logits(self.inter_region_prior))

        # GNN layers
        self.global_conv = GraphConv(feat_dim, self.hidden_dim, use_attention=True)
        self.intra_conv = GraphConv(feat_dim, self.hidden_dim // 2, use_attention=False)
        self.inter_conv = GraphConv(self.hidden_dim // 2, self.hidden_dim // 4, use_attention=True)

        # Fusion projection
        fused_dim = self.hidden_dim + (self.hidden_dim // 2) + (self.hidden_dim // 4)
        self.fuse_proj = nn.Linear(fused_dim, feat_dim)
        self.fuse_norm = nn.LayerNorm(feat_dim)
        # Learnable residual scale
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def _build_coords(self, channel_names):
        """Build [n_channels, 2] coordinate tensor."""
        coords = torch.zeros(self.n_channels, 2)
        if channel_names is None:
            return coords
        for i, name in enumerate(channel_names):
            name_upper = name.upper()
            if name_upper in self._CHANNEL_COORDS:
                coords[i] = torch.tensor(self._CHANNEL_COORDS[name_upper])
        return coords

    def _build_dist_adj(self, coords):
        """Gaussian kernel adjacency: exp(-d^2 / 2σ^2), no self-loop."""
        d2 = torch.cdist(coords, coords) ** 2
        adj = torch.exp(-d2 / (2 * self.dist_sigma ** 2))
        adj.fill_diagonal_(0)
        return adj / adj.sum(dim=1, keepdim=True).clamp(min=1e-8)

    def _adj_to_logits(self, adj):
        """Convert normalized adjacency to logits for softmax parametrization."""
        adj = adj.clamp(min=1e-8)
        return torch.log(adj)

    def _get_learnable_adj(self, raw_logits):
        """Softmax along rows to get normalized adjacency from learnable logits."""
        return F.softmax(raw_logits, dim=-1)

    def _build_region_mask(self):
        """Build [n_channels, n_regions] binary mask."""
        mask = torch.zeros(self.n_channels, self.n_regions)
        for r_idx, group in enumerate(self.region_groups):
            for ch_idx in group:
                if ch_idx < self.n_channels:
                    mask[ch_idx, r_idx] = 1.0
        return mask

    def _build_inter_region_prior(self):
        """Region adjacency from distance prior: all-connected with learnable weights."""
        adj = torch.ones(self.n_regions, self.n_regions)
        adj.fill_diagonal_(0)
        return adj / adj.sum(dim=1, keepdim=True).clamp(min=1)

    def _build_intra_adj(self, size, device):
        """Learnable intra-region adjacency for a region of given size."""
        # Default to uniform, but could be made learnable per-region
        adj = torch.ones(size, size, device=device)
        adj.fill_diagonal_(0)
        return adj / adj.sum(dim=1, keepdim=True).clamp(min=1)

    def forward(self, x):
        """x: [B, C, D] — per-channel features (time-averaged)."""
        B, C, D = x.shape
        assert C == self.n_channels, f"Expected {self.n_channels} channels, got {C}"

        # Build learnable adjacency matrices for this batch
        global_adj = self._get_learnable_adj(self.global_adj_raw)  # [C, C]

        # 1. Global: all channels, learnable distance-initialized adjacency + GAT
        g = self.global_conv(x, global_adj)  # [B, C, hidden_dim]

        # 2. Intra-region: per-region graphs
        region_features = []
        intra_list = []
        for r_idx, group in enumerate(self.region_groups):
            valid = [i for i in group if i < C]
            if len(valid) == 0:
                region_features.append(torch.zeros(B, self.hidden_dim // 2, device=x.device))
                continue
            indices = torch.tensor(valid, device=x.device)
            x_r = x[:, indices]  # [B, |R|, D]
            if len(valid) == 1:
                f = F.relu(self.intra_conv.W(x_r))
                r = f.mean(dim=1)
            else:
                r_adj = self._build_intra_adj(len(valid), x.device)
                f = self.intra_conv(x_r, r_adj)
                r = f.mean(dim=1)  # [B, hidden//2]
            region_features.append(r)
            # Broadcast region feature back to channels
            intra_list.append(r.unsqueeze(1).expand(-1, len(valid), -1))

        # Merge intra-region features back to channel space
        if intra_list:
            intra_catted = torch.cat(intra_list, dim=1)  # [B, total_valid, hidden//2]
            # Need to map back to original 60 channels
            intra_channel = torch.zeros(B, C, self.hidden_dim // 2, device=x.device)
            offset = 0
            for r_idx, group in enumerate(self.region_groups):
                valid = [i for i in group if i < C]
                if len(valid) > 0:
                    intra_channel[:, valid, :] = intra_catted[:, offset:offset+len(valid)]
                    offset += len(valid)
        else:
            intra_channel = torch.zeros(B, C, self.hidden_dim // 2, device=x.device)

        # 3. Inter-region: regions as super-nodes, learnable adjacency + GAT
        region_stack = torch.stack(region_features, dim=1)  # [B, n_regions, hidden//2]
        inter_adj = self._get_learnable_adj(self.inter_adj_raw)
        inter_region = self.inter_conv(region_stack, inter_adj)  # [B, n_regions, hidden//4]
        inter_channel = torch.matmul(
            self.region_mask.unsqueeze(0).expand(B, -1, -1), inter_region)

        # 4. Fusion with learnable residual
        fused = torch.cat([g, intra_channel, inter_channel], dim=-1)
        out = self.fuse_proj(fused)
        out = self.fuse_norm(out)
        out = F.relu(out)
        out = x + self.res_scale * out

        return out
