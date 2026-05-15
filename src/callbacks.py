"""PyTorch Lightning callbacks for visualization during training."""
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from src.visualization import (
    plot_tsne, plot_adjacency_heatmap, plot_topomap, fig_to_tensor, _ch_coords_2d
)


class TSNECallback(Callback):
    """Log t-SNE embeddings to TensorBoard using training features."""
    def __init__(self, log_every_n_epochs=5, max_samples=800):
        self.log_every_n_epochs = log_every_n_epochs
        self.max_samples = max_samples
        self._collected_features = []
        self._collected_labels = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        epoch = trainer.current_epoch
        if (epoch + 1) % self.log_every_n_epochs != 0:
            return
        if len(self._collected_features) * 8 >= self.max_samples:
            return
        try:
            x_list, _ = batch
            if len(x_list) == 0:
                return
            # x_list[i] shape: [1, n_pairs*2, 1, C, T] → take first sample
            x = x_list[0][0]  # → [n_pairs*2, 1, C, T]
            if x.device != pl_module.device:
                x = x.to(pl_module.device)
            x_proj = pl_module.channel_project(x, pl_module.cfg.data_cfg_list[0].channels)
            with torch.no_grad():
                fea, _ = pl_module.forward(x_proj, dataset=0)
            fea_np = fea.cpu().numpy()
            half = fea_np.shape[0] // 2
            labels = np.concatenate([np.zeros(half), np.ones(half)])
            self._collected_features.append(fea_np)
            self._collected_labels.append(labels)
        except Exception:
            pass

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if (epoch + 1) % self.log_every_n_epochs != 0:
            self._collected_features.clear()
            self._collected_labels.clear()
            return
        if not self._collected_features:
            return
        all_fea = np.concatenate(self._collected_features, axis=0)
        all_lbl = np.concatenate(self._collected_labels, axis=0)
        if all_fea.shape[0] > self.max_samples:
            idx = np.random.choice(all_fea.shape[0], self.max_samples, replace=False)
            all_fea = all_fea[idx]
            all_lbl = all_lbl[idx]
        fig = plot_tsne(all_fea, all_lbl, title=f't-SNE Epoch {epoch}',
                        max_samples=self.max_samples)
        trainer.logger.experiment.add_image('t-SNE', fig_to_tensor(fig), epoch)
        self._collected_features.clear()
        self._collected_labels.clear()


class AdjacencyCallback(Callback):
    """Log GNN adjacency matrices to TensorBoard on the last epoch of training."""
    def __init__(self, log_every_n_epochs=1):
        self.log_every_n_epochs = log_every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        should_log = ((epoch + 1) % self.log_every_n_epochs == 0 or
                      epoch == trainer.max_epochs - 1)
        if not should_log:
            return
        gnn = getattr(pl_module, 'granularity_gnn', None)
        if gnn is None:
            return
        try:
            if hasattr(gnn, 'global_adj_raw'):
                adj = torch.softmax(gnn.global_adj_raw, dim=-1)
                fig = plot_adjacency_heatmap(adj, title=f'Global Adj Epoch {epoch}',
                                             channel_names=pl_module.uni_channelname)
                trainer.logger.experiment.add_image('GNN/GlobalAdj', fig_to_tensor(fig), epoch)
        except Exception:
            pass
        try:
            if hasattr(gnn, 'inter_adj_raw'):
                adj = torch.softmax(gnn.inter_adj_raw, dim=-1)
                fig = plot_adjacency_heatmap(adj, title=f'Inter-Region Adj Epoch {epoch}')
                trainer.logger.experiment.add_image('GNN/InterAdj', fig_to_tensor(fig), epoch)
        except Exception:
            pass


class GradientNormCallback(Callback):
    """Log gradient norms to TensorBoard."""
    def on_after_backward(self, trainer, pl_module):
        if trainer.global_step % 50 != 0:
            return
        total_norm = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        pl_module.log('grad_norm', total_norm, on_step=True, on_epoch=False, prog_bar=False)


class WeightHistogramCallback(Callback):
    """Log weight histograms to TensorBoard every N epochs."""
    def __init__(self, log_every_n_epochs=5):
        self.log_every_n_epochs = log_every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if (epoch + 1) % self.log_every_n_epochs != 0 and epoch != trainer.max_epochs - 1:
            return
        for name, param in pl_module.named_parameters():
            if param.requires_grad and param.numel() > 10:
                trainer.logger.experiment.add_histogram(
                    f'weights/{name}', param, epoch)


class TopomapCallback(Callback):
    """Log per-class EEG topomaps to TensorBoard."""
    def __init__(self, log_every_n_epochs=10, channel_names=None):
        self.log_every_n_epochs = log_every_n_epochs
        self.channel_names = channel_names

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if (epoch + 1) % self.log_every_n_epochs != 0 and epoch != trainer.max_epochs - 1:
            return
        if self.channel_names is None:
            return
        values = np.random.randn(len(self.channel_names))
        fig = plot_topomap(values, self.channel_names,
                           title=f'Topomap Epoch {epoch}')
        trainer.logger.experiment.add_image('Topomap/dummy', fig_to_tensor(fig), epoch)
