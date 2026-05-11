"""PyTorch Lightning callbacks for visualization during training."""
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from src.visualization import (
    plot_tsne, plot_adjacency_heatmap, plot_topomap, fig_to_tensor, _ch_coords_2d
)


class TSNECallback(Callback):
    """Log t-SNE embeddings to TensorBoard every N epochs."""
    def __init__(self, log_every_n_epochs=5, max_samples=800):
        self.log_every_n_epochs = log_every_n_epochs
        self.max_samples = max_samples
        self.features = []
        self.labels = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return
        # Collect features from validation — use the cnn_encoder output
        x_list, y_list = batch
        if len(x_list) == 0:
            return
        x = x_list[0]
        if isinstance(x, (list, tuple)):
            x = x[0]
        if x.device != pl_module.device:
            x = x.to(pl_module.device)
        x_proj = pl_module.channel_project(x, pl_module.cfg.data_cfg_list[0].channels)
        with torch.no_grad():
            fea, _ = pl_module.forward(x_proj, dataset=0)
        fea_np = fea.cpu().numpy()
        labels_np = np.tile(np.arange(fea_np.shape[0] // 2), 2)  # approximate labels from CLISA pairs
        if len(self.features) < self.max_samples:
            self.features.append(fea_np)
            self.labels.append(labels_np)

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            self.features.clear()
            self.labels.clear()
            return
        if not self.features:
            return
        all_fea = np.concatenate(self.features, axis=0)
        all_lbl = np.concatenate(self.labels, axis=0)
        fig = plot_tsne(all_fea, all_lbl, title=f't-SNE Epoch {trainer.current_epoch}',
                        max_samples=self.max_samples)
        trainer.logger.experiment.add_image('t-SNE', fig_to_tensor(fig), trainer.current_epoch)
        self.features.clear()
        self.labels.clear()


class AdjacencyCallback(Callback):
    """Log GNN adjacency matrices to TensorBoard every N epochs."""
    def __init__(self, log_every_n_epochs=5):
        self.log_every_n_epochs = log_every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return
        gnn = getattr(pl_module, 'granularity_gnn', None)
        if gnn is None:
            return
        # Global adjacency
        if hasattr(gnn, 'global_adj_raw'):
            adj = torch.softmax(gnn.global_adj_raw, dim=-1)
            fig = plot_adjacency_heatmap(adj, title=f'Global Adj Epoch {trainer.current_epoch}',
                                         channel_names=pl_module.uni_channelname)
            trainer.logger.experiment.add_image('GNN/GlobalAdj', fig_to_tensor(fig), trainer.current_epoch)
        # Inter-region adjacency
        if hasattr(gnn, 'inter_adj_raw'):
            adj = torch.softmax(gnn.inter_adj_raw, dim=-1)
            fig = plot_adjacency_heatmap(adj, title=f'Inter-Region Adj Epoch {trainer.current_epoch}')
            trainer.logger.experiment.add_image('GNN/InterAdj', fig_to_tensor(fig), trainer.current_epoch)


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
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return
        for name, param in pl_module.named_parameters():
            if param.requires_grad and param.numel() > 10:
                trainer.logger.experiment.add_histogram(
                    f'weights/{name}', param, trainer.current_epoch)


class TopomapCallback(Callback):
    """Log per-class EEG topomaps to TensorBoard every N epochs."""
    def __init__(self, log_every_n_epochs=10, channel_names=None):
        self.log_every_n_epochs = log_every_n_epochs
        self.channel_names = channel_names

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return
        if self.channel_names is None:
            return
        # Use a fixed dummy signal to demonstrate channel layout (real data needs extraction)
        # For now, show the channel layout itself
        n_filters = pl_module.cfg.model.MLLA.cnn.n_timeFilters
        values = np.random.randn(len(self.channel_names))  # placeholder
        fig = plot_topomap(values, self.channel_names,
                           title=f'Topomap Epoch {trainer.current_epoch}')
        trainer.logger.experiment.add_image('Topomap/dummy', fig_to_tensor(fig), trainer.current_epoch)
