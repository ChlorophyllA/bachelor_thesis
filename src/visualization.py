"""Visualization utilities for EEG training: topomap, t-SNE, confusion, adj, heatmap."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize
import torch
import io

# ── A. Scalp Topology ──────────────────────────────────────────────────────────

_ch_coords_2d = {
    'FP1': (-0.4, 0.9), 'FPZ': (0.0, 0.9), 'FP2': (0.4, 0.9),
    'AF3': (-0.3, 0.7), 'AF4': (0.3, 0.7),
    'F7': (-0.8, 0.6), 'F5': (-0.55, 0.6), 'F3': (-0.35, 0.6),
    'F1': (-0.15, 0.6), 'FZ': (0.0, 0.6), 'F2': (0.15, 0.6),
    'F4': (0.35, 0.6), 'F6': (0.55, 0.6), 'F8': (0.8, 0.6),
    'FT7': (-0.85, 0.4), 'FC5': (-0.6, 0.4), 'FC3': (-0.35, 0.4),
    'FC1': (-0.15, 0.4), 'FCZ': (0.0, 0.4), 'FC2': (0.15, 0.4),
    'FC4': (0.35, 0.4), 'FC6': (0.6, 0.4), 'FT8': (0.85, 0.4),
    'T7': (-0.95, 0.15), 'C5': (-0.6, 0.15), 'C3': (-0.35, 0.15),
    'C1': (-0.15, 0.15), 'CZ': (0.0, 0.15), 'C2': (0.15, 0.15),
    'C4': (0.35, 0.15), 'C6': (0.6, 0.15), 'T8': (0.95, 0.15),
    'TP7': (-0.85, -0.1), 'CP5': (-0.6, -0.1), 'CP3': (-0.35, -0.1),
    'CP1': (-0.15, -0.1), 'CPZ': (0.0, -0.1), 'CP2': (0.15, -0.1),
    'CP4': (0.35, -0.1), 'CP6': (0.6, -0.1), 'TP8': (0.85, -0.1),
    'P7': (-0.8, -0.35), 'P5': (-0.55, -0.35), 'P3': (-0.35, -0.35),
    'P1': (-0.15, -0.35), 'PZ': (0.0, -0.35), 'P2': (0.15, -0.35),
    'P4': (0.35, -0.35), 'P6': (0.55, -0.35), 'P8': (0.8, -0.35),
    'PO7': (-0.7, -0.55), 'PO5': (-0.45, -0.55), 'PO3': (-0.25, -0.55),
    'POZ': (0.0, -0.55), 'PO4': (0.25, -0.55), 'PO6': (0.45, -0.55),
    'PO8': (0.7, -0.55),
    'O1': (-0.3, -0.75), 'OZ': (0.0, -0.75), 'O2': (0.3, -0.75),
}


def plot_topomap(values, channel_names, title='Topomap', vmin=None, vmax=None, cmap='RdBu_r'):
    """Plot EEG scalp topography using 10-20 coordinates and bilinear interpolation.

    Args:
        values: [C] numpy array, one value per channel
        channel_names: list of channel name strings
        title: plot title
        vmin, vmax: color scale range (auto if None)
        cmap: matplotlib colormap
    Returns:
        matplotlib Figure
    """
    coords = []
    vals = []
    for val, name in zip(values, channel_names):
        key = name.upper()
        if key in _ch_coords_2d:
            coords.append(_ch_coords_2d[key])
            vals.append(val)
    coords = np.array(coords)
    vals = np.array(vals)

    if vmin is None:
        vmin = float(np.nanmin(vals))
    if vmax is None:
        vmax = float(np.nanmax(vals))

    fig, ax = plt.subplots(figsize=(5, 5))
    # Draw head outline
    theta = np.linspace(0, 2 * np.pi, 200)
    head_x = np.cos(theta) * 1.02
    head_y = np.sin(theta) * 1.02
    ax.plot(head_x, head_y, 'k-', lw=1.5)
    # Nose
    ax.plot([-0.05, 0, 0.05], [-1.05, -1.15, -1.05], 'k-', lw=1.5)
    # Ears
    ax.plot([-1.06, -1.02], [0.1, 0], 'k-', lw=1.5)
    ax.plot([-1.06, -1.02], [-0.1, 0], 'k-', lw=1.5)
    ax.plot([1.06, 1.02], [0.1, 0], 'k-', lw=1.5)
    ax.plot([1.06, 1.02], [-0.1, 0], 'k-', lw=1.5)

    # Interpolate to grid
    grid_x = np.linspace(-1.0, 1.0, 100)
    grid_y = np.linspace(-1.0, 1.0, 100)
    gx, gy = np.meshgrid(grid_x, grid_y)
    # Simple IDW interpolation
    interp = _idw_interp(gx, gy, coords, vals)
    # Mask outside head
    mask = (gx ** 2 + gy ** 2) > 1.0
    interp[mask] = np.nan

    im = ax.pcolormesh(gx, gy, interp, cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax),
                       shading='auto', rasterized=True)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.scatter(coords[:, 0], coords[:, 1], c='k', s=8, zorder=5)
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    return fig


def _idw_interp(gx, gy, points, values, power=3):
    """Inverse distance weighted interpolation."""
    result = np.zeros_like(gx)
    weights_sum = np.zeros_like(gx)
    for (px, py), val in zip(points, values):
        dist = np.sqrt((gx - px) ** 2 + (gy - py) ** 2) + 1e-8
        w = 1.0 / (dist ** power)
        result += val * w
        weights_sum += w
    return result / weights_sum


def plot_class_topomaps(features_flat, labels_flat, channel_names, class_names=None,
                        title_prefix='', fea_mode='mean'):
    """Plot average topomap per class.

    Args:
        features_flat: [N, C] features per sample (time-averaged per channel)
        labels_flat: [N] integer labels
        channel_names: list of C channel names
    Returns:
        matplotlib Figure
    """
    n_classes = len(np.unique(labels_flat))
    if class_names is None:
        class_names = [f'Class {i}' for i in range(n_classes)]
    cols = min(n_classes, 4)
    rows = int(np.ceil(n_classes / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4))
    if rows * cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for cls in range(n_classes):
        mask = labels_flat == cls
        if fea_mode == 'mean':
            values = features_flat[mask].mean(axis=0)
        else:
            values = np.var(features_flat[mask], axis=0)
        _quick_topomap_on_ax(axes[cls], values, channel_names,
                             title=f'{class_names[cls]}', cmap='RdBu_r')

    for j in range(n_classes, len(axes)):
        axes[j].axis('off')
    fig.suptitle(f'{title_prefix} Class Topomaps ({fea_mode})', fontsize=14)
    plt.tight_layout()
    return fig


def _quick_topomap_on_ax(ax, values, channel_names, title='', cmap='RdBu_r'):
    """Lightweight topomap on given axis (no extra outline)."""
    coords, vals = [], []
    for val, name in zip(values, channel_names):
        if name.upper() in _ch_coords_2d:
            coords.append(_ch_coords_2d[name.upper()])
            vals.append(val)
    coords = np.array(coords)
    vals = np.array(vals)
    vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
    gx = np.linspace(-1.0, 1.0, 80)
    gy = np.linspace(-1.0, 1.0, 80)
    gx, gy = np.meshgrid(gx, gy)
    interp = _idw_interp(gx, gy, coords, vals)
    interp[(gx ** 2 + gy ** 2) > 1.0] = np.nan
    ax.pcolormesh(gx, gy, interp, cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax),
                  shading='auto', rasterized=True)
    ax.scatter(coords[:, 0], coords[:, 1], c='k', s=5)
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.axis('off')


# ── B. t-SNE Embeddings ────────────────────────────────────────────────────────

def plot_tsne(features, labels, title='t-SNE', max_samples=2000, perplexity=30):
    """t-SNE visualization of feature space.
    Args:
        features: [N, D] numpy array
        labels: [N] integer labels
    Returns: matplotlib Figure
    """
    if features.shape[0] > max_samples:
        idx = np.random.choice(features.shape[0], max_samples, replace=False)
        features = features[idx]
        labels = labels[idx]

    tsne = TSNE(n_components=2, perplexity=min(perplexity, features.shape[0] // 3),
                random_state=42, n_jobs=-1)
    embedded = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(7, 6))
    n_classes = len(np.unique(labels))
    cmap = plt.cm.tab10
    for cls in range(n_classes):
        mask = labels == cls
        ax.scatter(embedded[mask, 0], embedded[mask, 1], c=[cmap(cls % 10)],
                   label=f'Class {cls}', alpha=0.6, s=8)
    ax.legend(markerscale=2, loc='best')
    ax.set_title(title)
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    plt.tight_layout()
    return fig


# ── C. Confusion Matrix + ROC ──────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, class_names=None, title='Confusion Matrix'):
    """Plot normalized confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)
    n = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n)]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45)
    ax.set_yticklabels(class_names)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{cm[i, j]}', ha='center', va='center',
                    fontsize=8, color='white' if cm_norm[i, j] > 0.5 else 'black')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_roc_curves(y_true, y_prob, class_names=None, title='ROC Curves'):
    """Plot per-class ROC curves."""
    n_classes = y_prob.shape[1]
    if class_names is None:
        class_names = [f'Class {i}' for i in range(n_classes)]
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

    fig, ax = plt.subplots(figsize=(6, 5))
    for cls in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, cls], y_prob[:, cls])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{class_names[cls]} (AUC={roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig


# ── D. Adjacency Heatmap ───────────────────────────────────────────────────────

def plot_adjacency_heatmap(adj, title='Adjacency Matrix', channel_names=None):
    """Plot adjacency matrix as heatmap."""
    if isinstance(adj, torch.Tensor):
        adj = adj.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(adj, cmap='Reds', aspect='auto', vmin=0)
    plt.colorbar(im, ax=ax)
    if channel_names is not None and len(channel_names) == adj.shape[0]:
        step = max(1, len(channel_names) // 10)
        ticks = range(0, len(channel_names), step)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([channel_names[i] for i in ticks], rotation=45, fontsize=6)
        ax.set_yticklabels([channel_names[i] for i in ticks], fontsize=6)
    ax.set_title(title)
    plt.tight_layout()
    return fig


# ── F. Per-subject Heatmap ─────────────────────────────────────────────────────

def plot_subject_heatmap(acc_matrix, title='Per-Subject Accuracy', xlabel='Fold', ylabel='Subject'):
    """Plot per-subject accuracy heatmap.
    Args:
        acc_matrix: [n_subs, n_folds] accuracy per subject per fold
    Returns: matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(max(6, acc_matrix.shape[1] * 1.2),
                                    max(4, acc_matrix.shape[0] * 0.4)))
    im = ax.imshow(acc_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label='Accuracy')
    for i in range(acc_matrix.shape[0]):
        for j in range(acc_matrix.shape[1]):
            ax.text(j, i, f'{acc_matrix[i, j]:.2f}', ha='center', va='center', fontsize=7)
    ax.set_xticks(range(acc_matrix.shape[1]))
    ax.set_yticks(range(acc_matrix.shape[0]))
    ax.set_xticklabels([f'Fold {j}' for j in range(acc_matrix.shape[1])])
    ax.set_yticklabels([f'Sub {i}' for i in range(acc_matrix.shape[0])])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    return fig


# ── Helpers ────────────────────────────────────────────────────────────────────

def fig_to_image(fig):
    """Convert matplotlib figure to numpy image array (H, W, 3)."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = plt.imread(buf)
    plt.close(fig)
    return img


def fig_to_tensor(fig):
    """Convert matplotlib figure to [3, H, W] tensor for TensorBoard."""
    img = fig_to_image(fig)
    return torch.from_numpy(img).permute(2, 0, 1)
