# AGENTS.md — mdJPT Training Framework

## Project

Multi-dataset joint pretraining for EEG emotion recognition. Hydra + PyTorch Lightning. Three-stage pipeline: pretrain → feature extraction → MLP evaluation.

## Key Commands

```bash
# Full pipeline (all 3 stages sequentially — no args; edit YAML or use CLI overrides)
bash run_pipeline.sh

# Individual stages
python train_multi.py log.run_name=mytest train.max_epochs=10 train.n_pairs=512
python ext_fea.py log.run_name=mytest val.extractor.ckpt_epoch=10
python train_mlp_full.py log.run_name=mytest val.extractor.ckpt_epoch=10

# Quick test (fast pass)
python train_multi.py train.max_epochs=1 train.n_fold=1 train.n_pairs=64 train.num_workers=4 log.run_name=quicktest

# Web UI
python webui/server.py   # → http://localhost:8080

# Thesis
cd thesis && typst compile thesis.typ   # requires CJK fonts in /usr/share/fonts/
```

## Architecture

```
train_multi.py          → MultiDataModule + MultiModel_PL + Lightning Trainer
  src/data/             → multi_dataloader.py (EEGSampler, MultiDataset), dataset.py (EEG_Dataset), io_utils.py (per-dataset loaders)
  src/model/            → MultiModel_PL.py (main model), MLLA_new.py (encoder), CNN_Attention.py (projector), granularity_gnn.py (GNN)
  src/loss/             → loss.py (SimCLR/CLISA), CDA_loss.py
  src/callbacks.py      → TSNECallback, AdjacencyCallback, GradientNormCallback
  src/visualization.py  → matplotlib charts (topomap, t-SNE, confusion, adjacency)
ext_fea.py              → loads ckpt, extracts features batch-by-batch (not trainer.predict)
train_mlp_full.py       → 6-fold LOO MLP evaluation
```

## Configuration

All params in `cfgs_multi/config_multi.yaml`. Datasets selected via Hydra defaults:
```yaml
defaults:
  - data@data_0: SEEDVII
  - data@data_1: SEEDIV
  - data@data_val: SEEDV
```
Each dataset has its own yaml in `cfgs_multi/data/`. Override any param on CLI: `train.lr=1e-3`.

## Critical Quirks

### GPU & Env
- `train_multi.py:5` hardcodes `CUDA_VISIBLE_DEVICES="0,2"` — edit if needed
- `train_mlp_full.py:4` hardcodes `CUDA_VISIBLE_DEVICES="0"`
- `ext_fea.py:2` hardcodes `CUDA_VISIBLE_DEVICES="0"`
- `train_multi.py:74` hardcodes `accelerator='gpu'`
- `train_mlp_full.py:90` uses `accelerator='auto'` (NOT `'gpu'`) due to GPU backend detection issue in certain envs
- `num_workers=0` requires `prefetch_factor` removed from DataLoader kwargs — `multi_dataloader.py:231-232` only sets `prefetch_factor` when `num_workers > 0`

### Checkpoint Naming
- Lightning saves epoch 0 as `epoch=00.ckpt`, epoch 1 as `epoch=01.ckpt`
- `ext_fea.py:57` uses `ckpt_epoch - 1` to find the right file (e.g., `ckpt_epoch=2` → `epoch=01.ckpt`)
- `save_top_k=-1` keeps ALL checkpoints (can accumulate disk usage)

### ext_fea OOM
- Original `trainer.predict()` loaded all samples into GPU memory → killed
- Fixed: replaced with per-batch forward loop, features moved to CPU after each batch
- Use `num_workers=0` in ext_fea to avoid DataLoader worker issues

### TensorBoard Logger
- Was a bug: `TensorBoardLogger` created but never passed to `pl.Trainer()` — no event files written
- Fixed in `train_multi.py:76`: logger now passed as `logger=logger`

### MLLA Device Bug
- `MLLA_new.py:65` — `to_patch()` hardcodes `.to('cuda')` instead of using the input tensor's device
- This is the actively used encoder (`MultiModel_PL.py:15` imports `channel_MLLA` from `MLLA_new`)
- Same bug exists in `MLLA.py:61` (backup copy, not currently imported)

### Typst Thesis
- Typst 0.14.2 via conda, CJK fonts (SimSun/SimHei/KaiTi/FangSong) must be in `/usr/share/fonts/truetype/`
- **No `$...$` inline LaTeX math** — use `#mi("...")` or plain Unicode math. Display formulas use `#mitex(...)` with backtick strings
- `_{...}` not allowed in text mode — only inside `$...$` or `#mitex()` or `_(...)` blocks

## Adding a New Dataset

1. Create `cfgs_multi/data/MYDATA.yaml` (copy `data_exaple.yaml` — note the typo in the filename)
2. Add loader function in `src/data/io_utils.py` following `load_processed_SEEDIV_NEW_data` pattern
3. Register it in `get_load_data_func()` at `io_utils.py:7`
4. Add it to `cfgs_multi/config_multi.yaml` defaults
5. Ensure data directory has `processed_data/` with per-subject `.mat` files

## Web UI

- Backend: `webui/server.py` (FastAPI, reads TensorBoard events, manages subprocess)
- Frontend: `webui/static/index.html` (Chart.js, polls /api/metrics every 3s)
- Settings panel loads `GET /api/config` from YAML tree
- Start training via `POST /api/train/start` with `overrides` dict of Hydra params
- Note: `webui/` is in `.gitignore`

## GNN Module

- Enable: `model.use_granularity_gnn=True`
- Config: `model.granularity_gnn.hidden_dim`, `n_layers`, `dist_sigma`
- 7 brain regions, 3 graph levels (global/intra/inter), learnable adjacency from 10-20 spatial distance prior
- GAT attention on global and inter-region graphs
