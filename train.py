# -*- coding: utf-8 -*-
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import KFold
from collections import Counter
from typing import Any, Dict, List, Optional, Union
import matplotlib.pyplot as plt

from dataset import CustomDataset
from model.Histopath25D_MTL_MoE import Histopath25D_MTL_MoE

from utils.survival_loss import (
    NLLSurvLoss,
    compute_nll_bin_edges,
    discrete_survival_logits_to_risk,
    map_time_to_bins_with_edges,
)
from sksurv.metrics import concordance_index_censored

VALID_PATCH_AGGREGATORS = (
    'mean',
    'max',
    'mlp_mean',
    'transmil',
    'abmil',
    'dgrmil',
    'dsmil',
    'ilra',
    'hvtsurv',
    'acmil',
    'attrimil',
    's4mil',
)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# -------------------- Hyperparameters / paths --------------------
NUM_EPOCHS = 15
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-5
OPTIMIZER = 'Adam'

FEATURE_DIM = 1280
NUM_VIEWS = 7
LOCAL_OUT_DIM = 512    # Bag embedding dim; many MIL heads use 512 or 1024 depending on architecture.
LOCAL_MOE_MODE = 'patch'  # 'bag' or 'patch'
LOCAL_NUM_EXPERTS = 4
LOCAL_TOP_K = 1
LOCAL_FEATURE_LENGTH = 1
PATCH_AGGREGATOR = 'attrimil'  # choose from 'mean'|'max'|'mlp_mean'|'transmil'|'abmil'|'dgrmil'|'dsmil'|'ilra'|'hvtsurv'|'acmil'|'attrimil'|'s4mil'
PATCH_EXPERT_HIDDEN_DIM = None

EXPERT_DIM = 256
INTERNAL_STAGES = 5


NUM_CPS_CLS = 2
NUM_CLAUDING_CLS = 2

DATA_PATH = ''
LOG_ROOT = ''
KFold_SEED = 1546

COSINE_T_MAX = NUM_EPOCHS
COSINE_ETA_MIN = 5e-7
USE_FIXED_LEARNING_RATE = False

# Survival head uses NLLSurvLoss with discrete bins (quantiles from training times).
NLL_NUM_BINS = 4

TASK_EMBED_DIM = 64
MASK_DISCARD_THRESHOLD = 0.65

MASK_EXPERT_HIDDEN_DIM = None



def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ==================== Feature augmentation (training only) ====================

def _aug_gaussian_noise(feats: List[torch.Tensor], noise_level: float = 0.05) -> List[torch.Tensor]:
    """Add Gaussian noise to each view tensor."""
    return [f + torch.randn_like(f) * noise_level for f in feats]


def _aug_random_scale(feats: List[torch.Tensor], scale_range: tuple = (0.9, 1.1)) -> List[torch.Tensor]:
    """Per-view multiplicative scaling (one scalar per view, shared across patches)."""
    lo, hi = scale_range
    return [f * (lo + (hi - lo) * torch.rand(1, device=f.device)) for f in feats]


def _aug_random_dropout(feats: List[torch.Tensor], drop_prob: float = 0.1) -> List[torch.Tensor]:
    """Feature-channel dropout with probability drop_prob."""
    return [f * (torch.rand_like(f) > drop_prob).float() for f in feats]


def _aug_patch_shuffle(feats: List[torch.Tensor]) -> List[torch.Tensor]:
    """Randomly shuffle patches along N for each view."""
    shuffled = []
    for f in feats:
        if f.dim() == 3:  # (B, N, D)
            idx = torch.randperm(f.shape[1], device=f.device)
            shuffled.append(f[:, idx, :])
        elif f.dim() == 2:  # (N, D)
            idx = torch.randperm(f.shape[0], device=f.device)
            shuffled.append(f[idx, :])
        else:
            shuffled.append(f)
    return shuffled


def _aug_view_mixup(feats: List[torch.Tensor], alpha: float = 0.2) -> List[torch.Tensor]:
    lam = float(np.random.beta(alpha, alpha))
    mixed = []
    n = len(feats)
    for i, f in enumerate(feats):
        neighbor = feats[(i + 1) % n]
        # Align patch counts by truncating to min(N) for shape compatibility.
        if f.dim() == 3 and neighbor.dim() == 3:
            min_n = min(f.shape[1], neighbor.shape[1])
            mixed.append(lam * f[:, :min_n, :] + (1 - lam) * neighbor[:, :min_n, :])
        else:
            mixed.append(f)
    return mixed


# Toggle augmentation strengths in this block.
AUG_NOISE_LEVEL    = 0.3   # Gaussian noise std coefficient; set 0 to disable.
AUG_SCALE_RANGE    = (0.9, 1.1)  # Random scale range.
AUG_DROP_PROB      = 0.1    # Feature dropout probability; set 0 to disable.
AUG_PATCH_SHUFFLE  = True   # Shuffle patch order within each view.
AUG_MIXUP_ALPHA    = 0.2    # Beta(alpha, alpha) for view Mixup; set 0 to disable.


def augment_features(
    feats: List[torch.Tensor],
    noise_level: float = AUG_NOISE_LEVEL,
    scale_range: tuple = AUG_SCALE_RANGE,
    drop_prob: float = AUG_DROP_PROB,
    patch_shuffle: bool = AUG_PATCH_SHUFFLE,
    mixup_alpha: float = AUG_MIXUP_ALPHA,
) -> List[torch.Tensor]:
    """
    Apply augmentation pipeline in fixed order:
      1. Gaussian noise if noise_level > 0
      2. Random scale if scale_range is not None
      3. Feature dropout if drop_prob > 0
      4. Patch shuffle if patch_shuffle
      5. View Mixup if mixup_alpha > 0

    Args:
        feats: List of (B,N,D) or (N,D) tensors, length NUM_VIEWS.
        noise_level: Gaussian noise scale.
        scale_range: (low, high) multiplicative scaling.
        drop_prob: Probability of zeroing each feature element.
        patch_shuffle: Whether to shuffle patch dimension per view.
        mixup_alpha: Beta parameter; Mixup skipped if <= 0.

    Returns:
        Augmented feat list (same length; N may shrink after Mixup).
    """
    if noise_level > 0:
        feats = _aug_gaussian_noise(feats, noise_level)
    if scale_range is not None:
        feats = _aug_random_scale(feats, scale_range)
    if drop_prob > 0:
        feats = _aug_random_dropout(feats, drop_prob)
    if patch_shuffle:
        feats = _aug_patch_shuffle(feats)
    if mixup_alpha > 0:
        feats = _aug_view_mixup(feats, mixup_alpha)
    return feats


# ==================== /Feature augmentation ====================


def multiclass_to_multilabel(label):
    table = {0: [1, 0], 1: [0, 1], 2: [1, 1]}
    return table[int(label)]


def _survival_head_out_dim():
    return int(NLL_NUM_BINS)


def _nll_bin_edges_from_train_data(train_data):
    times = []
    for s in train_data:
        t = s['survival_time']
        if torch.is_tensor(t):
            t = float(t.detach().cpu().item())
        times.append(float(t))
    edges = compute_nll_bin_edges(np.array(times, dtype=np.float64), NLL_NUM_BINS)
    print(f'NLL: bin edges ({NLL_NUM_BINS} bins), edges={edges}')
    return edges


def process_surv(logits, label, censorship, loss_fn: NLLSurvLoss, nll_bin_edges):
    """Discrete-time NLL survival loss (NLLSurvLoss only; no Cox branch)."""
    if nll_bin_edges is None:
        raise ValueError('NLLSurvLoss requires nll_bin_edges (quantiles from training times).')
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    y = map_time_to_bins_with_edges(label.view(-1), nll_bin_edges).view(-1, 1)
    c = censorship.view(-1, 1).long()
    surv_loss_dict = loss_fn(logits, y, c)
    risk = discrete_survival_logits_to_risk(logits)
    return {
        'logits': logits,
        'risk': risk,
        'loss': surv_loss_dict['loss'],
    }




def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_eval_metrics(
    targets_all: Union[List[int], np.ndarray],
    preds_all: Union[List[int], np.ndarray],
    probs_all: Optional[Union[List[float], np.ndarray]] = None,
    unique_classes: Optional[List[int]] = None,
    prefix: str = "",
) -> Dict[str, Any]:
    """
    Classification metrics restricted to AUROC and F1 (weighted).

    Returns:
        ``{prefix}/weighted_f1`` always; ``{prefix}/auroc`` when ``probs_all`` is given and
        there is more than one distinct class in ``targets_all``.
    """
    targets_all = np.array(targets_all)
    preds_all = np.array(preds_all)
    unique_classes = (
        unique_classes
        if unique_classes is not None
        else sorted(np.unique(targets_all).tolist())
    )

    metrics: Dict[str, Any] = {
        f'{prefix}/weighted_f1': float(
            f1_score(
                targets_all,
                preds_all,
                average='weighted',
                labels=unique_classes,
                zero_division=0,
            )
        ),
    }

    if probs_all is None or len(np.unique(targets_all)) <= 1:
        return metrics

    probs_all = np.array(probs_all)
    if probs_all.ndim == 1:
        probs_all = np.column_stack([1.0 - probs_all, probs_all])

    n_cls_labels = len(unique_classes)
    try:
        is_binary = (probs_all.shape[1] == 2) and (n_cls_labels == 2)
        if is_binary:
            auroc = roc_auc_score(targets_all, probs_all[:, 1])
        else:
            auroc = roc_auc_score(
                targets_all,
                probs_all,
                labels=unique_classes,
                multi_class='ovr',
                average='macro',
            )
        metrics[f'{prefix}/auroc'] = float(auroc)
    except Exception as e:
        import warnings
        warnings.warn(f'[{prefix}] AUROC failed: {e}')

    return metrics


def evaluate_one_epoch(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    nll_bin_edges: np.ndarray,
    fold_idx: int,
    epoch: int,
) -> Dict[str, Any]:
    """
    Run validation forward pass once and aggregate metrics.

    Includes three classification heads (CPS, Clauding, Lauren): AUROC and weighted F1 where applicable;
    survival C-index from discrete-risk outputs.

    Args:
        model: Model in eval() mode.
        val_loader: Validation DataLoader.
        nll_bin_edges: Bin boundaries for NLL survival from training-time quantiles.
        fold_idx: Fold index (for tqdm description).
        epoch: Epoch index (for tqdm description).

    Returns:
        Dict of metrics plus mean_score (average of four headline scores for checkpointing).
    """
    model.eval()

    # Collect predictions
    all_labels_cps, all_probs_cps, all_preds_cps = [], [], []
    all_labels_clauding, all_probs_clauding, all_preds_clauding = [], [], []
    all_labels_lauren, all_probs_lauren = [], []
    all_risk_scores, all_censorships, all_event_times = [], [], []

    with torch.no_grad():
        for batch_idx, (ids, boundary_features, tumor_features, labels) in tqdm(
            enumerate(val_loader),
            desc=f'Fold{fold_idx + 1} Val E{epoch}',
            total=len(val_loader),
        ):
            if not tumor_features or len(tumor_features) != NUM_VIEWS:
                continue

            feats = prepare_taskmoe_inputs(tumor_features, 'cuda')
            outputs, _ = model(feats)

            # ── CPS ──────────────────────────────────────────────────────────
            labels_cps = labels['CPS_label'].cuda()
            probs_cps = F.softmax(outputs[0], dim=1)
            preds_cps = torch.argmax(probs_cps, dim=1)
            all_labels_cps.extend(labels_cps.cpu().numpy())
            all_probs_cps.extend(probs_cps.cpu().numpy())
            all_preds_cps.extend(preds_cps.cpu().numpy())

            # ── Clauding ─────────────────────────────────────────────────────
            labels_clauding = labels['Clauding_label'].cuda()
            probs_clauding = F.softmax(outputs[1], dim=1)
            preds_clauding = torch.argmax(probs_clauding, dim=1)
            all_labels_clauding.extend(labels_clauding.cpu().numpy())
            all_probs_clauding.extend(probs_clauding.cpu().numpy())
            all_preds_clauding.extend(preds_clauding.cpu().numpy())

            # ── Lauren (multi-label) ────────────────────────────────────────────
            labels_lauren_ml = torch.tensor(
                [multiclass_to_multilabel(lbl.item()) for lbl in labels['lauren_label']],
                dtype=torch.float32,
            ).cuda()
            probs_lauren = torch.sigmoid(outputs[2])
            all_labels_lauren.extend(labels_lauren_ml.cpu().numpy())
            all_probs_lauren.extend(probs_lauren.cpu().numpy())

            # ── Survival (batch=1; risk each step) ─────────────────────────────
            risk = discrete_survival_logits_to_risk(outputs[3])
            all_risk_scores.append(risk.detach().cpu().numpy().reshape(-1))
            all_censorships.append(labels['survival_status'].cpu().numpy())
            all_event_times.append(labels['survival_time'].cpu().numpy())

    # ── CPS metrics ────────────────────────────────────────────────────────────
    metrics = get_eval_metrics(
        targets_all=all_labels_cps,
        preds_all=all_preds_cps,
        probs_all=all_probs_cps,
        unique_classes=[0, 1],
        prefix='CPS',
    )

    # ── Clauding metrics ───────────────────────────────────────────────────────
    metrics.update(get_eval_metrics(
        targets_all=all_labels_clauding,
        preds_all=all_preds_clauding,
        probs_all=all_probs_clauding,
        unique_classes=[0, 1],
        prefix='Clauding',
    ))

    # ── Lauren AUROC (multi-label mean of two columns) ─────────────────────────
    all_labels_lauren = np.array(all_labels_lauren)
    all_probs_lauren = np.array(all_probs_lauren)
    try:
        auc_lauren = (
            roc_auc_score(all_labels_lauren[:, 0], all_probs_lauren[:, 0])
            + roc_auc_score(all_labels_lauren[:, 1], all_probs_lauren[:, 1])
        ) / 2.0
    except ValueError:
        auc_lauren = 0.0
    metrics['Lauren/auroc'] = float(auc_lauren)

    preds_lauren_bin = (all_probs_lauren >= 0.5).astype(np.int64)
    try:
        lauren_macro_f1 = f1_score(
            all_labels_lauren, preds_lauren_bin, average='macro', zero_division=0
        )
    except ValueError:
        lauren_macro_f1 = 0.0
    metrics['Lauren/macro_f1'] = float(lauren_macro_f1)

    # ── Survival C-index ─────────────────────────────────────────────────────
    all_rs = np.concatenate(all_risk_scores).reshape(-1)
    all_ce = np.concatenate(all_censorships).reshape(-1)
    all_et = np.concatenate(all_event_times).reshape(-1)
    c_index = concordance_index_censored(
        (1 - all_ce).astype(bool), all_et, all_rs, tied_tol=1e-08
    )[0]
    metrics['Survival/c_index'] = float(c_index)

    # ── Combined score: mean of CPS AUROC, Clauding AUROC, Lauren AUROC, C-index ──
    mean_auc_cps = metrics.get('CPS/auroc', 0.0)
    mean_auc_clauding = metrics.get('Clauding/auroc', 0.0)
    metrics['mean_score'] = (mean_auc_cps + mean_auc_clauding + auc_lauren + c_index) / 4.0

    return metrics


def prepare_taskmoe_inputs(tumor_features, device, num_views=None):
    """Map raw tumor tensors to a list of (B,N,D) on ``device``; length = NUM_VIEWS."""
    expected = NUM_VIEWS if num_views is None else int(num_views)
    out = []
    for f in tumor_features:
        x = f.to(device, non_blocking=True)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        elif x.dim() != 3:
            raise ValueError(f'Expected (N,D) or (B,N,D), got dim={x.dim()} shape={tuple(x.shape)}')
        out.append(x)
    if len(out) != expected:
        raise ValueError(f'Expected {expected} views, got {len(out)}')
    return out



def train_model_single_fold(fold_idx, train_data, val_data, model_save_path):
    clear_gpu_cache()
      
    nll_bin_edges = _nll_bin_edges_from_train_data(train_data)
    survival_head_dim = _survival_head_out_dim()

    train_dataset = CustomDataset(train_data)
    val_dataset = CustomDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = _build_model(survival_head_dim)

    print(f'\nparameters: {count_parameters(model):,}\n')

    criterion = nn.CrossEntropyLoss()
    criterion_lauren = nn.BCEWithLogitsLoss()
    survival_criterion = NLLSurvLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    if USE_FIXED_LEARNING_RATE:
        scheduler = None
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=COSINE_T_MAX, eta_min=COSINE_ETA_MIN
        )

    best_auc = 0.0
    best_c_index = 0.0
    best_snap = {
        'cps': 0.0,
        'clauding': 0.0,
        'lauren': 0.0,
        'c_index': 0.0,
        'cps_f1': 0.0,
        'clauding_f1': 0.0,
        'lauren_f1': 0.0,
    }
    fold_train_losses = []
    fold_val_scores = []

    for epoch in range(NUM_EPOCHS):
        clear_gpu_cache()
        model.train()
        train_loss, n_steps = 0.0, 0
        all_risk_scores, all_censorships, all_event_times = [], [], []

        for batch_idx, (ids, boundary_features, tumor_features, labels) in tqdm(
            enumerate(train_loader), desc=f'Fold{fold_idx + 1} E{epoch}', total=len(train_loader),
        ):
            if not tumor_features or len(tumor_features) != NUM_VIEWS:
                continue

            labels_cps = labels['CPS_label'].cuda()
            labels_clauding = labels['Clauding_label'].cuda()
            labels_lauren = torch.tensor(
                [multiclass_to_multilabel(lbl.item()) for lbl in labels['lauren_label']],
                dtype=torch.float32,
            ).cuda()
            survival_label = labels['survival_time'].cuda()
            censorship = labels['survival_status'].cuda()

            feats = prepare_taskmoe_inputs(tumor_features, 'cuda')
            feats = augment_features(feats)
            outputs, _ = model(feats)

            loss_cps = criterion(outputs[0], labels_cps)
            loss_clauding = criterion(outputs[1], labels_clauding)
            loss_lauren = criterion_lauren(outputs[2], labels_lauren)

            surv_dict = process_surv(
                outputs[3],
                survival_label.unsqueeze(1),
                censorship.unsqueeze(1),
                survival_criterion,
                nll_bin_edges=nll_bin_edges,
            )
            total_loss = loss_cps + loss_clauding + loss_lauren + surv_dict['loss']

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()
            n_steps += 1
            all_risk_scores.append(surv_dict['risk'].detach().cpu().numpy().reshape(-1))
            all_censorships.append(censorship.cpu().numpy())
            all_event_times.append(survival_label.cpu().numpy())

        train_loss_avg = train_loss / max(n_steps, 1)
        if all_risk_scores:
            all_rs = np.concatenate(all_risk_scores).reshape(-1)
            all_ce = np.concatenate(all_censorships)
            all_et = np.concatenate(all_event_times)
            train_c_index = concordance_index_censored(
                (1 - all_ce).astype(bool), all_et, all_rs, tied_tol=1e-08,
            )[0]
        else:
            train_c_index = 0.0

        # ---------- Validation ----------
        val_metrics = evaluate_one_epoch(
            model=model,
            val_loader=val_loader,
            nll_bin_edges=nll_bin_edges,
            fold_idx=fold_idx,
            epoch=epoch,
        )
        mean_auc = val_metrics['mean_score']
        mean_auc_cps = val_metrics.get('CPS/auroc', 0.0)
        mean_auc_clauding = val_metrics.get('Clauding/auroc', 0.0)
        auc_lauren = val_metrics.get('Lauren/auroc', 0.0)
        test_c_index = val_metrics.get('Survival/c_index', 0.0)
        cps_f1 = val_metrics.get('CPS/weighted_f1', 0.0)
        clauding_f1 = val_metrics.get('Clauding/weighted_f1', 0.0)
        lauren_f1 = val_metrics.get('Lauren/macro_f1', 0.0)

        lr_now = optimizer.param_groups[0]['lr']
        print(
            f'Fold {fold_idx + 1} Epoch {epoch}: train_loss={train_loss_avg:.4f}, train_c={train_c_index:.4f}, '
            f'mean_score={mean_auc:.4f}, '
            f'CPS AUC={mean_auc_cps:.4f} F1={cps_f1:.4f} | '
            f'Clauding AUC={mean_auc_clauding:.4f} F1={clauding_f1:.4f} | '
            f'Lauren AUC={auc_lauren:.4f} F1={lauren_f1:.4f} | '
            f'C-index={test_c_index:.4f}, lr={lr_now:.2e}'
        )

        if mean_auc > best_auc:
            best_auc = mean_auc
            best_c_index = test_c_index
            best_snap = {
                'cps': mean_auc_cps,
                'clauding': mean_auc_clauding,
                'lauren': auc_lauren,
                'c_index': float(test_c_index),
                'cps_f1': cps_f1,
                'clauding_f1': clauding_f1,
                'lauren_f1': lauren_f1,
            }
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': (
                        scheduler.state_dict() if scheduler is not None else None
                    ),
                    'best_combined_score': best_auc,
                    'metrics': best_snap,
                    'val_metrics_full': val_metrics,
                },
                os.path.join(model_save_path, f'fold_{fold_idx + 1}_best_checkpoint.pth'),
            )

        if scheduler is not None:
            scheduler.step()
        fold_train_losses.append(train_loss_avg)
        fold_val_scores.append(mean_auc)

    return (
        best_auc,
        best_snap['cps'],
        0.0,
        best_snap['clauding'],
        best_snap['lauren'],
        best_c_index,
        best_snap.get('cps_f1', 0.0),
        best_snap.get('clauding_f1', 0.0),
        best_snap.get('lauren_f1', 0.0),
        fold_train_losses,
        fold_val_scores,
    )


def _build_model(survival_head_dim: int) -> nn.Module:
    """
    Build the pathology multi-view MIL backbone from global flags.
    ``Histopath25D_MTL_MoE`` 
    """
    
    return Histopath25D_MTL_MoE(
        feature_dim=FEATURE_DIM,
        local_out_dim=LOCAL_OUT_DIM,
        expert_dim=EXPERT_DIM,
        n_task=4,
        num_views=NUM_VIEWS,
        n_stages=INTERNAL_STAGES,
        task_head_out_dims=[2, 2, 2, survival_head_dim],
        use_local_fusion=True,
        local_moe_mode=LOCAL_MOE_MODE,
        local_num_experts=LOCAL_NUM_EXPERTS,
        local_top_k=LOCAL_TOP_K,
        local_feature_length=LOCAL_FEATURE_LENGTH,
        patch_aggregator=PATCH_AGGREGATOR,
        patch_expert_hidden_dim=PATCH_EXPERT_HIDDEN_DIM,
        task_embed_dim=TASK_EMBED_DIM,
        mask_discard_threshold=MASK_DISCARD_THRESHOLD,
        mask_expert_hidden_dim=MASK_EXPERT_HIDDEN_DIM,
    ).cuda()
   


def eval_model(checkpoint_dir: str, start_fold: int = 0):
    """
    Evaluation-only mode: same KFold split as ``train_model``, load each fold's best checkpoint,
    run ``evaluate_one_epoch`` on that fold's validation set, aggregate metrics across folds.

    Args:
        checkpoint_dir: Directory containing ``fold_*_best_checkpoint.pth`` (same as training ``model_save_path``).
        start_fold: First fold index to evaluate (skip earlier folds); default 0 runs all folds.
    """
    full_data = torch.load(DATA_PATH, weights_only=False)
    print(f'[eval_model] Total samples: {len(full_data)}')
    kfold = KFold(n_splits=5, shuffle=True, random_state=KFold_SEED)
    survival_head_dim = _survival_head_out_dim()

    fold_metrics_list: List[Dict[str, Any]] = []

    for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(full_data)):
        if fold_idx < start_fold:
            print(f'Skipping fold {fold_idx + 1}')
            continue

        train_data = [full_data[i] for i in train_indices]
        val_data = [full_data[i] for i in val_indices]

     
        # nll_bin_edges from training-fold times only (no label leakage from val augmentation).
        nll_bin_edges = _nll_bin_edges_from_train_data(train_data)

        val_dataset = CustomDataset(val_data)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        ckpt_path = os.path.join(checkpoint_dir, f'fold_{fold_idx + 1}_best_checkpoint.pth')
        if not os.path.isfile(ckpt_path):
            print(f'[eval_model] Fold {fold_idx + 1}: checkpoint missing -> {ckpt_path}, skip')
            continue

        print(f'\n========== [eval_model] Fold {fold_idx + 1} / 5 ==========')
        print(f'Val samples: {len(val_data)}')

        model = _build_model(survival_head_dim)
        ckpt = torch.load(ckpt_path, map_location='cuda', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        print(
            f'Loaded checkpoint: epoch={ckpt.get("epoch", "?")}, '
            f'best_score={ckpt.get("best_combined_score", 0.0):.4f}'
        )
        val_metrics = evaluate_one_epoch(
            model=model,
            val_loader=val_loader,
            nll_bin_edges=nll_bin_edges,
            fold_idx=fold_idx,
            epoch=ckpt.get('epoch', -1),
        )
        fold_metrics_list.append(val_metrics)

        print(
            f'Fold {fold_idx + 1}: mean_score={val_metrics["mean_score"]:.4f}, '
            f'CPS AUC={val_metrics.get("CPS/auroc", 0):.4f} F1={val_metrics.get("CPS/weighted_f1", 0):.4f} | '
            f'Clauding AUC={val_metrics.get("Clauding/auroc", 0):.4f} F1={val_metrics.get("Clauding/weighted_f1", 0):.4f} | '
            f'Lauren AUC={val_metrics.get("Lauren/auroc", 0):.4f} F1={val_metrics.get("Lauren/macro_f1", 0):.4f} | '
            f'C-index={val_metrics.get("Survival/c_index", 0):.4f}'
        )

        del model
        clear_gpu_cache()

    if not fold_metrics_list:
        print('[eval_model] No fold results available; exiting.')
        return

    def _mean(key):
        vals = [m.get(key, 0.0) for m in fold_metrics_list]
        return float(np.mean(vals)), float(np.std(vals))

    print('\n========== [eval_model] 5-fold summary ==========')
    for key, label in [
        ('mean_score', 'Combined'),
        ('CPS/auroc', 'CPS AUC'),
        ('Clauding/auroc', 'Clauding AUC'),
        ('Lauren/auroc', 'Lauren AUC'),
        ('Survival/c_index', 'C-index'),
        ('CPS/weighted_f1', 'CPS F1'),
        ('Clauding/weighted_f1', 'Clauding F1'),
        ('Lauren/macro_f1', 'Lauren F1 (macro)'),
    ]:
        mu, sd = _mean(key)
        print(f'  {label}: {mu:.4f} ± {sd:.4f}')

    result_path = os.path.join(checkpoint_dir, 'eval_results.txt')
    with open(result_path, 'w') as f:
        f.write('eval_model 5-fold results\n')
        f.write('=' * 50 + '\n')
        for i, m in enumerate(fold_metrics_list):
            f.write(
                f'Fold {i + 1}: combined={m["mean_score"]:.4f}, '
                f'CPS AUC={m.get("CPS/auroc", 0):.4f} F1={m.get("CPS/weighted_f1", 0):.4f}, '
                f'Clauding AUC={m.get("Clauding/auroc", 0):.4f} F1={m.get("Clauding/weighted_f1", 0):.4f}, '
                f'Lauren AUC={m.get("Lauren/auroc", 0):.4f} F1={m.get("Lauren/macro_f1", 0):.4f}, '
                f'C-index={m.get("Survival/c_index", 0):.4f}\n'
            )
        f.write('=' * 50 + '\n')
        for key, label in [
            ('mean_score', 'Combined'),
            ('CPS/auroc', 'CPS AUC'),
            ('CPS/weighted_f1', 'CPS F1'),
            ('Clauding/auroc', 'Clauding AUC'),
            ('Clauding/weighted_f1', 'Clauding F1'),
            ('Lauren/auroc', 'Lauren AUC'),
            ('Lauren/macro_f1', 'Lauren F1 (macro)'),
            ('Survival/c_index', 'C-index'),
        ]:
            mu, sd = _mean(key)
            f.write(f'{label}: {mu:.4f} ± {sd:.4f}\n')
    print(f'[eval_model] Results saved to {result_path}')


def train_model(start_fold=0):
    _tc_tag = 'TaskMoE'
    params = (
        f'Virchow2_{PATCH_AGGREGATOR}_{LOCAL_MOE_MODE}_lr{LEARNING_RATE}_'
        f'locE{LOCAL_NUM_EXPERTS}_locK{LOCAL_TOP_K}_5fold_locOut{LOCAL_OUT_DIM}_expert{EXPERT_DIM}'
    )
    model_save_path = os.path.join(LOG_ROOT, params)
    text_path = os.path.join(model_save_path, 'training_log.txt')
    os.makedirs(model_save_path, exist_ok=True)

    full_data = torch.load(DATA_PATH, weights_only=False)
    print(f'Total samples: {len(full_data)}')
    kfold = KFold(n_splits=5, shuffle=True, random_state=KFold_SEED)

    fold_results = {
        'best_combined_scores': [],
        'cps_aucs': [],
        'mmr_aucs': [],
        'clauding_aucs': [],
        'lauren_aucs': [],
        'survival_c_indices': [],
        'cps_f1s': [],
        'clauding_f1s': [],
        'lauren_f1s': [],
        'train_losses': [],
        'val_scores': [],
    }

    for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(full_data)):
        if fold_idx < start_fold:
            print(f'Skipping fold {fold_idx + 1}')
            continue
        print(f'\n========== Fold {fold_idx + 1} / 5 ==========')
        train_data = [full_data[i] for i in train_indices]
        val_data = [full_data[i] for i in val_indices]
        print(f'Train: {len(train_data)}, Val: {len(val_data)}')

        (
            best_combined,
            cps_auc,
            mmr_auc,
            clauding_auc,
            lauren_auc,
            best_c_index,
            cps_f1,
            clauding_f1,
            lauren_f1,
            train_losses,
            val_scores,
        ) = train_model_single_fold(fold_idx, train_data, val_data, model_save_path)

        fold_results['best_combined_scores'].append(best_combined)
        fold_results['cps_aucs'].append(cps_auc)
        fold_results['mmr_aucs'].append(mmr_auc)
        fold_results['clauding_aucs'].append(clauding_auc)
        fold_results['lauren_aucs'].append(lauren_auc)
        fold_results['survival_c_indices'].append(best_c_index)
        fold_results['cps_f1s'].append(cps_f1)
        fold_results['clauding_f1s'].append(clauding_f1)
        fold_results['lauren_f1s'].append(lauren_f1)
        fold_results['train_losses'].append(train_losses)
        fold_results['val_scores'].append(val_scores)
        print(
            f'Fold {fold_idx + 1} best: combined={best_combined:.4f}, C-index={best_c_index:.4f} | '
            f'CPS AUC={cps_auc:.4f} F1={cps_f1:.4f} | Clauding AUC={clauding_auc:.4f} F1={clauding_f1:.4f} | '
            f'Lauren AUC={lauren_auc:.4f} F1={lauren_f1:.4f}'
        )

    mean_best = np.mean(fold_results['best_combined_scores'])
    std_best = np.std(fold_results['best_combined_scores'])
    mean_cps = np.mean(fold_results['cps_aucs'])
    std_cps = np.std(fold_results['cps_aucs'])
    mean_cl = np.mean(fold_results['clauding_aucs'])
    std_cl = np.std(fold_results['clauding_aucs'])
    mean_lau = np.mean(fold_results['lauren_aucs'])
    std_lau = np.std(fold_results['lauren_aucs'])
    mean_ci = np.mean(fold_results['survival_c_indices'])
    std_ci = np.std(fold_results['survival_c_indices'])
    mean_cps_f1 = np.mean(fold_results['cps_f1s'])
    std_cps_f1 = np.std(fold_results['cps_f1s'])
    mean_cl_f1 = np.mean(fold_results['clauding_f1s'])
    std_cl_f1 = np.std(fold_results['clauding_f1s'])
    mean_lau_f1 = np.mean(fold_results['lauren_f1s'])
    std_lau_f1 = np.std(fold_results['lauren_f1s'])

    print('\n========== 5-fold summary ==========')
    print(f'Combined: {mean_best:.4f} ± {std_best:.4f}')
    print(f'CPS AUC: {mean_cps:.4f} ± {std_cps:.4f}')
    print(f'CPS F1:  {mean_cps_f1:.4f} ± {std_cps_f1:.4f}')
    print(f'Clauding AUC: {mean_cl:.4f} ± {std_cl:.4f}')
    print(f'Clauding F1:  {mean_cl_f1:.4f} ± {std_cl_f1:.4f}')
    print(f'Lauren AUC: {mean_lau:.4f} ± {std_lau:.4f}')
    print(f'Lauren F1:  {mean_lau_f1:.4f} ± {std_lau_f1:.4f}')
    print(f'C-index: {mean_ci:.4f} ± {std_ci:.4f}')

    with open(text_path, 'w') as f:
        f.write('=' * 50 + '\n')
        for i, row in enumerate(
            zip(
                fold_results['best_combined_scores'],
                fold_results['cps_aucs'],
                fold_results['clauding_aucs'],
                fold_results['lauren_aucs'],
                fold_results['survival_c_indices'],
                fold_results['cps_f1s'],
                fold_results['clauding_f1s'],
                fold_results['lauren_f1s'],
            )
        ):
            f.write(
                f'Fold {i + 1}: combined={row[0]:.4f}, CPS AUC={row[1]:.4f} F1={row[5]:.4f}, '
                f'Clauding AUC={row[2]:.4f} F1={row[6]:.4f}, Lauren AUC={row[3]:.4f} F1={row[7]:.4f}, '
                f'C-index={row[4]:.4f}\n'
            )
        f.write('=' * 50 + '\n')
        f.write(f'Mean combined: {mean_best:.4f} ± {std_best:.4f}\n')
        f.write(f'Mean CPS AUC: {mean_cps:.4f} ± {std_cps:.4f}\n')
        f.write(f'Mean CPS F1:  {mean_cps_f1:.4f} ± {std_cps_f1:.4f}\n')
        f.write(f'Mean Clauding AUC: {mean_cl:.4f} ± {std_cl:.4f}\n')
        f.write(f'Mean Clauding F1:  {mean_cl_f1:.4f} ± {std_cl_f1:.4f}\n')
        f.write(f'Mean Lauren AUC: {mean_lau:.4f} ± {std_lau:.4f}\n')
        f.write(f'Mean Lauren F1:  {mean_lau_f1:.4f} ± {std_lau_f1:.4f}\n')
        f.write(f'Mean C-index: {mean_ci:.4f} ± {std_ci:.4f}\n')




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode', choices=['train', 'eval'], default='train',
        help='train: fit model (default); eval: load checkpoints and evaluate validation folds only.',
    )

    parser.add_argument(
        '--checkpoint_dir', type=str, default=None,
    )
    
    parser.add_argument(
        '--start_fold', type=int, default=0,
        help='First fold index to run (0-based); default 0 processes all folds.',
    )

    parser.add_argument(
        '--patch-aggregator',
        type=str,
        default=None,
        choices=list(VALID_PATCH_AGGREGATORS),
        help=(
            "Override PATCH_AGGREGATOR: bag-level MIL head after per-patch MoE when LOCAL_MOE_MODE='patch'. "
            "Must match PatchLevelLocalFusionMoE.valid_agg."
        ),
    )
    args = parser.parse_args()

    if args.patch_aggregator is not None:
        globals()['PATCH_AGGREGATOR'] = args.patch_aggregator

    if args.mode == 'train':
        train_model(start_fold=args.start_fold)
    else:
        if args.checkpoint_dir is None:
            raise ValueError('eval mode requires --checkpoint_dir')
        eval_model(checkpoint_dir=args.checkpoint_dir, start_fold=args.start_fold)

