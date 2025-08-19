#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import re
import time
import json
import random
import warnings
from pathlib import Path
from typing import List, Set, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

warnings.filterwarnings('ignore')


# ============================== 随机种子 ==============================
def set_random_seed(seed: int = 25):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"[Seed] = {seed}")


def worker_init_fn(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================== 模型与损失 ==============================
class ImprovedFeatureFusionTransformer(nn.Module):
    def __init__(self, input_dim: int, reduced_dim: int, num_layers: int = 2, nhead: int = 4):
        super().__init__()
        self.shared_fc = nn.Linear(input_dim, reduced_dim)
        self.bn = nn.BatchNorm1d(reduced_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=reduced_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.modal_attention = nn.Linear(reduced_dim, 1)
        self.classifier = nn.Linear(reduced_dim, 3)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # features: (batch, modal, feature_dim)
        x = self.shared_fc(features)  # (batch, modal, reduced_dim)
        batch_size, modal_count, dim = x.shape
        x = x.view(-1, dim)
        x = self.bn(x)
        x = x.view(batch_size, modal_count, dim)
        transformer_out = self.transformer_encoder(x)  # (batch, modal, reduced_dim)
        attn_w = torch.softmax(self.modal_attention(transformer_out).squeeze(-1), dim=1)
        fused = torch.sum(transformer_out * attn_w.unsqueeze(-1), dim=1)
        logits = self.classifier(fused)
        return logits, attn_w, transformer_out


class MultiTaskLoss(nn.Module):
    def __init__(self, lambda1: float = 0.1, lambda2: float = 0.05):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.ce = nn.CrossEntropyLoss()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor,
                attn_w: torch.Tensor, modal_f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        L_ce = self.ce(preds, targets)
        L_mod = self.modal_balance_loss(modal_f, attn_w)
        L_att = self.attention_regularization_loss(attn_w)
        return L_ce + self.lambda1 * L_mod + self.lambda2 * L_att, L_ce, L_mod, L_att

    def modal_balance_loss(self, modal_f: torch.Tensor, attn_w: torch.Tensor) -> torch.Tensor:
        norms = torch.norm(modal_f, dim=2)  # (batch, modal)
        norms = F.normalize(norms, p=1, dim=1)
        return F.mse_loss(attn_w, norms)

    def attention_regularization_loss(self, attn_w: torch.Tensor) -> torch.Tensor:
        ent = -torch.sum(attn_w * torch.log(attn_w + 1e-8), dim=1)
        max_ent = torch.log(torch.tensor(attn_w.size(1), dtype=torch.float32, device=attn_w.device))
        return torch.mean(max_ent - ent)


# ============================== 解析工具 ==============================
DEFAULT_SET_PATTERNS = [
    r'(?:^|[_-])s(?P<sid>\d+)(?:[_-]|$)',     # s1_...
    r'(?:^|[_-])set(?P<sid>\d+)(?:[_-]|$)',   # set1_...
    r'(?:^|[_-])c(?P<sid>\d+)(?:[_-]|$)',     # c1_...   (你的示例 c1_023.npy)
]


def infer_set_id(stem: str, patterns: List[str]) -> Optional[int]:
    for p in patterns:
        m = re.search(p, stem, flags=re.IGNORECASE)
        if m:
            try:
                return int(m.group('sid'))
            except Exception:
                continue
    return None


def parse_label_from_stem(stem: str) -> int:
    # 假设最后一个 '_' 之后为 cut 数字，例如 c1_023 -> 23
    parts = stem.split('_')
    if len(parts) < 2:
        raise ValueError(f"无法从文件名解析 cut 值: {stem}")
    try:
        cut = int(parts[-1])
    except Exception:
        digits = re.findall(r'\d+', parts[-1])
        if not digits:
            raise
        cut = int(digits[-1])

    if cut <= 50:
        return 0
    elif cut <= 175:
        return 1
    else:
        return 2


# ============================== 数据集 ==============================
class FilteredNpyFeatureDataset(Dataset):
    def __init__(self, dirs: List[Path], allowed_set_ids: Set[int], set_patterns: List[str]):
        self.files: List[Path] = []
        for d in dirs:
            if d.exists():
                self.files.extend(sorted(d.glob("*.npy")))
        if not self.files:
            raise FileNotFoundError(f"未在这些目录中找到 .npy: {', '.join([str(d) for d in dirs])}")

        filtered = []
        for f in self.files:
            sid = infer_set_id(f.stem, set_patterns)
            if sid is None:
                continue
            if sid in allowed_set_ids:
                filtered.append(f)

        if not filtered:
            raise RuntimeError(f"根据 set_id 过滤后没有样本。允许的 set_id={sorted(list(allowed_set_ids))}，文件总数={len(self.files)}")
        self.files = filtered

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file = self.files[idx]
        data = np.load(file)  # (modal, feature_dim)
        feat = torch.tensor(data, dtype=torch.float32)
        label = parse_label_from_stem(file.stem)
        return feat, torch.tensor(label, dtype=torch.long)


# ============================== 训练/验证循环 ==============================
@torch.no_grad()
def validate_epoch(model: nn.Module, loader: DataLoader, criterion: MultiTaskLoss, device: torch.device) -> Dict[str, float]:
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for feats, labels in loader:
        feats, labels = feats.to(device), labels.to(device)
        preds, att_w, modal_f = model(feats)
        l, _, _, _ = criterion(preds, labels, att_w, modal_f)
        loss_sum += l.item() * labels.size(0)
        _, pred = preds.max(1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return {'val_loss': loss_sum / total, 'val_accuracy': correct / total}


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer,
                criterion: MultiTaskLoss, device: torch.device) -> Dict[str, float]:
    model.train()
    running = {'loss': 0.0, 'ce': 0.0, 'mod': 0.0, 'att': 0.0, 'correct': 0, 'total': 0}
    for feats, labels in loader:
        feats, labels = feats.to(device), labels.to(device)
        preds, att_w, modal_f = model(feats)
        loss, Lce, Lmod, Latt = criterion(preds, labels, att_w, modal_f)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        bs = labels.size(0)
        running['loss'] += loss.item() * bs
        running['ce'] += Lce.item() * bs
        running['mod'] += Lmod.item() * bs
        running['att'] += Latt.item() * bs
        _, pred = preds.max(1)
        running['correct'] += (pred == labels).sum().item()
        running['total'] += bs

    n = running['total']
    return {
        'train_loss': running['loss'] / n,
        'train_ce_loss': running['ce'] / n,
        'train_modal_loss': running['mod'] / n,
        'train_attention_loss': running['att'] / n,
        'train_accuracy': running['correct'] / n
    }


# ============================== 核心运行器 ==============================
def build_combo_dir(features_root: Path, resnet_arch: str) -> Path:
    # 固定最优组合
    mel_filters, window_ms, overlap_ratio = 128, 20, 0.5
    combo = f"mel{mel_filters}_w{window_ms}_o{int(overlap_ratio * 100)}_{resnet_arch}"
    return features_root / combo


def collect_candidate_dirs(combo_dir: Path) -> List[Path]:
    # 优先 all，其次 train/val，最后直接 combo 根
    all_dir = combo_dir / "all"
    train_dir = combo_dir / "train"
    val_dir = combo_dir / "val"

    if all_dir.exists():
        return [all_dir]
    cand = []
    if train_dir.exists():
        cand.append(train_dir)
    if val_dir.exists():
        cand.append(val_dir)
    if cand:
        return cand
    return [combo_dir]


def run_one_split(
    split_name: str,
    train_ids: List[int],
    val_ids: List[int],
    features_root: Path,
    resnet_arch: str,
    device: torch.device,
    batch_size: int,
    epochs: int,
    patience: int,
    seed: int,
    results_dir: Path,
    set_regex: Optional[str] = None
) -> Dict[str, float]:
    print(f"\n=== 运行划分: {split_name} | Train={train_ids} | Val={val_ids} ===")
    set_random_seed(seed)

    combo_dir = build_combo_dir(features_root, resnet_arch)
    if not combo_dir.exists():
        raise FileNotFoundError(f"组合目录不存在: {combo_dir}")

    cand_dirs = collect_candidate_dirs(combo_dir)
    print(f"使用目录: {', '.join([str(d) for d in cand_dirs])}")

    patterns = [set_regex] if set_regex else DEFAULT_SET_PATTERNS

    train_ds = FilteredNpyFeatureDataset(cand_dirs, set(train_ids), patterns)
    val_ds = FilteredNpyFeatureDataset(cand_dirs, set(val_ids), patterns)

    # 读取一个样本确定 feature_dim
    feat_dim = train_ds[0][0].shape[1]
    print(f"样本数: train={len(train_ds)}, val={len(val_ds)}, feature_dim={feat_dim}")

    # 构建 DataLoader
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2,
        pin_memory=torch.cuda.is_available(), worker_init_fn=worker_init_fn, generator=g
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2,
        pin_memory=torch.cuda.is_available(), worker_init_fn=worker_init_fn
    )

    # 模型/优化器/调度器/损失
    model = ImprovedFeatureFusionTransformer(input_dim=feat_dim, reduced_dim=128).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6)
    criterion = MultiTaskLoss(lambda1=0.1, lambda2=0.05)

    best_acc, best_epoch = 0.0, 0
    best_train_loss, best_val_loss = None, None
    best_train_acc = None
    early_stop = 0
    history = []

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_metrics['val_loss'])

        duration = time.time() - t0
        history.append({
            'split': split_name,
            'epoch': epoch,
            'train_ids': train_ids,
            'val_ids': val_ids,
            **train_metrics,
            **val_metrics,
            'epoch_time': duration
        })

        # 打印完整指标（包含 loss）
        print(
            f"  Epoch {epoch:02d} | "
            f"Train Loss={train_metrics['train_loss']:.4f} | Val Loss={val_metrics['val_loss']:.4f} | "
            f"Train Acc={train_metrics['train_accuracy']:.4f} | Val Acc={val_metrics['val_accuracy']:.4f} | "
            f"Time={duration:.1f}s"
        )

        if val_metrics['val_accuracy'] > best_acc:
            best_acc, best_epoch, early_stop = val_metrics['val_accuracy'], epoch, 0
            best_train_loss = train_metrics['train_loss']
            best_val_loss = val_metrics['val_loss']
            best_train_acc = train_metrics['train_accuracy']
            torch.save(
                {'model_state_dict': model.state_dict(), 'feat_dim': feat_dim},
                results_dir.joinpath(f'best_model_{split_name}.pth')
            )
        else:
            early_stop += 1

        if early_stop >= patience:
            print(f"  早停于 epoch {epoch}")
            break

    total_time = time.time() - t_start

    # 保存明细
    df_hist = pd.DataFrame(history)
    df_hist.to_csv(results_dir.joinpath(f'detailed_{split_name}.csv'), index=False)
    with open(results_dir.joinpath(f'detailed_{split_name}.json'), 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    torch.cuda.empty_cache()

    summary = {
        'split': split_name,
        'train_ids': train_ids,
        'val_ids': val_ids,
        'best_val_accuracy': best_acc,
        'best_epoch': best_epoch,
        'best_train_loss': best_train_loss,
        'best_val_loss': best_val_loss,
        'best_train_accuracy': best_train_acc,
        'total_time_sec': total_time,
        'train_size': len(train_ds),
        'val_size': len(val_ds),
        'feature_dim': feat_dim,
        'combo_dir': str(combo_dir),
        'candidate_dirs': [str(d) for d in cand_dirs]
    }
    return summary


def main():
    import argparse

    parser = argparse.ArgumentParser(description="验证最优组合在不同划分下的准确率")
    parser.add_argument("--features-root", type=str, default="D:\\py\\RPS-MODIFY\\PHM2010\\features",
                        help="预计算特征根目录（其下应包含 mel128_w20_o50_{arch}）")
    parser.add_argument("--results-dir", type=str, default="verify_results",
                        help="结果输出目录")
    parser.add_argument("--resnet-arch", type=str, default="resnet18",
                        help="用于生成特征的 ResNet 架构名（决定特征目录名的一部分）")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="设备: cuda 或 cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--set-regex", type=str, default=None,
                        help="可选，自定义解析 set_id 的正则（命名捕获组 sid），例如 '(?:_|-)unit(?P<sid>\\d+)(?:_|-|$)'")

    args = parser.parse_args()

    features_root = Path(args.features_root)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    splits = [
        ("train14_val6", [1, 4], [6]),
        ("train16_val4", [1, 6], [4]),
        ("train46_val1", [4, 6], [1]),
    ]

    summaries = []
    for name, train_ids, val_ids in splits:
        try:
            summary = run_one_split(
                split_name=name,
                train_ids=train_ids,
                val_ids=val_ids,
                features_root=features_root,
                resnet_arch=args.resnet_arch,
                device=device,
                batch_size=args.batch_size,
                epochs=args.epochs,
                patience=args.patience,
                seed=args.seed,
                results_dir=results_dir,
                set_regex=args.set_regex
            )
            summaries.append(summary)
        except Exception as e:
            print(f"[错误] 划分 {name} 失败: {e}")

    # 生成“每个划分一行”的汇总，满足列名要求
    if summaries:
        df = pd.DataFrame(summaries)

        # 你要求的列名映射（使用最优验证准确率所对应 epoch 的指标）
        out_df = pd.DataFrame({
            'Train id': [",".join(map(str, ids)) for ids in df['train_ids']],
            'Val id':   [",".join(map(str, ids)) for ids in df['val_ids']],
            'Train loss': df['best_train_loss'],
            'Val loss':   df['best_val_loss'],
            'Train acc':  df['best_train_accuracy'],
            'Val acc':    df['best_val_accuracy'],
        })
        out_df.to_csv(results_dir.joinpath("summary_metrics.csv"), index=False)

        # 原始 summary（包含更多元信息）
        df.sort_values(by="best_val_accuracy", ascending=False).to_csv(results_dir.joinpath("summary.csv"), index=False)
        with open(results_dir.joinpath("summary.json"), "w", encoding="utf-8") as f:
            json.dump(summaries, f, ensure_ascii=False, indent=2)

        # 合并所有 epoch 的明细为一个大表
        detailed_frames = []
        for name, _, _ in splits:
            f = results_dir.joinpath(f"detailed_{name}.csv")
            if f.exists():
                dfi = pd.read_csv(f)
                detailed_frames.append(dfi)
        if detailed_frames:
            pd.concat(detailed_frames, ignore_index=True).to_csv(results_dir.joinpath("all_detailed.csv"), index=False)

        print("\n=== 各划分最佳结果（按 Val Acc） ===")
        for s in summaries:
            print(
                f"{s['split']}: "
                f"Train id={','.join(map(str, s['train_ids']))} | Val id={','.join(map(str, s['val_ids']))} | "
                f"Train Loss={s['best_train_loss']:.4f} | Val Loss={s['best_val_loss']:.4f} | "
                f"Train Acc={s['best_train_accuracy']:.4f} | Val Acc={s['best_val_accuracy']:.4f} "
                f"@ epoch {s['best_epoch']}"
            )
    else:
        print("没有任何成功的划分结果。请检查特征目录与文件命名。")


if __name__ == "__main__":
    main()