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

# 固定最优组合与路径
COMBO_NAME = "mel128_w20_o50_resnet18"
FEATURES_ROOT = Path("D:/py/RPS-MODIFY/PHM2010/NF")

# 固定划分
TRAIN_SET_IDS = [1, 4]
VAL_SET_IDS = [6]

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
        x = self.shared_fc(features)  # (batch, modal, reduced_dim)
        b, m, d = x.shape
        x = x.view(-1, d)
        x = self.bn(x)
        x = x.view(b, m, d)
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

# ============================== 解析与数据 ==============================
DEFAULT_SET_PATTERNS = [
    r'(?:^|[_-])c(?P<sid>\d+)(?:[_-]|$)',
    r'(?:^|[_-])s(?P<sid>\d+)(?:[_-]|$)',
    r'(?:^|[_-])set(?P<sid>\d+)(?:[_-]|$)',
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
    if cut <= 50: return 0
    if cut <= 175: return 1
    return 2

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

# ============================== 训练/验证 ==============================
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

# ============================== 发现噪声组合 ==============================
def discover_noise_conditions(combo_dir: Path) -> List[Tuple[int, int, Path]]:
    # 期望子目录名形如: snr{SNR}_imp{PCT}
    ret = []
    if not combo_dir.exists():
        return ret
    for sub in combo_dir.iterdir():
        if not sub.is_dir():
            continue
        m = re.match(r'^snr(?P<snr>\d+)_imp(?P<imp>\d+)$', sub.name, flags=re.IGNORECASE)
        if not m:
            continue
        snr = int(m.group('snr'))
        imp = int(m.group('imp'))  # 百分比，例 5 表示 5%
        # 必须包含 train 与 val
        if not (sub.joinpath("train").exists() and sub.joinpath("val").exists()):
            continue
        ret.append((snr, imp, sub))
    ret.sort(key=lambda x: (x[0], x[1]))
    return ret

# ============================== 单组噪声训练 ==============================
def run_one_noise(
    device: torch.device,
    noise_dir: Path,
    snr: int,
    imp: int,
    batch_size: int,
    epochs: int,
    patience: int,
    seed: int,
    results_dir: Path
) -> Dict[str, float]:
    print(f"\n=== 噪声组合: SNR={snr} dB, IMP={imp}% ===")
    set_random_seed(seed)

    # 数据集
    train_dir = noise_dir / "train"
    val_dir   = noise_dir / "val"
    patterns = DEFAULT_SET_PATTERNS

    train_ds = FilteredNpyFeatureDataset([train_dir], set(TRAIN_SET_IDS), patterns)
    val_ds   = FilteredNpyFeatureDataset([val_dir],   set(VAL_SET_IDS),   patterns)

    feat_dim = train_ds[0][0].shape[1]
    print(f"样本数: train={len(train_ds)}, val={len(val_ds)}, feature_dim={feat_dim}")

    g = torch.Generator(); g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2,
                              pin_memory=torch.cuda.is_available(), worker_init_fn=worker_init_fn, generator=g)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2,
                              pin_memory=torch.cuda.is_available(), worker_init_fn=worker_init_fn)

    model = ImprovedFeatureFusionTransformer(input_dim=feat_dim, reduced_dim=128).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6)
    criterion = MultiTaskLoss(lambda1=0.1, lambda2=0.05)

    best_acc, best_epoch = 0.0, 0
    best_train_loss, best_val_loss = None, None
    best_train_acc = None
    early_stop = 0
    history = []

    tag = f"snr{snr}_imp{imp}"
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_metrics['val_loss'])
        duration = time.time() - t0

        row = {
            'noise': tag,
            'epoch': epoch,
            'train_ids': ",".join(map(str, TRAIN_SET_IDS)),
            'val_ids': ",".join(map(str, VAL_SET_IDS)),
            **train_metrics,
            **val_metrics,
            'epoch_time': duration
        }
        history.append(row)

        print(f"  Epoch {epoch:02d} | "
              f"Train Loss={train_metrics['train_loss']:.4f} | Val Loss={val_metrics['val_loss']:.4f} | "
              f"Train Acc={train_metrics['train_accuracy']:.4f} | Val Acc={val_metrics['val_accuracy']:.4f} | "
              f"Time={duration:.1f}s")

        if val_metrics['val_accuracy'] > best_acc:
            best_acc, best_epoch, early_stop = val_metrics['val_accuracy'], epoch, 0
            best_train_loss = train_metrics['train_loss']
            best_val_loss = val_metrics['val_loss']
            best_train_acc = train_metrics['train_accuracy']
            torch.save({'model_state_dict': model.state_dict(), 'feat_dim': feat_dim},
                       results_dir.joinpath(f'best_model_{tag}.pth'))
        else:
            early_stop += 1

        if early_stop >= patience:
            print(f"  早停于 epoch {epoch}")
            break

    total_time = time.time() - t_start

    # 保存明细
    df_hist = pd.DataFrame(history)
    df_hist.to_csv(results_dir.joinpath(f'detailed_{tag}.csv'), index=False)
    with open(results_dir.joinpath(f'detailed_{tag}.json'), 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    torch.cuda.empty_cache()

    return {
        'noise': tag,
        'snr_db': snr,
        'imp_percent': imp,
        'train_ids': ",".join(map(str, TRAIN_SET_IDS)),
        'val_ids': ",".join(map(str, VAL_SET_IDS)),
        'best_val_accuracy': best_acc,
        'best_epoch': best_epoch,
        'best_train_loss': best_train_loss,
        'best_val_loss': best_val_loss,
        'best_train_accuracy': best_train_acc,
        'total_time_sec': total_time,
        'train_size': len(train_ds),
        'val_size': len(val_ds),
        'feature_dim': feat_dim,
        'noise_dir': str(noise_dir),
    }

# ============================== 主流程 ==============================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="训练带噪声的特征（train=[1,4], val=[6]）")
    parser.add_argument("--features-root", type=str, default=str(FEATURES_ROOT),
                        help="噪声特征根目录（其下应有 mel128_w20_o50_resnet18/snr*_imp*）")
    parser.add_argument("--combo-name", type=str, default=COMBO_NAME,
                        help="固定组合目录名，默认为 mel128_w20_o50_resnet18")
    parser.add_argument("--results-dir", type=str, default="NF_train_results",
                        help="训练结果输出目录")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=25)
    # 可选：仅训练指定噪声组合
    parser.add_argument("--snr", type=int, default=None, help="仅训练指定 SNR（如 10）")
    parser.add_argument("--imp", type=int, default=None, help="仅训练指定 IMP 百分比（如 5 表示 5%）")
    args = parser.parse_args()

    device = torch.device(args.device)
    features_root = Path(args.features_root)
    combo_dir = features_root / args.combo_name
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 发现噪声组合
    all_noise = discover_noise_conditions(combo_dir)
    if not all_noise:
        print(f"未在 {combo_dir} 下发现噪声子目录（形如 snr10_imp5）")
        return

    # 过滤（可选）
    if args.snr is not None or args.imp is not None:
        all_noise = [t for t in all_noise if
                     (args.snr is None or t[0] == args.snr) and
                     (args.imp is None or t[1] == args.imp)]
        if not all_noise:
            print("没有匹配指定 --snr/--imp 的噪声组合")
            return

    summaries = []
    for snr, imp, ndir in all_noise:
        try:
            s = run_one_noise(
                device=device,
                noise_dir=ndir,
                snr=snr,
                imp=imp,
                batch_size=args.batch_size,
                epochs=args.epochs,
                patience=args.patience,
                seed=args.seed,
                results_dir=results_dir
            )
            summaries.append(s)
        except Exception as e:
            print(f"[错误] 训练 SNR={snr}, IMP={imp}% 失败: {e}")

    if summaries:
        df = pd.DataFrame(summaries).sort_values(by="best_val_accuracy", ascending=False)
        df.to_csv(results_dir.joinpath("summary_noisy.csv"), index=False)

        # 精简表，含你常用指标
        out_df = pd.DataFrame({
            'SNR(dB)': df['snr_db'],
            'IMP(%)': df['imp_percent'],
            'Train loss': df['best_train_loss'],
            'Val loss': df['best_val_loss'],
            'Train acc': df['best_train_accuracy'],
            'Val acc': df['best_val_accuracy'],
        })
        out_df.to_csv(results_dir.joinpath("summary_metrics_noisy.csv"), index=False)

        print("\n=== 噪声组合最佳结果（按 Val Acc） ===")
        for _, r in df.iterrows():
            print(f"SNR={int(r['snr_db'])}dB, IMP={int(r['imp_percent'])}% | "
                  f"Train Loss={r['best_train_loss']:.4f} | Val Loss={r['best_val_loss']:.4f} | "
                  f"Train Acc={r['best_train_accuracy']:.4f} | Val Acc={r['best_val_accuracy']:.4f} "
                  f"@ epoch {int(r['best_epoch'])}")
    else:
        print("没有任何成功的训练结果。请检查噪声特征目录结构。")

if __name__ == "__main__":
    main()