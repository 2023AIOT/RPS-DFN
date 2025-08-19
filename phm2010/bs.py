#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import re
import json
from pathlib import Path
from typing import List, Set, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ============================== 解析与数据集 ==============================
DEFAULT_SET_PATTERNS = [
    r'(?:^|[_-])s(?P<sid>\d+)(?:[_-]|$)',
    r'(?:^|[_-])set(?P<sid>\d+)(?:[_-]|$)',
    r'(?:^|[_-])c(?P<sid>\d+)(?:[_-]|$)',
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
    if cut <= 50:
        return 0
    elif cut <= 175:
        return 1
    else:
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

# ============================== 模型定义 ==============================
class ImprovedFeatureFusionTransformer(nn.Module):
    def __init__(self, input_dim: int, reduced_dim: int = 128, num_layers: int = 2, nhead: int = 4):
        super().__init__()
        self.shared_fc = nn.Linear(input_dim, reduced_dim)
        self.bn = nn.BatchNorm1d(reduced_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=reduced_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.modal_attention = nn.Linear(reduced_dim, 1)
        self.classifier = nn.Linear(reduced_dim, 3)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.shared_fc(features)
        batch_size, modal_count, dim = x.shape
        x = x.view(-1, dim)
        x = self.bn(x)
        x = x.view(batch_size, modal_count, dim)
        transformer_out = self.transformer_encoder(x)
        attn_w = torch.softmax(self.modal_attention(transformer_out).squeeze(-1), dim=1)
        fused = torch.sum(transformer_out * attn_w.unsqueeze(-1), dim=1)
        logits = self.classifier(fused)
        return logits, attn_w, transformer_out

# ============================== 工具函数 ==============================
def build_combo_dir(features_root: Path, resnet_arch: str) -> Path:
    mel_filters, window_ms, overlap_ratio = 128, 20, 0.5
    combo = f"mel{mel_filters}_w{window_ms}_o{int(overlap_ratio * 100)}_{resnet_arch}"
    return features_root / combo

def collect_candidate_dirs(combo_dir: Path) -> List[Path]:
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

@torch.no_grad()
def infer_val_predictions(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits = []
    all_labels = []
    for feats, labels in loader:
        feats = feats.to(device, non_blocking=False)
        logits, _, _ = model(feats)
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels)
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0).numpy()
    preds = logits.argmax(dim=1).numpy()
    return labels, preds

def accuracy_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))

def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 1000,
                 alpha: float = 0.05, random_state: int = 42) -> Tuple[float, float, float]:
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    idx = np.arange(n)
    values = []
    for _ in range(n_boot):
        s = rng.choice(idx, size=n, replace=True)
        values.append(accuracy_score_np(y_true[s], y_pred[s]))
    values = np.asarray(values, dtype=np.float64)
    lower = np.percentile(values, 100.0 * alpha / 2.0)
    upper = np.percentile(values, 100.0 * (1.0 - alpha / 2.0))
    point = accuracy_score_np(y_true, y_pred)
    return point, float(lower), float(upper)

# ============================== 主流程 ==============================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="用自助法重采样计算验证集准确率的95%置信区间")
    parser.add_argument("--features-root", type=str, default="D:\\py\\RPS-MODIFY\\PHM2010\\features",
                        help="预计算特征根目录")
    parser.add_argument("--results-dir", type=str, default="verify_results",
                        help="训练脚本保存 best_model_*.pth 的目录，同时输出本脚本结果")
    parser.add_argument("--resnet-arch", type=str, default="resnet18",
                        help="与特征目录命名一致的 ResNet 架构名")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="cuda 或 cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-boot", type=int, default=1000, help="自助法重复次数，建议≥1000")
    parser.add_argument("--alpha", type=float, default=0.05, help="置信区间显著性水平")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--set-regex", type=str, default=None,
                        help="可选，自定义解析 set_id 的正则，命名捕获组为 sid")
    args = parser.parse_args()

    features_root = Path(args.features_root)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    combo_dir = build_combo_dir(features_root, args.resnet_arch)
    if not combo_dir.exists():
        raise FileNotFoundError(f"组合目录不存在: {combo_dir}")
    cand_dirs = collect_candidate_dirs(combo_dir)

    patterns = [args.set_regex] if args.set_regex else DEFAULT_SET_PATTERNS

    splits = [
        ("train14_val6", [1, 4], [6]),
        ("train16_val4", [1, 6], [4]),
        ("train46_val1", [4, 6], [1]),
    ]

    rows = []
    for split_name, train_ids, val_ids in splits:
        ckpt_path = results_dir.joinpath(f"best_model_{split_name}.pth")
        if not ckpt_path.exists():
            print(f"[跳过] 未找到权重: {ckpt_path}")
            continue

        val_ds = FilteredNpyFeatureDataset(cand_dirs, set(val_ids), patterns)
        feat_dim = val_ds[0][0].shape[1]

        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False
        )

        state = torch.load(ckpt_path, map_location=device)
        model = ImprovedFeatureFusionTransformer(input_dim=int(state.get('feat_dim', feat_dim))).to(device)
        model.load_state_dict(state['model_state_dict'])

        y_true, y_pred = infer_val_predictions(model, val_loader, device)
        acc_point, acc_lo, acc_hi = bootstrap_ci(
            y_true, y_pred, n_boot=int(args.n_boot), alpha=float(args.alpha), random_state=int(args.seed)
        )

        rows.append({
            "split": split_name,
            "Train id": ",".join(map(str, train_ids)),
            "Val id": ",".join(map(str, val_ids)),
            "Val acc point": acc_point,
            "Val acc 95% CI low": acc_lo,
            "Val acc 95% CI high": acc_hi,
            "Bootstrap n": int(args.n_boot),
            "features_dir_used": ";".join(str(d) for d in cand_dirs),
            "checkpoint_path": str(ckpt_path),
        })

        print(f"{split_name} | Val acc={acc_point:.4f} | 95% CI [{acc_lo:.4f}, {acc_hi:.4f}] | N={args.n_boot}")

    if rows:
        df = pd.DataFrame(rows)
        out_csv = results_dir.joinpath("bootstrap_acc_summary.csv")
        df.to_csv(out_csv, index=False)
        with open(results_dir.joinpath("bootstrap_acc_summary.json"), "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"已保存: {out_csv}")
    else:
        print("未生成任何结果。请检查权重路径与特征目录。")

if __name__ == "__main__":
    main()