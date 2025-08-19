#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

# 从你的模型文件导入（BN 版本）
from mudestreda import CrossModalTransformerFusionModel

# ----------------- 配置（按需修改） -----------------
SEED = 58
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CHECKPOINT_DIR = 'model_params'
CSV_ROOT = r"D:\Mudestreda\labels_original\labels_original\tool_distribution"
FEATURES_ROOT_SPEC = r"D:\py\RPS-MODIFY\Mudestreda\mel_feature"
FEATURES_ROOT_TOOL = r"D:\py\RPS-MODIFY\mudestreda\tool_feature"
SPEC_TYPES = ['specX','specY','specZ']

BATCH_SIZE = 16
NUM_CLASSES = 3
N_BOOTSTRAP = 2000  # 自助重采样次数
CONF_LEVEL = 0.95   # 置信水平

# BACKBONES = ['resnet18','resnet34','resnet50','densenet121','mnasnet1_0',
#              'convnext_tiny','regnet_y_400mf','vit_b_16']

BACKBONES = ['resnet18']

OUT_CSV = 'results_eval_bootstrap.csv'  # 含 backbone, fold, val_id, test_id, N, acc, ci_low, ci_high, n_boot

# ----------------- 确定性（可选） -----------------
def set_deterministic(seed: int = SEED):
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

# ----------------- 数据集与工具 -----------------
def load_all(csv_root):
    a = pd.read_csv(os.path.join(csv_root, 'train.csv'))
    b = pd.read_csv(os.path.join(csv_root, 'val.csv'))
    c = pd.read_csv(os.path.join(csv_root, 'test.csv'))
    df = pd.concat([a, b, c], ignore_index=True)
    df['id'] = df['id'].astype(str)
    df['tool_folder'] = df['id'].apply(lambda x: re.search(r'(T\d+)', x).group(1) if re.search(r'(T\d+)', x) else 'unknown')
    return df

def build_folds():
    tools = [f"T{i}" for i in range(1, 11)]
    folds = []
    for i in range(0, len(tools), 2):
        val  = [tools[i+1]]  # e.g., T2
        test = [tools[i]]    # e.g., T1
        train = [t for t in tools if t not in set(val + test)]
        folds.append({'train': train, 'val': val, 'test': test})
    return folds

class FeatureDataset(Dataset):
    def __init__(self, labels_df, split_tools, backbone,
                 features_root_spec, features_root_tool,
                 spec_types=('specX','specY','specZ')):
        self.df = labels_df[labels_df['tool_folder'].isin(split_tools)].reset_index(drop=True).copy()
        self.backbone = backbone
        self.features_root_spec = features_root_spec
        self.features_root_tool = features_root_tool
        self.spec_types = list(spec_types)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        _id = row['id']; label = int(row['tool_label']) - 1
        tool_folder = row['tool_folder']

        mels = []
        for spec in self.spec_types:
            p = os.path.join(self.features_root_spec, tool_folder, self.backbone, spec, _id + '.npy')
            arr = np.load(p)
            mels.append(torch.from_numpy(arr).float())
        mel_feats = torch.stack(mels, dim=0)

        p_tool = os.path.join(self.features_root_tool, tool_folder, self.backbone, 'tool', _id + '.npy')
        arr_tool = np.load(p_tool)
        tool_feat = torch.from_numpy(arr_tool).float()
        if tool_feat.ndim == 2:
            tool_feat = torch.amax(tool_feat, dim=0, keepdim=True)
        else:
            tool_feat = tool_feat.unsqueeze(0)

        return mel_feats, tool_feat, torch.tensor(label, dtype=torch.long)

@torch.inference_mode()
def collect_predictions(model, loader, device):
    model.eval()
    ys, ps = [], []
    for mel, tool, lbl in tqdm(loader, desc="Infer (test)", leave=False):
        mel, tool = mel.to(device), tool.to(device)
        logits, _, _, _ = model(mel, tool)
        pred = logits.argmax(dim=1).cpu().numpy()
        ys.append(lbl.numpy())
        ps.append(pred)
    y_true = np.concatenate(ys, axis=0) if ys else np.array([], dtype=int)
    y_pred = np.concatenate(ps, axis=0) if ps else np.array([], dtype=int)
    return y_true, y_pred

def bootstrap_acc_ci(y_true, y_pred, n_boot=2000, conf_level=0.95, seed=SEED):
    assert y_true.shape[0] == y_pred.shape[0], "y_true/y_pred length mismatch"
    n = y_true.shape[0]
    if n == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    base_acc = (y_true == y_pred).mean()
    if n == 1 or n_boot <= 0:
        return base_acc, base_acc, base_acc
    idx = np.arange(n)
    boot_acc = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        samp = rng.choice(idx, size=n, replace=True)
        boot_acc[b] = (y_true[samp] == y_pred[samp]).mean()
    alpha = (1.0 - conf_level) / 2.0
    low, high = np.quantile(boot_acc, [alpha, 1.0 - alpha], interpolation='nearest')
    return base_acc, float(low), float(high)

# ----------------- 主逻辑 -----------------
def main():
    set_deterministic(SEED)

    device = DEVICE
    labels_df = load_all(CSV_ROOT)
    folds = build_folds()

    rows = []
    for backbone in BACKBONES:
        print(f"\n===== Evaluating backbone: {backbone} =====")
        backbone_dir = os.path.join(CHECKPOINT_DIR, backbone)
        for fold_idx, sp in enumerate(folds, 1):
            val_id = ",".join(sp['val'])
            test_id = ",".join(sp['test'])

            # 数据集/loader（仅 test）
            ds = FeatureDataset(labels_df, sp['test'], backbone,
                                FEATURES_ROOT_SPEC, FEATURES_ROOT_TOOL, SPEC_TYPES)
            if len(ds) == 0:
                print(f"[WARN] {backbone} fold {fold_idx}: 测试集合为空，跳过")
                continue

            loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=0, pin_memory=torch.cuda.is_available(), persistent_workers=False)

            # 模型与权重
            sample_mel, _, _ = ds[0]
            input_dim = sample_mel.shape[-1]
            model = CrossModalTransformerFusionModel(input_dim=input_dim, reduced_dim=256).to(device)

            ckpt_path = os.path.join(backbone_dir, f"best_model_{backbone}_fold{fold_idx}_2.pth")
            if not os.path.exists(ckpt_path):
                print(f"[WARN] 缺少权重，跳过: {ckpt_path}")
                continue
            model.load_state_dict(torch.load(ckpt_path, map_location=device))

            # 推断与自助法置信区间
            y_true, y_pred = collect_predictions(model, loader, device)
            acc, ci_low, ci_high = bootstrap_acc_ci(y_true, y_pred, n_boot=N_BOOTSTRAP, conf_level=CONF_LEVEL, seed=SEED)

            print(f"[{backbone}][Fold {fold_idx}] N={len(ds)}  acc={acc:.4f}  {int(CONF_LEVEL*100)}% CI: [{ci_low:.4f}, {ci_high:.4f}]")

            rows.append({
                'backbone': backbone,
                'fold': fold_idx,
                'val_id': val_id,
                'test_id': test_id,
                'N': len(ds),
                'acc': acc,
                'ci_low': ci_low,
                'ci_high': ci_high,
                'n_boot': N_BOOTSTRAP,
            })

    if rows:
        pd.DataFrame(rows, columns=['backbone','fold','val_id','test_id','N','acc','ci_low','ci_high','n_boot']).to_csv(
            OUT_CSV, index=False, encoding='utf-8-sig'
        )
        print(f"\n已保存到: {OUT_CSV}")
    else:
        print("\n未生成任何评估结果，请检查权重与路径是否正确。")

if __name__ == '__main__':
    main()