#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 环境变量需在导入 torch 前设置
import os
os.environ.setdefault("PYTHONHASHSEED", "11")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import re
import gc
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 你的模型与损失
from mudestreda import CrossModalTransformerFusionModel, MultiModalLoss

# ========= 随机/确定性 =========
seed = 58

def reset_seeds(seed_val: int):
    os.environ['PYTHONHASHSEED'] = str(seed_val)
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

reset_seeds(seed)

# ========= 路径配置 =========
# 谱图特征根目录（train/eval）
features_root_spec_train = r"D:\py\RPS-MODIFY\Mudestreda\mel_feature"
features_root_spec_eval  = r"D:\py\RPS-MODIFY\Mudestreda\mel_feature"

# 刀具特征根目录（train/eval，位于 tool_feature）
features_root_tool_train = r"D:\py\RPS-MODIFY\mudestreda\tool_feature"
features_root_tool_eval  = r"D:\py\RPS-MODIFY\mudestreda\tool_feature"


# CSV 根目录
trainval_csv_root  = r"D:\Mudestreda\labels_original\labels_original\tool_distribution"
eval_csv_root      = r"D:\Mudestreda\labels_original\labels_original\tool_distribution"


# ========= 训练超参 =========
device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size   = 16
num_epochs   = 50
lr           = 1e-4
weight_decay = 1e-6
patience     = 10


checkpoint_dir = 'model_params5'
os.makedirs(checkpoint_dir, exist_ok=True)

backbones = ['resnet18','resnet34','resnet50','densenet121','mnasnet1_0',
             'convnext_tiny','regnet_y_400mf','vit_b_16']
spec_types = ['specX','specY','specZ']

# ========= 工具函数 =========
def load_all(csv_root):
    a = pd.read_csv(os.path.join(csv_root, 'train.csv'))
    b = pd.read_csv(os.path.join(csv_root, 'val.csv'))
    c = pd.read_csv(os.path.join(csv_root, 'test.csv'))
    df = pd.concat([a, b, c], ignore_index=True)
    df['id'] = df['id'].astype(str)
    df['tool_folder'] = df['id'].apply(lambda x: re.search(r'(T\d+)', x).group(1) if re.search(r'(T\d+)', x) else 'unknown')
    return df

labels_df_tv   = load_all(trainval_csv_root)  # 训练
def drop_aug(df):
    if 'aug' in df.columns:
        return df[df['aug'] == 0].copy()
    return df[~df['id'].str.contains('aug', case=False, na=False)].copy()

labels_df_eval = drop_aug(load_all(eval_csv_root))

# labels_df_eval = load_all(eval_csv_root)      # 验证/测试

# ========= 工具划分（5折：val=1、test=1，成对轮换）=========
tools = [f"T{i}" for i in range(1, 11)]
folds = []
for i in range(0, len(tools), 2):
    val  = [tools[i+1]]   # e.g., T2
    test = [tools[i]]     # e.g., T1
    train = [t for t in tools if t not in set(val + test)]
    folds.append({'train': train, 'val': val, 'test': test})

print("5-fold splits (val 1, test 1):")
for i, sp in enumerate(folds, 1):
    print(i, "train:", sp['train'], "val:", sp['val'], "test:", sp['test'])

# ========= 数据集 =========
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

        # 谱图特征 (3, D)
        mels = []
        for spec in self.spec_types:
            p = os.path.join(self.features_root_spec, tool_folder, self.backbone, spec, _id + '.npy')
            arr = np.load(p)
            mels.append(torch.from_numpy(arr).float())
        mel_feats = torch.stack(mels, dim=0)

        # 刀具特征 (K,D)/(D,) -> (1,D)
        p_tool = os.path.join(self.features_root_tool, tool_folder, self.backbone, 'tool', _id + '.npy')
        arr_tool = np.load(p_tool)
        tool_feat = torch.from_numpy(arr_tool).float()
        if tool_feat.ndim == 2:
            tool_feat = torch.amax(tool_feat, dim=0, keepdim=True)
        else:
            tool_feat = tool_feat.unsqueeze(0)

        return mel_feats, tool_feat, torch.tensor(label, dtype=torch.long)

# ========= 训练 / 验证 =========
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for mel, tool, lbl in tqdm(loader, desc="Training", leave=False):
        mel, tool, lbl = mel.to(device), tool.to(device), lbl.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, att_w, mel_e, tool_r = model(mel, tool)
        loss, *_ = criterion(logits, lbl, mel_e, tool_r, att_w)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * lbl.size(0)
        correct += (logits.argmax(1) == lbl).sum().item()
        total += lbl.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def eval_one_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for mel, tool, lbl in tqdm(loader, desc="Validation", leave=False):
        mel, tool, lbl = mel.to(device), tool.to(device), lbl.to(device)
        logits, att_w, mel_e, tool_r = model(mel, tool)
        loss, *_ = criterion(logits, lbl, mel_e, tool_r, att_w)
        total_loss += loss.item() * lbl.size(0)
        correct += (logits.argmax(1) == lbl).sum().item()
        total += lbl.size(0)
    return total_loss / total, correct / total

# ========= 主流程 =========
results = []

for backbone in backbones:
    print(f"\n===== Starting backbone: {backbone} =====")
    backbone_dir = os.path.join(checkpoint_dir, backbone)
    os.makedirs(backbone_dir, exist_ok=True)

    backbone_results = []

    for fold_idx, sp in enumerate(folds, 1):
        print(f"\n------ Backbone: {backbone} | Fold: {fold_idx} ------")
        print(f" train: {sp['train']}\n val: {sp['val']}\n test: {sp['test']}")

        # 确保每个 backbone/fold 与“单独跑”结果一致
        run_seed = seed
        reset_seeds(run_seed)

        # 数据集
        train_ds = FeatureDataset(labels_df_tv,   sp['train'], backbone,
                                  features_root_spec_train, features_root_tool_train, spec_types)
        val_ds   = FeatureDataset(labels_df_eval, sp['val'],   backbone,
                                  features_root_spec_eval,  features_root_tool_eval,  spec_types)
        test_ds  = FeatureDataset(labels_df_eval, sp['test'],  backbone,
                                  features_root_spec_eval,  features_root_tool_eval,  spec_types)

        # DataLoader 的确定性洗牌
        g = torch.Generator()
        g.manual_seed(run_seed)

        pin = torch.cuda.is_available()
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  generator=g, num_workers=0, pin_memory=pin, persistent_workers=False)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                                  num_workers=0, pin_memory=pin, persistent_workers=False)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                                  num_workers=0, pin_memory=pin, persistent_workers=False)

        # 输入维度
        sample_mel, sample_tool_feat, _ = train_ds[0]
        input_dim = sample_mel.shape[-1]

        # 模型初始化前再次重置
        reset_seeds(run_seed)

        model     = CrossModalTransformerFusionModel(input_dim=input_dim, reduced_dim=256).to(device)
        criterion = MultiModalLoss(lambda1=0.1, lambda2=0.05)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6)

        best_val_acc, best_train_acc, early_stop = 0.0, 0.0, 0

        for epoch in range(1, num_epochs + 1):
            print(f"\n[{backbone}][Fold {fold_idx}] Epoch {epoch}/{num_epochs}")
            tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
            print(f" Train -> loss: {tr_loss:.4f}, acc: {tr_acc:.4f}")

            val_loss, val_acc = eval_one_epoch(model, val_loader, criterion)
            print(f" Val   -> loss: {val_loss:.4f}, acc: {val_acc:.4f}")
            scheduler.step(val_loss)

            if val_acc > best_val_acc:
                best_val_acc, best_train_acc, early_stop = val_acc, tr_acc, 0
                save_path = os.path.join(backbone_dir, f"best_model_{backbone}_fold{fold_idx}_5.pth")
                torch.save(model.state_dict(), save_path)
                print(f"  New best val_acc, saved {save_path}")
            else:
                early_stop += 1
                if early_stop >= patience:
                    print("  早停触发")
                    break

        # 测试
        best_model_path = os.path.join(backbone_dir, f"best_model_{backbone}_fold{fold_idx}_5.pth")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        test_loss, test_acc = eval_one_epoch(model, test_loader, criterion)
        print(f" Test  -> loss: {test_loss:.4f}, acc: {test_acc:.4f}")

        record = dict(
            backbone=backbone, fold=fold_idx,
            val_id=",".join(sp['val']), test_id=",".join(sp['test']),
            train_acc=best_train_acc, val_acc=best_val_acc, test_acc=test_acc
        )
        backbone_results.append(record); results.append(record)

        # 可选：释放缓存，避免长跑内存累积（不影响确定性）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # 结果保存
    pd.DataFrame(backbone_results).to_csv(os.path.join(backbone_dir, f"results_{backbone}5.csv"), index=False)
    torch.save(model.state_dict(), os.path.join(backbone_dir, f"best_model_{backbone}_final_5.pth"))
    print(f"Saved results for {backbone}")

pd.DataFrame(results).to_csv('results_all_backbones5.csv', index=False)
print("Saved overall results to results_all_backbones5.csv")