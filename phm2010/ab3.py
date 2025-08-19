#!/usr/bin/env python3
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import itertools
import time
import json
from datetime import datetime
import warnings
import random
from pathlib import Path

warnings.filterwarnings('ignore')

# ================================== 随机种子管理 ==================================
def set_random_seed(seed=16):
    """
    设置所有相关库的随机种子，确保实验可复现
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"随机种子设置为: {seed}")


def worker_init_fn(worker_id):
    """DataLoader 工作进程初始化函数"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ================================== 消融实验配置 ==================================
ABLATION_CONFIG = {
    'resnet_arch': ['resnet18', 'resnet34', 'resnet50'],
    'overlap_ratio': [0.75, 0.5, 0.25],
    'feature_configs': [
        {'window_ms': 50, 'mel_filters': [128, 256, 512, 1024]},
        {'window_ms': 20, 'mel_filters': [128, 256]},
        {'window_ms': 10, 'mel_filters': [128]},
    ]
}

EXPERIMENT_CONFIG = {
    'random_seed':               23,
    'epochs_per_experiment':     30,
    'early_stopping_patience':   5,
    'batch_size':                16,
    'device':                    torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'base_path':                 "D:\\PHM2010\\",
    'features_dir':              "D:\\py\\RPS-MODIFY\\mel_feature",  # 预计算特征根目录
    'results_dir':               'ablation_results',
    'train_set_ids':             [1, 4],
    'val_set_ids':               [6],
    'data_windowing': {
        'enable':     True,
        'offset':     5000,
        'window_size':100000
    }
}


# ================================== NpyFeatureDataset ==================================
class NpyFeatureDataset(Dataset):
    """
    加载预计算的 .npy 特征（shape: modal × feature_dim）
    """
    def __init__(self, features_root, split):
        features_dir = Path(features_root) / split
        self.files = sorted(features_dir.glob("*.npy"))
        if not self.files:
            raise FileNotFoundError(f"未在 {features_dir} 找到任何 .npy 文件")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        data = np.load(file)                          # (modal, feature_dim)
        feat = torch.tensor(data, dtype=torch.float32)
        # 从文件名解析 cut 值 (e.g. c1_023.npy)
        parts = file.stem.split("_")
        if len(parts) < 2:
            raise ValueError(f"无法从文件名解析 cut 值: {file.name}")
        cut = int(parts[-1])
        if   cut <= 50:   label = 0
        elif cut <= 175:  label = 1
        else:             label = 2
        return feat, torch.tensor(label, dtype=torch.long)


# ================================== 改进的 Feature Fusion Transformer ==================================
class ImprovedFeatureFusionTransformer(nn.Module):
    def __init__(self, input_dim, reduced_dim, num_layers=2, nhead=4):
        super().__init__()
        self.shared_fc = nn.Linear(input_dim, reduced_dim)
        self.bn = nn.BatchNorm1d(reduced_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=reduced_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.modal_attention = nn.Linear(reduced_dim, 1)
        self.classifier = nn.Linear(reduced_dim, 3)

    def forward(self, features):
        # mel_feature: (batch, modal, feature_dim)
        x = self.shared_fc(features)  # (batch, modal, reduced_dim)
        batch, modal, dim = x.shape
        x = x.view(-1, dim)
        x = self.bn(x)
        x = x.view(batch, modal, dim)
        transformer_out = self.transformer_encoder(x)  # (batch, modal, reduced_dim)
        attn_w = torch.softmax(self.modal_attention(transformer_out).squeeze(-1), dim=1)
        fused = torch.sum(transformer_out * attn_w.unsqueeze(-1), dim=1)
        logits = self.classifier(fused)
        return logits, attn_w, transformer_out


# ================================== 多任务损失 ==================================
class MultiTaskLoss(nn.Module):
    def __init__(self, lambda1=0.1, lambda2=0.05):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.ce = nn.CrossEntropyLoss()

    def forward(self, preds, targets, attn_w, modal_f):
        L_ce = self.ce(preds, targets)
        L_mod = self.modal_balance_loss(modal_f, attn_w)
        L_att = self.attention_regularization_loss(attn_w)
        return L_ce + self.lambda1 * L_mod + self.lambda2 * L_att, L_ce, L_mod, L_att

    def modal_balance_loss(self, modal_f, attn_w):
        norms = torch.norm(modal_f, dim=2)  # (batch, modal)
        norms = F.normalize(norms, p=1, dim=1)
        return F.mse_loss(attn_w, norms)

    def attention_regularization_loss(self, attn_w):
        ent = -torch.sum(attn_w * torch.log(attn_w + 1e-8), dim=1)
        max_ent = torch.log(torch.tensor(attn_w.size(1), dtype=torch.float32, device=attn_w.device))
        return torch.mean(max_ent - ent)


# ================================== 实验日志记录 ==================================
class ExperimentLogger:
    def __init__(self, results_dir):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.detailed_results = []
        self.summary_results = []

    def log_epoch(self, exp_id, params, epoch, metrics, epoch_time):
        rec = {
            'experiment_id': exp_id,
            **params,
            'epoch': epoch,
            **metrics,
            'epoch_time': epoch_time
        }
        self.detailed_results.append(rec)

    def log_summary(self, exp_id, params, best_acc, best_epoch, total_time, status, feat_dim):
        rec = {
            'experiment_id': exp_id,
            **params,
            'best_val_accuracy': best_acc,
            'best_epoch': best_epoch,
            'total_time': total_time,
            'convergence_status': status,
            'feature_dim': feat_dim,
            'data_windowing_enabled': EXPERIMENT_CONFIG['data_windowing']['enable'],
            'window_size': EXPERIMENT_CONFIG['data_windowing']['window_size'],
            'offset': EXPERIMENT_CONFIG['data_windowing']['offset']
        }
        self.summary_results.append(rec)

    def save(self):
        pd.DataFrame(self.detailed_results).to_csv(os.path.join(self.results_dir, 'detailed_results.csv'), index=False)
        pd.DataFrame(self.summary_results).to_csv(os.path.join(self.results_dir, 'summary_results.csv'), index=False)
        with open(os.path.join(self.results_dir, 'detailed_results.json'), 'w') as f:
            json.dump(self.detailed_results, f, indent=2)
        with open(os.path.join(self.results_dir, 'summary_results.json'), 'w') as f:
            json.dump(self.summary_results, f, indent=2)
        print(f"✓ 已保存结果至 {self.results_dir}")


# ================================== 实验管理器 ==================================
class AblationExperimentManager:
    def __init__(self, config):
        self.config = config
        self.logger = ExperimentLogger(config['results_dir'])
        self.device = config['device']
        set_random_seed(config['random_seed'])

        # 生成所有参数组合
        self.param_combinations = []
        for fc in ABLATION_CONFIG['feature_configs']:
            for mel_n in fc['mel_filters']:
                for overlap in ABLATION_CONFIG['overlap_ratio']:
                    for arch in ABLATION_CONFIG['resnet_arch']:
                        self.param_combinations.append({
                            'mel_filters': mel_n,
                            'window_ms': fc['window_ms'],
                            'overlap_ratio': overlap,
                            'resnet_arch': arch
                        })

        total = len(self.param_combinations)
        print(f"总实验组合数: {total}, 每个实验 {config['epochs_per_experiment']} 轮, 设备: {self.device}")
        if config['data_windowing']['enable']:
            print(f"数据窗口裁剪: 启用 (窗口={config['data_windowing']['window_size']}, 偏移={config['data_windowing']['offset']})")
        print("="*80)

    def run_single_experiment(self, exp_id, params):
        print(f"\n实验 {exp_id}/{len(self.param_combinations)} - 参数: {params}")
        set_random_seed(self.config['random_seed'])
        start = time.time()
        try:
            combo = f"mel{params['mel_filters']}_w{params['window_ms']}_o{int(params['overlap_ratio']*100)}_{params['resnet_arch']}"
            base = Path(self.config['features_dir']) / combo
            train_ds = NpyFeatureDataset(base, 'train')
            val_ds   = NpyFeatureDataset(base, 'val')
            feat_dim = train_ds[0][0].shape[1]

            g = torch.Generator(); g.manual_seed(self.config['random_seed'])
            train_loader = DataLoader(train_ds, batch_size=self.config['batch_size'],
                                      shuffle=True, num_workers=2, pin_memory=True,
                                      worker_init_fn=worker_init_fn, generator=g)
            val_loader = DataLoader(val_ds, batch_size=self.config['batch_size'],
                                    shuffle=False, num_workers=2, pin_memory=True,
                                    worker_init_fn=worker_init_fn)

            model = ImprovedFeatureFusionTransformer(input_dim=feat_dim, reduced_dim=128).to(self.device)
            optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6)
            criterion = MultiTaskLoss(lambda1=0.1, lambda2=0.05)

            best_acc, best_epoch = 0.0, 0
            early_stop = 0

            for epoch in range(self.config['epochs_per_experiment']):
                t0 = time.time()
                train_metrics = self.train_epoch(model, train_loader, optimizer, criterion)
                val_metrics   = self.validate_epoch(model, val_loader, criterion)
                scheduler.step(val_metrics['val_loss'])

                duration = time.time() - t0
                self.logger.log_epoch(exp_id, params, epoch+1, {**train_metrics, **val_metrics}, duration)

                if val_metrics['val_accuracy'] > best_acc:
                    best_acc, best_epoch, early_stop = val_metrics['val_accuracy'], epoch+1, 0
                    torch.save({'model_state_dict': model.state_dict()},
                               os.path.join(self.config['results_dir'], f'best_model_{exp_id}.pth'))
                else:
                    early_stop += 1

                print(f"  Epoch {epoch+1}: Train Acc={train_metrics['train_accuracy']:.4f}, Val Acc={val_metrics['val_accuracy']:.4f}, Time={duration:.1f}s")
                if early_stop >= self.config['early_stopping_patience']:
                    print(f"  早停于 epoch {epoch+1}")
                    break

            total_time = time.time() - start
            status = 'early_stopped' if early_stop >= self.config['early_stopping_patience'] else 'completed'
            self.logger.log_summary(exp_id, params, best_acc, best_epoch, total_time, status, feat_dim)

            torch.cuda.empty_cache()
            return True

        except Exception as e:
            print(f"实验 {exp_id} 失败: {e}")
            return False

    def train_epoch(self, model, loader, optimizer, criterion):
        model.train()
        running = {'loss':0,'ce':0,'mod':0,'att':0,'correct':0,'total':0}
        for feats, labels in loader:
            feats, labels = feats.to(self.device), labels.to(self.device)
            preds, att_w, modal_f = model(feats)
            loss, Lce, Lmod, Latt = criterion(preds, labels, att_w, modal_f)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = labels.size(0)
            running['loss'] += loss.item()*bs
            running['ce']   += Lce.item()*bs
            running['mod']  += Lmod.item()*bs
            running['att']  += Latt.item()*bs
            _, pred = preds.max(1)
            running['correct'] += (pred==labels).sum().item()
            running['total']   += bs

        n = running['total']
        return {
            'train_loss': running['loss']/n,
            'train_ce_loss': running['ce']/n,
            'train_modal_loss': running['mod']/n,
            'train_attention_loss': running['att']/n,
            'train_accuracy': running['correct']/n
        }

    def validate_epoch(self, model, loader, criterion):
        model.eval()
        loss_sum, correct, total = 0,0,0
        with torch.no_grad():
            for feats, labels in loader:
                feats, labels = feats.to(self.device), labels.to(self.device)
                preds, att_w, modal_f = model(feats)
                l, _, _, _ = criterion(preds, labels, att_w, modal_f)
                loss_sum += l.item()*labels.size(0)
                _, pred = preds.max(1)
                correct += (pred==labels).sum().item()
                total += labels.size(0)
        return {
            'val_loss': loss_sum/total,
            'val_accuracy': correct/total
        }

    def run_all_experiments(self):
        print("开始消融实验...")
        start = time.time()
        succ = 0
        for i, params in enumerate(self.param_combinations, 1):
            if self.run_single_experiment(i, params):
                succ += 1
            if i % 10 == 0:
                self.logger.save()
                print(f"进度: {i}/{len(self.param_combinations)} (成功: {succ})")
        self.logger.save()
        total = time.time() - start
        print(f"所有实验完成! 总耗时: {total:.1f}s, 成功 {succ}/{len(self.param_combinations)}")
        self.analyze_results()

    def analyze_results(self):
        if not self.logger.summary_results:
            return
        df = pd.DataFrame(self.logger.summary_results)
        best = df.loc[df['best_val_accuracy'].idxmax()]
        print("\n最佳组合:", best.to_dict())


# ================================== 主程序入口 ==================================
if __name__ == "__main__":
    print("="*80)
    print("🔥 消融实验 - 网格搜索 (基于预计算 .npy 特征)")
    print("="*80)
    manager = AblationExperimentManager(EXPERIMENT_CONFIG)
    manager.run_all_experiments()
    print("\n实验完成! 结果已保存至", EXPERIMENT_CONFIG['results_dir'])
