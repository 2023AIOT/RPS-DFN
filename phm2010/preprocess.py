#!/usr/bin/env python3
# extract_features.py

import os
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
import librosa
from pathlib import Path

# 解决 OpenMP 库冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 消融配置：每组 window_ms 只保留合法的 mel_filters
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
    'base_path': Path("D:/PHM2010"),
    'train_set_ids': [1, 4],
    'val_set_ids': [6],
    'data_windowing': {
        'enable': True,
        'offset': 5000,
        'window_size': 100000
    },
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'output_dir': Path("/RPS-MODIFY/PHM2010/tool_feature")
}

def get_resnet_extractor(arch_name: str, device: torch.device):
    """加载截断后的 ResNet 并返回提取器与输出特征维度。"""
    weights_map = {
        'resnet18': (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, 512),
        'resnet34': (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1, 512),
        'resnet50': (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1, 2048),
    }
    try:
        constructor, weights, feat_dim = weights_map[arch_name]
    except KeyError:
        raise ValueError(f"Unsupported ResNet architecture: {arch_name}")
    model = constructor(weights=weights)
    extractor = nn.Sequential(*list(model.children())[:-1]).to(device).eval()
    return extractor, feat_dim

def extract_window(full_data: np.ndarray, offset: int, window_size: int) -> np.ndarray:
    """对信号做固定长度的窗口裁剪与前向补零。"""
    total_len = full_data.shape[0]
    end = total_len - offset
    start = max(0, end - window_size)
    window = full_data[start:end]
    if window.shape[0] < window_size:
        pad_len = window_size - window.shape[0]
        pad = np.zeros((pad_len, full_data.shape[1]))
        window = np.vstack((pad, window))
    return window

def generate_mel(data: np.ndarray, sr: int, n_fft: int, hop_length: int, n_mels: int) -> torch.Tensor:
    """生成归一化的 3 通道 Mel 频谱图。"""
    S = librosa.feature.melspectrogram(
        y=data, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
    return torch.tensor(S_norm, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)

def process_combo(window_ms: int, mel_n: int, overlap: float, arch: str):
    """对单个 (window_ms, mel_n, overlap, arch) 组合提取特征。"""
    sr = 50000
    n_fft = int(window_ms * sr / 1000)
    hop_length = int(n_fft * (1 - overlap))
    combo_name = f"mel{mel_n}_w{window_ms}_o{int(overlap*100)}_{arch}"
    combo_dir = EXPERIMENT_CONFIG['output_dir'] / combo_name
    combo_dir.mkdir(parents=True, exist_ok=True)

    extractor, feat_dim = get_resnet_extractor(arch, EXPERIMENT_CONFIG['device'])

    for split, ids in [('train', EXPERIMENT_CONFIG['train_set_ids']),
                       ('val',   EXPERIMENT_CONFIG['val_set_ids'])]:
        split_dir = combo_dir / split
        split_dir.mkdir(exist_ok=True)
        for set_id in ids:
            df = pd.read_csv(EXPERIMENT_CONFIG['base_path'] / f"c{set_id}" / f"c{set_id}_wear.csv")
            for _, row in df.iterrows():
                cut = int(row['cut'])
                data_file = EXPERIMENT_CONFIG['base_path'] / f"c{set_id}" / f"c{set_id}" / f"c_{set_id}_{cut:03d}.csv"
                full_data = pd.read_csv(data_file, header=None).values[:, :7]

                if EXPERIMENT_CONFIG['data_windowing']['enable']:
                    data = extract_window(
                        full_data,
                        EXPERIMENT_CONFIG['data_windowing']['offset'],
                        EXPERIMENT_CONFIG['data_windowing']['window_size']
                    )
                else:
                    data = full_data

                features = []
                for ch in range(data.shape[1]):
                    mel = generate_mel(data[:, ch], sr, n_fft, hop_length, mel_n)
                    mel_resized = F.interpolate(
                        mel.unsqueeze(0), size=(224, 224),
                        mode='bilinear', align_corners=False
                    ).squeeze(0)

                    with torch.no_grad():
                        feat = extractor(mel_resized.unsqueeze(0).to(EXPERIMENT_CONFIG['device']))
                    features.append(feat.view(-1).cpu().numpy())

                features = np.stack(features)  # (7, feat_dim)
                out_file = split_dir / f"c{set_id}_{cut:03d}.npy"
                np.save(out_file, features)
                print(f"[{split}] Saved {out_file}")

def main():
    EXPERIMENT_CONFIG['output_dir'].mkdir(parents=True, exist_ok=True)
    for cfg in ABLATION_CONFIG['feature_configs']:
        for mel_n in cfg['mel_filters']:
            for overlap in ABLATION_CONFIG['overlap_ratio']:
                for arch in ABLATION_CONFIG['resnet_arch']:
                    process_combo(cfg['window_ms'], mel_n, overlap, arch)

if __name__ == "__main__":
    main()