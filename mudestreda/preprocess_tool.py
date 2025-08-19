#!/usr/bin/env python3
import os
import re
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

# ================== 可调参数 ==================
NUM_TILES = 1           # 水平切块数 K
ALLOW_OVERWRITE = True   # True: 已存在则覆盖；False: 已存在则跳过
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

# 1) 根据模型名加载并去掉分类头（输出 B×C×1×1）
def get_feature_extractor(name, device):
    if name == 'densenet121':
        m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        feat = nn.Sequential(m.features, nn.AdaptiveAvgPool2d(1))
    elif name == 'efficientnet_b0':
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        feat = nn.Sequential(m.features, nn.AdaptiveAvgPool2d(1))
    elif name == 'mobilenet_v2':
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        feat = nn.Sequential(m.features, nn.AdaptiveAvgPool2d(1))
    elif name == 'mnasnet1_0':
        m = models.mnasnet1_0(weights=models.MNASNet1_0_Weights.IMAGENET1K_V1)
        feat = nn.Sequential(m.layers, nn.AdaptiveAvgPool2d(1))
    elif name == 'convnext_tiny':
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        feat = nn.Sequential(m.features, nn.AdaptiveAvgPool2d(1))
    elif name == 'regnet_y_400mf':
        m = models.regnet_y_400mf(weights=models.RegNet_Y_400MF_Weights.IMAGENET1K_V1)
        feat = nn.Sequential(*list(m.children())[:-2], nn.AdaptiveAvgPool2d(1))
    elif name == 'vit_b_16':
        m = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        try:
            m.head = nn.Identity()
        except AttributeError:
            m.heads = nn.Identity()
        class Unsqueeze(nn.Module):
            def forward(self, x): return x.unsqueeze(-1).unsqueeze(-1)
        feat = nn.Sequential(m, Unsqueeze())   # -> [B, D, 1, 1]
    elif name == 'swin_tiny_patch4_window7_224':
        m = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        feat = nn.Sequential(*list(m.children())[:-1], nn.AdaptiveAvgPool2d(1))
    # ===== 新增 ResNet 系列 =====
    elif name == 'resnet18':
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        feat = nn.Sequential(*list(m.children())[:-2], nn.AdaptiveAvgPool2d(1))
    elif name == 'resnet34':
        m = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        feat = nn.Sequential(*list(m.children())[:-2], nn.AdaptiveAvgPool2d(1))
    elif name == 'resnet50':
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        feat = nn.Sequential(*list(m.children())[:-2], nn.AdaptiveAvgPool2d(1))
    else:
        raise ValueError(f'Unknown model {name}')
    return feat.to(device).eval()

# 2) 预处理
preprocess = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])

# 3) 等宽水平切成 K 块
def split_equal_horizontal(img: Image.Image, k: int):
    W, H = img.size
    xs = np.linspace(0, W, k + 1, dtype=int)
    return [img.crop((xs[i], 0, xs[i + 1], H)) for i in range(k)]

# 4) 刀具图像分块特征提取并保存为 (K, C)
def extract_and_save_tool_tiled(img_path, model, device, out_path, num_tiles=6):
    if (not ALLOW_OVERWRITE) and os.path.exists(out_path):
        return
    img = Image.open(img_path).convert('RGB')
    tiles = split_equal_horizontal(img, num_tiles)
    xs = torch.stack([preprocess(t) for t in tiles], dim=0).to(device)  # (K,3,224,224)
    with torch.no_grad():
        f = model(xs)                                     # (K,C,1,1)
        feats = f.squeeze(-1).squeeze(-1).cpu().numpy()   # (K,C)
    np.save(out_path, feats)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbones = ['vit_b_16',
    'resnet50', 'densenet121','resnet18','resnet34',
        'efficientnet_b0','mobilenet_v2','mnasnet1_0',
        'convnext_tiny','regnet_y_400mf','vit_b_16',
    ]

    src_dir = os.path.join(args.input_root, 'tool')
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"Directory not found: {src_dir}")

    for bb in backbones:
        print(f"\n==== Using backbone: {bb} ====")
        model = get_feature_extractor(bb, device)
        for fname in os.listdir(src_dir):
            if not fname.lower().endswith(IMG_EXTS):
                continue
            m = re.search(r'(T\d+)', fname)
            tool_folder = m.group(1) if m else 'unknown'
            out_dir = os.path.join(args.output_root, tool_folder, bb, 'tool')
            os.makedirs(out_dir, exist_ok=True)
            base, _ = os.path.splitext(fname)
            out_path = os.path.join(out_dir, base + '.npy')
            src_path = os.path.join(src_dir, fname)
            try:
                extract_and_save_tool_tiled(src_path, model, device, out_path, num_tiles=NUM_TILES)
                print(f"[OK] {fname} -> {out_path}")
            except Exception as e:
                print(f"[WARN] fail on {src_path}: {e}")

if __name__ == '__main__':
    input_root  = r"D:\Mudestreda\dataset_original\dataset_original"
    output_root = r"D:\py\RPS-MODIFY\Mudestreda\features2"
    class Args: pass
    args = Args()
    args.input_root  = input_root
    args.output_root = output_root
    main(args)