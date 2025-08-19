#!/usr/bin/env python3
import torch
import torch.nn as nn
# ============================= 跨模态注意力 =============================
class CrossModalAttention(nn.Module):
    def __init__(self, d_model, nhead=4):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model*4, d_model)
        )

    def forward(self, signal_features, tool_features):
        # signal_features: (B, 3, D), tool_features: (B, K, D)
        attn_out, attn_w = self.multihead_attn(
            query=signal_features,
            key=tool_features,
            value=tool_features
        )   # attn_out: (B,3,D), attn_w: (B,3,K)
        # 残差 + LayerNorm
        x = self.norm1(signal_features + self.dropout(attn_out))
        # 前馈
        y = self.ffn(x)
        out = self.norm2(x + self.dropout(y))
        return out, attn_w                     # out: (B,3,D)

# ======================= 改进的融合模型 =======================
class CrossModalTransformerFusionModel(nn.Module):
    def __init__(self, input_dim, reduced_dim, num_layers=2, nhead=4, num_classes=3):
        super().__init__()
        # Shared FC + LN
        self.shared_fc = nn.Linear(input_dim, reduced_dim)
        self.bn = nn.BatchNorm1d(reduced_dim)

        # 跨模态注意力
        self.cross_modal_attention = CrossModalAttention(reduced_dim, nhead)

        # TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=reduced_dim,
            nhead=nhead,
            batch_first=True,
            dropout=0.3
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 模态注意力 & 分类器
        self.modal_attention = nn.Linear(reduced_dim, 1)
        self.classifier = nn.Linear(reduced_dim, num_classes)

    def forward(self, mel_features, tool_features):
        # x: (B, 3+K, D) -> (B, 3+K, R)
        x = torch.cat([mel_features, tool_features], dim=1)
        x = self.shared_fc(x)

        # BN 在通道维 R 上：把 (B, 3+K, R) 变成 (B, R, 3+K)
        x = x.transpose(1, 2)  # (B, R, 3+K)
        x = self.bn(x)  # BN over R
        x = x.transpose(1, 2)  # (B, 3+K, R)

        # 跨模态注意力（mel→tool）
        mel_red, tool_red = x[:, :3, :], x[:, 3:, :]         # mel:(B,3,R), tool:(B,K,R)
        mel_enh, cross_attn_w = self.cross_modal_attention(mel_red, tool_red)  # attn_w:(B,3,K)

        # Transformer + 池化
        fusion_seq = torch.cat([mel_enh, tool_red], dim=1)   # (B, 3+K, R)
        trans_out = self.transformer_encoder(fusion_seq)     # (B, 3+K, R)

        attn_w = torch.softmax(self.modal_attention(trans_out).squeeze(-1), dim=1)  # (B, 3+K)
        fused = torch.sum(trans_out * attn_w.unsqueeze(-1), dim=1)                  # (B, R)

        logits = self.classifier(fused)                      # (B, num_classes)
        return logits, cross_attn_w, mel_enh, tool_red

# =========================== 损失函数 ===========================
class MultiModalLoss(nn.Module):
    def __init__(self, lambda1=0.1, lambda2=0.01):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.ce = nn.CrossEntropyLoss()

    def forward(self, preds, targets, mel_f, tool_f, attn_w):
        L_ce = self.ce(preds, targets)
        mel_norm = torch.norm(mel_f, dim=-1).mean()
        tool_norm = torch.norm(tool_f, dim=-1).mean()
        L_mod = torch.abs(mel_norm - tool_norm)
        entropy = -torch.sum(attn_w * torch.log(attn_w + 1e-8), dim=-1).mean()
        L_att = -entropy
        return L_ce + self.lambda1 * L_mod + self.lambda2 * L_att, L_ce, L_mod, L_att


def print_model_size(model: torch.nn.Module) -> None:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    nontrainable_params = total_params - trainable_params
    total_buffers = sum(b.numel() for b in model.buffers())

    params_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffers_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    total_bytes = params_bytes + buffers_bytes

    state_dict_bytes = sum(t.numel() * t.element_size() for t in model.state_dict().values())

    to_mb = lambda x: x / 1024**2

    print(f"参数总数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"不可训练参数: {nontrainable_params:,}")
    print(f"缓冲区元素数: {total_buffers:,}")
    print(f"参数内存: {to_mb(params_bytes):.2f} MB")
    print(f"缓冲区内存: {to_mb(buffers_bytes):.2f} MB")
    print(f"模型总内存 参数+缓冲区: {to_mb(total_bytes):.2f} MB")
    print(f"state_dict 体积 估算: {to_mb(state_dict_bytes):.2f} MB")

if __name__ == "__main__":
    model = CrossModalTransformerFusionModel(
        input_dim=512,      # 你的 D
        reduced_dim=128,     # 你的 R
        num_layers=2,
        nhead=4,
        num_classes=3
    )
    print_model_size(model)