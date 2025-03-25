import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class MaterialAnalysisModel(nn.Module):
    def __init__(self, num_colors, color_embed_dim=32, num_numerical_features=16,
                 hidden_dim=128, num_classes=10, dropout=0.3):
        """
        初始化材料分析模型

        参数:
            num_colors: 颜色词汇表大小
            color_embed_dim: 颜色嵌入维度
            num_numerical_features: 数值特征数量
            hidden_dim: 隐藏层维度
            num_classes: 分类类别数量
            dropout: Dropout比率
        """
        super(MaterialAnalysisModel, self).__init__()

        # 1. 颜色嵌入分支 - 将颜色索引转换为稠密向量
        self.color_embedding = nn.Embedding(num_colors, color_embed_dim)
        self.color_fc = nn.Linear(color_embed_dim, hidden_dim)

        # 2. 数值特征分支 - 使用多层感知器处理数值特征
        self.numerical_fc1 = nn.Linear(num_numerical_features, 64)
        self.numerical_fc2 = nn.Linear(64, hidden_dim)

        # 3. Transformer 模块用于模态融合 - 使用自注意力机制
        encoder_layers = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,  # 8个注意力头
            dim_feedforward=512,  # 前馈网络维度
            dropout=dropout,
            batch_first=True  # 批量维度在前
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, num_layers=2)  # 2层Transformer编码器

        # 4. 融合层 - 整合来自不同模态的特征
        self.fusion_fc1 = nn.Linear(hidden_dim * 2, 256)
        self.fusion_fc2 = nn.Linear(256, 128)
        self.batch_norm = nn.BatchNorm1d(128)  # 批量归一化提高训练稳定性

        # 5. 输出层 - 产生分类结果
        self.output_fc = nn.Linear(128, num_classes)

        # Dropout - 正则化技术，防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, color_idx, numerical_features):
        """
        前向传播函数

        参数:
            color_idx: 颜色索引 [batch_size]
            numerical_features: 数值特征 [batch_size, num_features]

        返回:
            logits: 分类logits [batch_size, num_classes]
        """
        # 颜色分支处理
        color_embed = self.color_embedding(
            color_idx)  # [batch, color_embed_dim]
        color_hidden = F.relu(self.color_fc(color_embed)
                              )  # [batch, hidden_dim]

        # 数值分支处理
        numerical_hidden = F.relu(self.numerical_fc1(
            numerical_features))  # [batch, 64]
        numerical_hidden = self.dropout(numerical_hidden)  # 应用dropout
        numerical_hidden = F.relu(self.numerical_fc2(
            numerical_hidden))  # [batch, hidden_dim]
        numerical_hidden = self.dropout(numerical_hidden)  # 应用dropout

        # 准备 Transformer 的输入 - 将两个模态特征拼接成序列
        transformer_input = torch.stack(
            [color_hidden, numerical_hidden], dim=1)  # [batch, 2, hidden_dim]

        # Transformer 进行模态间交互和特征融合
        transformer_output = self.transformer_encoder(
            transformer_input)  # [batch, 2, hidden_dim]

        # 将 Transformer 输出展平
        batch_size = transformer_output.size(0)
        fused_features = transformer_output.reshape(
            batch_size, -1)  # [batch, 2*hidden_dim]

        # 融合层 - 进一步整合多模态特征
        fusion_hidden = F.relu(self.fusion_fc1(fused_features))
        fusion_hidden = self.dropout(fusion_hidden)
        fusion_hidden = F.relu(self.fusion_fc2(fusion_hidden))
        fusion_hidden = self.batch_norm(fusion_hidden)  # 批量归一化
        fusion_hidden = self.dropout(fusion_hidden)

        # 输出层 - 生成各类别的得分
        logits = self.output_fc(fusion_hidden)

        return logits
