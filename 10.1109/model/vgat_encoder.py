import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse


class VGATBlock(nn.Module):

    def __init__(
        self,
        d_model=256,
        num_heads=8,
        ffn_dim=256,
        dropout=0.1
    ):
        super().__init__()
        # برای اینکه بعد خروجی پس از کانکت کردن هدها همان d_model (128) باقی بماند:
        # out_channels = d_model // num_heads -> 128 // 8 = 16
        self.gat = GATConv(
            in_channels=d_model,
            out_channels=d_model // num_heads,
            heads=num_heads,
            concat=True,
            dropout=dropout,
        )
        
        # لایه پیش‌خور (VGAT FFNN Hidden Dim = 512) همراه با فعال‌ساز LeakyReLU
        self.ffnn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, edge_index):
        # بخش اتنشن به همراه اتصال باقی‌مانده (Residual Connection)
        attn_out = self.gat(x, edge_index)
        attn_out = F.leaky_relu(attn_out, negative_slope=0.2)
        x = self.norm1(x + attn_out)
        
        # بخش لایه پیش‌خور (FFNN) به همراه اتصال باقی‌مانده
        ffnn_out = self.ffnn(x)
        x = self.norm2(x + ffnn_out)
        return x


class VGATEncoder(nn.Module):

    def __init__(
        self,
        node_feature_dim=6,
        hidden_dim=256,      # جدول: VGAT Hidden Units = 256
        ffn_dim=256,         # جدول فقط Hidden Units=256 را گفته
        out_dim=128,         # خروجی فعلی حفظ شود
        num_heads=8,         # جدول
        num_layers=4,        # جدول
        dropout=0.1,
    ):
        super().__init__()

        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout = dropout

        # پروجکشن اولیه ورودی با لایه فعال‌ساز LeakyReLU
        self.input_proj = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.LayerNorm(hidden_dim),
        )

        # ایجاد لایه‌های متوالی بر اساس تعداد لایه‌های جدول (4 لایه)
        self.layers = nn.ModuleList([
            VGATBlock(d_model=hidden_dim, num_heads=num_heads, ffn_dim=ffn_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        # لایه نهایی خروجی (Readout Projection)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.LayerNorm(out_dim),
        )

    def forward(self, node_features, adj_matrix):
        single_graph = False

        if node_features.dim() == 2:
            node_features = node_features.unsqueeze(0)
            adj_matrix = adj_matrix.unsqueeze(0)
            single_graph = True

        batch_size = node_features.size(0)
        outputs = []

        for b in range(batch_size):
            x = node_features[b].float()
            adj = adj_matrix[b].float()

            x = self.input_proj(x)

            edge_index, _ = dense_to_sparse(adj)

            # بک‌آپ امن در صورت خالی بودن گراف
            if edge_index.numel() == 0:
                graph_embedding = x.mean(dim=0)
                graph_embedding = self.readout(graph_embedding)
                outputs.append(graph_embedding)
                continue

            # عبور متوالی داده از هر 4 لایه VGAT
            for layer in self.layers:
                x = layer(x, edge_index)

            # میانگین‌گیری (Pooling) برای به دست آوردن امبدینگ کل گراف
            graph_embedding = x.mean(dim=0)
            graph_embedding = self.readout(graph_embedding)

            outputs.append(graph_embedding)

        out = torch.stack(outputs, dim=0)

        if single_graph:
            return out.squeeze(0)

        return out
        

if __name__ == "__main__":
    # تست صحت اجرای ساختار جدید
    B, N, feat_dim = 2, 12, 6
    node_features = torch.randn(B, N, feat_dim)
    adj_matrix = torch.randint(0, 2, (B, N, N)).float()

    # مقداردهی دقیقاً بر اساس مشخصات جدول هایپرپارامترها
    model = VGATEncoder(
        node_feature_dim=6,
        hidden_dim=256,
        ffn_dim=256,
        out_dim=128,
        num_heads=8,
        num_layers=4
    )
    out = model(node_features, adj_matrix)

    print("Output shape:", out.shape)  # خروجی باید [2, 128] باشد