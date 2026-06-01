import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse
 

class VGATEncoder(nn.Module):
    """
    Graph encoder for task structure only.

    Input:
        node_features: [B, N, F] or [N, F]
        adj_matrix:    [B, N, N] or [N, N]

    Output:
        graph_embedding: [B, out_dim] or [out_dim]
    """

    def __init__(
        self,
        node_feature_dim=6,
        hidden_dim=64,
        out_dim=128,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()

        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout = dropout

        self.input_proj = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.gat1 = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=num_heads,
            concat=True,
            dropout=dropout,
        )

        self.gat2 = GATConv(
            in_channels=hidden_dim * num_heads,
            out_channels=out_dim,
            heads=1,
            concat=False,
            dropout=dropout,
        )

        self.readout = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
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

            # Safe fallback if graph is empty for any reason
            if edge_index.numel() == 0:
                graph_embedding = x.mean(dim=0)
                graph_embedding = self.readout(graph_embedding)
                outputs.append(graph_embedding)
                continue

            x = self.gat1(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.gat2(x, edge_index)
            x = F.elu(x)

            graph_embedding = x.mean(dim=0)
            graph_embedding = self.readout(graph_embedding)

            outputs.append(graph_embedding)

        out = torch.stack(outputs, dim=0)

        if single_graph:
            return out.squeeze(0)

        return out
    
if __name__ == "__main__":
    B, N, feat_dim = 2, 12, 6
    node_features = torch.randn(B, N, feat_dim)
    adj_matrix = torch.randint(0, 2, (B, N, N)).float()

    model = VGATEncoder(node_feature_dim=6, hidden_dim=64, out_dim=128, num_heads=4)
    out = model(node_features, adj_matrix)

    print(out.shape)