import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from gymnasium import spaces
from stable_baselines3 import PPO  # type: ignore
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor  # type: ignore
from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse


class VGATBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.gat = GATConv(
            in_channels=d_model,
            out_channels=d_model // num_heads,
            heads=num_heads,
            concat=True,
            dropout=dropout,
        )

        self.ffnn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, edge_index):
        attn_out = self.gat(x, edge_index)
        attn_out = F.gelu(attn_out)
        x = self.norm1(x + attn_out)

        ffnn_out = self.ffnn(x)
        x = self.norm2(x + ffnn_out)
        return x


class VGATEncoder(nn.Module):
    def __init__(
        self,
        node_feature_dim: int = 6,
        hidden_dim: int = 256,
        ffn_dim: int = 512,
        out_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
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

        self.layers = nn.ModuleList([
            VGATBlock(
                d_model=hidden_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
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

            if edge_index.numel() == 0:
                graph_embedding = x.mean(dim=0)
                graph_embedding = self.readout(graph_embedding)
                outputs.append(graph_embedding)
                continue

            for layer in self.layers:
                x = layer(x, edge_index)

            graph_embedding = x.mean(dim=0)
            graph_embedding = self.readout(graph_embedding)
            outputs.append(graph_embedding)

        out = torch.stack(outputs, dim=0)

        if single_graph:
            return out.squeeze(0)

        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]  # type: ignore[index]


class TransformerPredictor(nn.Module):
    def __init__(
        self,
        input_dim: int = 6,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        output_dim: int = 128,
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.output_head = nn.Sequential(
            nn.Linear(d_model, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )

    def forward(self, locations):
        B, L, T, feat_dim = locations.shape

        x = locations.reshape(B * L, T, feat_dim)
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.output_head(x)
        x = x.reshape(B, L, x.shape[-1])
        x = x.mean(dim=1)
        return x


class DVTPFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=384)

        self.vgat = VGATEncoder()
        self.transformer = TransformerPredictor()

        self.node_mlp = nn.Sequential(
            nn.Linear(2, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
        )

    def forward(self, observations):
        stask = self.vgat(
            node_features=observations["node_features"],
            adj_matrix=observations["adj_matrix"],
        )

        slocations = self.transformer(observations["locations"])

        runtime = observations["node_runtime"]
        runtime = runtime.mean(dim=1)
        snodes = self.node_mlp(runtime)

        state = torch.cat([stask, slocations, snodes], dim=-1)
        return state


class IMPDVTP(PPO):
    def __init__(self, *args, **kwargs):
        policy_kwargs = kwargs.pop("policy_kwargs", {})

        policy_kwargs.update({
            "features_extractor_class": DVTPFeatureExtractor,
            "features_extractor_kwargs": {},
        })

        super().__init__(
            "MultiInputPolicy",
            *args,
            policy_kwargs=policy_kwargs,
            **kwargs,
        )
class DVTPFeatureExtractor2(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=384)

        self.vgat = VGATEncoder(hidden_dim=128, ffn_dim=512, num_layers=8)
        self.transformer = TransformerPredictor(num_layers=4, d_model=128)

        self.node_mlp = nn.Sequential(
            nn.Linear(2, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
        )

    def forward(self, observations):
        stask = self.vgat(
            node_features=observations["node_features"],
            adj_matrix=observations["adj_matrix"],
        )

        slocations = self.transformer(observations["locations"])

        runtime = observations["node_runtime"]
        runtime = runtime.mean(dim=1)
        snodes = self.node_mlp(runtime)

        state = torch.cat([stask, slocations, snodes], dim=-1)
        return state
class IMP2DVTP(PPO):
    def __init__(self, *args, **kwargs):
        policy_kwargs = kwargs.pop("policy_kwargs", {})

        policy_kwargs.update({
            "features_extractor_class": DVTPFeatureExtractor2,
            "features_extractor_kwargs": {},
        })

        super().__init__(
            "MultiInputPolicy",
            *args,
            policy_kwargs=policy_kwargs,
            **kwargs,
        )