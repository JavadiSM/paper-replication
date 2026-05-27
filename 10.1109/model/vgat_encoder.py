import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse


class VGATEncoder(nn.Module):
    """
    Equation (23)

    stask = GAT(otask)

    Inputs:
        - node_features
        - adjacency matrix
        - trajectory features
    """

    def __init__(
        self,
        node_feature_dim=6,
        trajectory_dim=5,
        hidden_dim=64,
        out_dim=128,
        num_heads=4
    ):

        super().__init__()

        self.trajectory_encoder = nn.Sequential(

            nn.Linear(
                trajectory_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                hidden_dim
            )
        )

        self.gat1 = GATConv(
            node_feature_dim + hidden_dim,
            hidden_dim,
            heads=num_heads,
            concat=True
        )

        self.gat2 = GATConv(
            hidden_dim * num_heads,
            out_dim,
            heads=1,
            concat=False
        )

    def forward(
        self,
        node_features,
        adj_matrix,
        trajectory_features
    ):
        """
        node_features:
            [B, N, 6]

        adj_matrix:
            [B, N, N]

        trajectory_features:
            [B, T, 5]
        """

        batch_size = node_features.shape[0]

        outputs = []

        for b in range(batch_size):

            x = node_features[b]

            adj = adj_matrix[b]

            traj = trajectory_features[b]

            edge_index, _ = dense_to_sparse(adj)

            # aggregate trajectory context
            traj_embed = self.trajectory_encoder(
                traj
            )

            traj_embed = traj_embed.mean(
                dim=0,
                keepdim=True
            )

            traj_embed = traj_embed.repeat(
                x.shape[0],
                1
            )

            x = torch.cat(
                [
                    x,
                    traj_embed
                ],
                dim=-1
            )

            x = self.gat1(
                x,
                edge_index
            )

            x = F.elu(x)

            x = self.gat2(
                x,
                edge_index
            )

            x = x.mean(dim=0)

            outputs.append(x)

        return torch.stack(outputs)

if __name__ == "__main__":

    from torch_geometric.data import Data

    num_nodes = 10

    input_dim = 6

    x = torch.randn(
        num_nodes,
        input_dim
    )

    edge_index = torch.tensor([
        [0, 0, 1, 2, 3, 4],
        [1, 2, 3, 4, 5, 6]
    ], dtype=torch.long)

    data = Data(
        x=x,
        edge_index=edge_index
    )

    model = VGATEncoder(
        input_dim=input_dim
    )

    output = model(
        data.x,
        data.edge_index
    )

    print("Latent Shape:")
    print(output["z"].shape)

    print("\nGraph Embedding Shape:")
    print(output["graph_embedding"].shape)

    loss_dict = model.loss_function(
        x=data.x,
        reconstruction=output["reconstruction"],
        mean=output["mean"],
        logvar=output["logvar"]
    )

    print("\nLosses:")
    print(loss_dict)