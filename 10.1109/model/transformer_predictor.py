import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        max_len=500
    ):

        super().__init__()

        pe = torch.zeros(
            max_len,
            d_model
        )

        position = torch.arange(
            0,
            max_len,
            dtype=torch.float
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2
            ).float()
            *
            (
                -math.log(10000.0)
                /
                d_model
            )
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        pe = pe.unsqueeze(0)

        self.register_buffer(
            "pe",
            pe
        )

    def forward(self, x):

        return x + self.pe[:, :x.size(1)] # type: ignore


class TransformerPredictor(nn.Module):
    """
    Equation (24)

    slocations = Transformer(olocations)

    olocations:
        (
            ul_channel,
            task_num,
            latitude,
            longitude,
            f,
            CFT
        )

    Input:
        [B, num_locations, history_len, 6]

    Output:
        [B, 128]
    """

    def __init__(
        self,
        input_dim=6,
        d_model=512,
        nhead=8,
        num_layers=6,
        ffn_dim=512,
        output_dim=128
    ):

        super().__init__()

        self.input_projection = nn.Linear(
            input_dim,
            d_model
        )

        self.positional_encoding = PositionalEncoding(
            d_model
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=0.1,
            activation="relu",
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.output_head = nn.Sequential(
            nn.Linear(
                d_model,
                output_dim
            ),
            nn.ReLU()
        )

    def forward(
        self,
        locations
    ):
        """
        locations:
            [B, num_locations, history_len, 6]
        """

        B, L, T, F = locations.shape

        # Merge batch and location dimensions
        x = locations.reshape(
            B * L,
            T,
            F
        )

        # Input projection
        x = self.input_projection(x)

        # Positional encoding
        x = self.positional_encoding(x)

        # Transformer encoder
        x = self.transformer(x)

        # Temporal pooling
        x = x.mean(dim=1)

        # Output projection
        x = self.output_head(x)

        # Restore location dimension
        x = x.reshape(
            B,
            L,
            output_dim := x.shape[-1]
        )

        # Aggregate all locations
        x = x.mean(dim=1)

        return x


if __name__ == "__main__":

    B = 2
    num_locations = 10
    history_len = 20
    feature_dim = 6

    locations = torch.randn(
        B,
        num_locations,
        history_len,
        feature_dim
    )

    model = TransformerPredictor()

    output = model(locations)

    print("Output shape:", output.shape)
    # Expected: [2, 128]