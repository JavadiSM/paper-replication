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

        return x + self.pe[:, :x.size(1)]


class TransformerPredictor(nn.Module):
    """
    Equation (24)

    slocations = Transformer(olocations)

    olocation:
        (
            ul_channel,
            task_num,
            latitude,
            longitude,
            f,
            CFT
        )
    """

    def __init__(
        self,
        input_dim=6,
        d_model=64,
        nhead=4,
        num_layers=2,
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
            [B, num_locations, 6]
        """

        x = self.input_projection(
            locations
        )

        x = self.positional_encoding(x)

        x = self.transformer(x)

        x = x.mean(dim=1)

        x = self.output_head(x)

        return x