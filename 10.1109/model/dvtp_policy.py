import torch
import torch.nn as nn

from gymnasium import spaces

from stable_baselines3 import PPO # type: ignore

from stable_baselines3.common.torch_layers import ( # type: ignore
    BaseFeaturesExtractor
)

from model.vgat_encoder import VGATEncoder
from model.transformer_predictor import TransformerPredictor


class DVTPFeatureExtractor(
    BaseFeaturesExtractor
):
    """
    Faithful implementation of equations (23)-(26)
    """

    def __init__(
        self,
        observation_space: spaces.Dict
    ):

        super().__init__(
            observation_space,
            features_dim=384
        )

        # ==================================================
        # Equation (23)
        # ==================================================

        self.vgat = VGATEncoder(
            node_feature_dim=6,
            trajectory_dim=5,
            hidden_dim=64,
            out_dim=128
        )

        # ==================================================
        # Equation (24)
        # ==================================================

        self.transformer = TransformerPredictor(
            input_dim=6,
            d_model=64,
            output_dim=128
        )

        # ==================================================
        # Equation (25)
        # ==================================================

        self.node_mlp = nn.Sequential(

            nn.Linear(
                2,
                64
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                128
            ),

            nn.ReLU()
        )

    def forward(
        self,
        observations
    ):

        # ==================================================
        # task graph stream
        # ==================================================

        stask = self.vgat(

            node_features=observations[
                "node_features"
            ],

            adj_matrix=observations[
                "adj_matrix"
            ],

            trajectory_features=observations[
                "trajectory"
            ]
        )

        # ==================================================
        # location stream
        # ==================================================

        slocations = self.transformer(

            observations[
                "locations"
            ]
        )

        # ==================================================
        # node runtime stream
        # ==================================================

        runtime = observations[
            "node_runtime"
        ]

        runtime = runtime.mean(dim=1)

        snodes = self.node_mlp(
            runtime
        )

        # ==================================================
        # Equation (26)
        # ==================================================

        state = torch.cat(
            [
                stask,
                slocations,
                snodes
            ],
            dim=-1
        )

        return state


class DVTPPPO(PPO):

    def __init__(
        self,
        *args,
        **kwargs
    ):

        policy_kwargs = kwargs.pop(
            "policy_kwargs",
            {}
        )

        policy_kwargs.update({

            "features_extractor_class":
                DVTPFeatureExtractor,

            "net_arch": {

                "pi": [256, 128],

                "vf": [256, 128]
            }
        })

        super().__init__(
            "MultiInputPolicy",
            *args,
            policy_kwargs=policy_kwargs,
            **kwargs
        )