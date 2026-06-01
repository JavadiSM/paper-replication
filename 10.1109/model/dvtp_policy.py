import torch
import torch.nn as nn

from gymnasium import spaces

from stable_baselines3 import PPO # type: ignore

from stable_baselines3.common.torch_layers import (# type: ignore
    BaseFeaturesExtractor
)

try:
    from model.vgat_encoder import VGATEncoder
    from model.transformer_predictor import TransformerPredictor
except:
    from vgat_encoder import VGATEncoder
    from transformer_predictor import TransformerPredictor

class DVTPFeatureExtractor(
    BaseFeaturesExtractor
):
    """
    Equations (23)-(26)

    Final state:
        state = concat(
            stask,
            slocations,
            snodes
        )
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
        # VGAT over DAG only
        # ==================================================

        self.vgat = VGATEncoder()

        # ==================================================
        # Equation (24)
        # Transformer over location histories
        # ==================================================

        self.transformer = TransformerPredictor()

        # ==================================================
        # Equation (25)
        # Runtime encoder
        # ==================================================

        self.node_mlp = nn.Sequential(

            nn.Linear(2, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
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
        # runtime stream
        # ==================================================

        runtime = observations[
            "node_runtime"
        ]

        # runtime:
        # [B, num_locations, 2]

        runtime = runtime.mean(
            dim=1
        )

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

            "features_extractor_kwargs": {},

            "net_arch": {

                "pi": [256, 128],

                "vf": [256, 128]
            }
        })
        
        kwargs.setdefault("learning_rate", 1e-4)

        kwargs.setdefault("batch_size", 256)

        kwargs.setdefault("gamma", 0.9999)

        kwargs.setdefault("gae_lambda", 0.95)

        kwargs.setdefault("vf_coef", 0.5)

        kwargs.setdefault("ent_coef", 0.01)

        kwargs.setdefault("clip_range", 0.2)

        super().__init__(
            "MultiInputPolicy",
            *args,
            policy_kwargs=policy_kwargs,
            **kwargs
        )