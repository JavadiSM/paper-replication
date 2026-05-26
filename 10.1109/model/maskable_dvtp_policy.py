import torch
import torch.nn as nn

from gymnasium import spaces

from sb3_contrib import MaskablePPO

from stable_baselines3.common.torch_layers import ( # type: ignore
    BaseFeaturesExtractor
)
from model.vgat_encoder import VGATEncoder
from model.transformer_predictor import (
        TransformerPredictor
)


# =========================================================
# Feature Extractor
# =========================================================

class DVTPFeatureExtractor(
    BaseFeaturesExtractor
):
    """
    Implements equations (23)-(26)

    stask = GAT(otask)
    slocations = Transformer(olocations)
    snodes = MLP(onodes)

    S = concat(...)
    """

    def __init__(
        self,
        observation_space: spaces.Dict
    ):

        super().__init__(
            observation_space,
            features_dim=384
        )

        # -------------------------------------------------
        # Equation (23)
        # -------------------------------------------------

        self.vgat = VGATEncoder(
            node_feature_dim=6,
            trajectory_dim=5,
            hidden_dim=64,
            out_dim=128
        )

        # -------------------------------------------------
        # Equation (24)
        # -------------------------------------------------

        self.transformer = TransformerPredictor(
            input_dim=6,
            d_model=64,
            output_dim=128
        )

        # -------------------------------------------------
        # Equation (25)
        # -------------------------------------------------

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

        # =================================================
        # Task graph stream
        # =================================================

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

        # =================================================
        # Location stream
        # =================================================

        slocations = self.transformer(

            observations[
                "locations"
            ]
        )

        # =================================================
        # Runtime stream
        # =================================================

        runtime = observations[
            "node_runtime"
        ]

        runtime = runtime.mean(
            dim=1
        )

        snodes = self.node_mlp(
            runtime
        )

        # =================================================
        # Equation (26)
        # =================================================

        state = torch.cat(
            [
                stask,
                slocations,
                snodes
            ],
            dim=-1
        )

        return state


# =========================================================
# Main PPO Wrapper
# =========================================================

class MaskableDVTPPPO(
    MaskablePPO
):
    """
    SB3-style API.

    Usage:

    model = MaskableDVTPPPO(
        env=env
    )

    model.learn(...)
    """

    def __init__(
        self,
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device="cpu",
        verbose=1,
        tensorboard_log=None,
        seed=None,
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

        super().__init__(

            policy="MultiInputPolicy",

            env=env,

            learning_rate=learning_rate,

            n_steps=n_steps,

            batch_size=batch_size,

            gamma=gamma,

            gae_lambda=gae_lambda,

            clip_range=clip_range,

            ent_coef=ent_coef,

            vf_coef=vf_coef,

            max_grad_norm=max_grad_norm,

            policy_kwargs=policy_kwargs,

            tensorboard_log=tensorboard_log,

            verbose=verbose,

            device=device,

            seed=seed,

            **kwargs
        )


# =========================================================
# Debug Run
# =========================================================

if __name__ == "__main__":

    from env.my_env import VEC

    env = VEC(
        num_nodes=10,
        num_ve=5,
        num_ves=2
    )

    model = MaskableDVTPPPO(
        env=env,
        verbose=1
    )

    print(model)

    obs, info = env.reset()

    action, _ = model.predict(
        obs,
        action_masks=info["action_mask"]
    )

    print("\nSample Action:")
    print(action)
