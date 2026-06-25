#!/usr/bin/env python3
"""
训练SAC和TD3模型
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.rocket_env import RocketLandingEnv


def train_algorithm(algorithm_name, total_timesteps=200000):
    """训练单个RL算法"""
    print(f"\n{'='*60}")
    print(f"训练算法: {algorithm_name}")
    print(f"{'='*60}")

    env = RocketLandingEnv()
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    if algorithm_name == "SAC":
        model = SAC(
            "MlpPolicy", env,
            learning_rate=3e-4, buffer_size=100000, batch_size=256,
            gamma=0.99, tau=0.005, ent_coef='auto',
            verbose=1, device="cpu"
        )
    elif algorithm_name == "TD3":
        model = TD3(
            "MlpPolicy", env,
            learning_rate=3e-4, buffer_size=100000, batch_size=256,
            gamma=0.99, tau=0.005,
            verbose=1, device="cpu"
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    # 保存模型
    model_dir = Path(__file__).parent / "models" / algorithm_name
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(model_dir / "model"))
    env.save(str(model_dir / "vec_normalize.pkl"))

    print(f"模型已保存到: {model_dir}")
    return model, env


if __name__ == "__main__":
    train_algorithm("SAC", total_timesteps=200000)
    train_algorithm("TD3", total_timesteps=200000)
