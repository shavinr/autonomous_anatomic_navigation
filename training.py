import os
import time
import numpy as np
import pandas as pd
from datetime import datetime

from simulus.utilities import linear_schedule, make_wrapped_env

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed


if __name__ == "__main__":
    # Everything is placed in main to ensure threadsafe behavior, please refer to _ for details

    # Environment configurations
    env_id = "CTsim-v0"

    # For future appropriate training/test/validation splitup of all CT cases, not in use per now
    training_cases = ...
    validation_cases = ...
    testing_cases = ...

    # Saving and loading configurations
    save_dir = "./PPO_probe_navigation_training/MIP_{}".format(datetime.now().strftime('%Y-%m-%d-%H-%M'))
    os.makedirs(save_dir, exist_ok=True)

    monitor_log_dir = os.path.join(save_dir, "monitor_pure_env")
    os.makedirs(monitor_log_dir, exist_ok=True)

    tb_log_dir = os.path.join(save_dir, "tensorboard")
    os.makedirs(tb_log_dir, exist_ok=True)

    checkpoint_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    stats_save_path = os.path.join(save_dir, "vec_normalize.pkl")
    model_save_path = os.path.join(save_dir, "saved_model_wrapped_env")

    best_model_save_dir = os.path.join(save_dir, "best_model")
    os.makedirs(best_model_save_dir, exist_ok=True)

    # SB3 configurations
    total_n_timesteps = 1000000
    n_envs = 16  # Number of processors to use
    save_freq = 50000
    save_freq = max(save_freq // n_envs, 1)
    n_eval_eps = 3

    # Create the vectorized environment, i.e. parallel environments for multiprocessed RL training.
    # The following creates a wrapped, _monitored_ VecEnv, also automatically normalizes the input features and reward
    wrapped_env = VecNormalize(SubprocVecEnv([make_wrapped_env(env_id, monitor_log_dir=monitor_log_dir, training=True,
                                                               seed=i) for i in range(n_envs)]))
    wrapped_env.reset()

    n_eval_envs = 1
    vec_eval_env = VecNormalize(SubprocVecEnv([make_wrapped_env(env_id, monitor_log_dir=monitor_log_dir, training=False,
                                                                seed=i) for i in range(n_eval_envs)]))
    vec_eval_env.reset()

    # Create callbacks
    checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=checkpoint_dir, verbose=1)
    # Note that the report used eval_freq = 10000, by misfortune
    eval_callback = EvalCallback(eval_env=wrapped_env, best_model_save_path=best_model_save_dir,
                                 log_path=best_model_save_dir, eval_freq=save_freq, verbose=1)

    # Create model.
    model = PPO(policy="MultiInputPolicy", env=wrapped_env, verbose=1, tensorboard_log=tb_log_dir, batch_size=256,
                n_steps=1024, device="cuda")

    # Training
    model.learn(total_timesteps=total_n_timesteps, callback=[checkpoint_callback, eval_callback], tb_log_name="events",
                reset_num_timesteps=True)

    model.save(model_save_path)

    # Must also save the running mean and std along with the model when using the VecNormalize wrapper,
    # otherwise won't get proper results when loading the agent again.
    wrapped_env.save(stats_save_path)

    # Ensure safe closure of all environments, avoids problems with memory overload
    wrapped_env.close()
    vec_eval_env.close()

    # Run plot_utils as main to see results