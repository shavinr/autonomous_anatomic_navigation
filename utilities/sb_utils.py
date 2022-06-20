import os
import gym

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from typing import Optional, Callable

from simulus.wrappers import NormalizeActionWrapper


def make_wrapped_env(env_id: str,
                     training: bool = True,
                     monitor_log_dir: Optional[str] = None,
                     seed: int = 0
) -> Callable:

    case_id = 2
    monitor_path = os.path.join(monitor_log_dir, f"case_id_{case_id}_seed_{seed}_training={training}") if monitor_log_dir is not None else None

    def _init() -> gym.Env:
        env = gym.make(env_id, case_id=case_id)
        set_random_seed(seed+case_id)
        env.reset(seed=seed+case_id)
        # Wrap the environment in a Monitor wrapper to have additional training information
        # Create the monitor folder if needed
        env = gym.wrappers.TimeLimit(env, max_episode_steps=200)
        env = NormalizeActionWrapper(Monitor(env, filename=monitor_path))
        return env

    return _init

