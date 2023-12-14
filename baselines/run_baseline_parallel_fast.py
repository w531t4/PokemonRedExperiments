from os.path import exists
from pathlib import Path
import uuid
from typing import Dict, Union, List
from red_gym_env import RedGymEnv, RGEnvConfig
from stable_baselines3 import PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from tensorboard_callback import TensorboardCallback
from datetime import datetime
from find_dir import get_current_dir
import sys

def get_stamp() -> str:
    return datetime.utcnow().strftime('%Y%m%d_%H%M')

def make_env(rank: int,
             env_conf: Dict[str, Union[bool, int, str, float]],
             seed: int = 0,
             ):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = RedGymEnv(env_conf)
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':

    if len(sys.argv) == 2:
        num_cpu = int(sys.argv[1])
    else:
        num_cpu = 40 # Also sets the number of episodes per training iteration

    use_wandb_logging = False
    ep_length = 2048 * 10
    sess_id = str(uuid.uuid4())[:8]
    stamp = get_stamp()
    sess_path = Path(f'sessions/session_{stamp}_{sess_id}')

    env_config: RGEnvConfig = {'headless': True,
                  'save_final_state': True,
                  'early_stop': False,
                  'action_freq': 24,
                  'init_state': '../has_pokedex_nballs.state',
                  'max_steps': ep_length,
                  'print_rewards': True,
                  'save_video': False,
                  'fast_video': True,
                  'session_path': sess_path,
                  'gb_path': '../PokemonRed.gb',
                  'debug': False,
                  'sim_frame_dist': 2_000_000.0,
                  'use_screen_explore': True,
                  'reward_scale': 4,
                  'extra_buttons': False,
                  'explore_weight': 3 # 2.5
            }

    print(env_config)

    env = SubprocVecEnv([make_env(rank=i,
                                  env_conf=env_config,
                                  ) for i in range(num_cpu)])

    checkpoint_callback = CheckpointCallback(save_freq=ep_length,
                                             save_path=sess_path,
                                             name_prefix='poke',
                                             )
    callbacks: List[Union[CheckpointCallback,
                          TensorboardCallback]] = [checkpoint_callback,
                                                   TensorboardCallback(),
                                                   ]

    if use_wandb_logging:
        import wandb
        from wandb.integration.sb3 import WandbCallback
        run = wandb.init(
            project="pokemon-train",
            id=sess_id,
            config=env_config,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        callbacks.append(WandbCallback())

    #env_checker.check_env(env)
    learn_steps = 40
    # put a checkpoint here you want to start from
    #file_name = 'session_e41c9eff/poke_38207488_steps'
    try:
        current_dir = get_current_dir(basepath=Path(__file__).parent / "sessions")
        step_files = list(current_dir.glob("poke_*_steps.zip"))
    except IndexError:
        step_files = list()

    if len(step_files) > 0:
        last_stepfile = sorted(step_files, key=lambda z: int(z.name.split("_")[1]))[-1]
        prefix = "%s/" % str(Path().cwd() / "baselines")
        file_name = str(last_stepfile).replace("%s/" % str(last_stepfile), "").replace(".zip", "")

        print('\nloading checkpoint file=%s.zip' % file_name)
        model = PPO.load(file_name, env=env)
        model.n_steps = ep_length
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = ep_length
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        model = PPO('CnnPolicy',
                    env,
                    verbose=1,
                    n_steps=ep_length // 8,
                    batch_size=128,
                    n_epochs=3,
                    gamma=0.998,
                    tensorboard_log=sess_path,
                    )

    for i in range(learn_steps):
        model.learn(total_timesteps=(ep_length)*num_cpu*1000,
                    callback=CallbackList(callbacks),
                    )

    if use_wandb_logging:
        run.finish()
