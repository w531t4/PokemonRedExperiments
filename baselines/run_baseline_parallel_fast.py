from pathlib import Path
import uuid
from typing import Union, List
import pokemonred_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from tensorboard_callback import TensorboardCallback
from datetime import datetime
from find_dir import get_current_dir
import sys

def get_stamp() -> str:
    return datetime.utcnow().strftime('%Y%m%d_%H%M')

if __name__ == '__main__':
    if len(sys.argv) == 2:
        num_cpu = int(sys.argv[1])
    else:
        num_cpu = 40 # Also sets the number of episodes per training iteration

    use_wandb_logging = False

    sess_id = str(uuid.uuid4())[:8]
    stamp = get_stamp()
    ep_length = 2048 * 10
    sess_path = Path(f'sessions/session_{stamp}_{sess_id}')
    env = make_vec_env(env_id="PokeRed-v0",
                       n_envs=num_cpu,
                       seed=None,
                       vec_env_cls=SubprocVecEnv,
                       env_kwargs=dict(pyboy_bequiet='--quiet' in sys.argv,
                                       gb_path=Path("../PokemonRed.gb"),
                                       init_state=Path('../has_pokedex_nballs.state'),
                                       session_path=sess_path,
                                       max_steps=ep_length,
                                       )
                       )
    checkpoint_callback = CheckpointCallback(save_freq=ep_length,
                                             save_path=sess_path,
                                             name_prefix='poke',
                                             )
    callbacks: List[Union[CheckpointCallback,
                          TensorboardCallback]] = [
                                                   checkpoint_callback,
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
        model = PPO.load(file_name,
                         env=env,
                         force_reset=True,
                         )
        # This code appears to break the values originally set the original policy...
        #model.n_steps = ep_length // 8
        #model.n_envs = num_cpu
        #model.rollout_buffer.buffer_size = ep_length
        #model.rollout_buffer.n_envs = num_cpu
        #model.rollout_buffer.reset()
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
