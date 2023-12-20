from os.path import exists
from pathlib import Path
import sys
import pokemonred_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

def run_save(save):
    save = Path(save)
    ep_length = 2048 * 8
    sess_path = f'grid_renders/session_{save.stem}'
    num_cpu = 40  # Also sets the number of episodes per training iteration
    env = make_vec_env(env_id="PokeRed-v0",
                       n_envs=num_cpu,
                       seed=None,
                       vec_env_cls=SubprocVecEnv,
                       env_kwargs=dict(pyboy_bequiet='--quiet' in sys.argv,
                                       gb_path=Path("../PokemonRed.gb"),
                                       init_state=Path('../has_pokedex_nballs.state'),
                                       session_path=sess_path,
                                       max_steps=ep_length,
                                       save_video=True,
                                       fast_vide=False,
                                       )
                       )
    checkpoint_callback = CheckpointCallback(save_freq=ep_length,
                                             save_path=sess_path,
                                             name_prefix='poke',
                                             )
    #env_checker.check_env(env)
    learn_steps = 1
    file_name = save
    if exists(file_name):
        print('\nloading checkpoint')
        custom_objects = {"learning_rate": 0.0,
                          "lr_schedule": lambda _: 0.0,
                          "clip_range": lambda _: 0.0,
                          "n_steps": ep_length,
        }
        model = PPO.load(file_name,
                         env=env,
                         custom_objects=custom_objects,
                         force_reset=True,
                         )
        # This code appears to break the values originally set the original policy...
        #model.n_steps = ep_length
        #model.n_envs = num_cpu
        #model.rollout_buffer.buffer_size = ep_length
        #model.rollout_buffer.n_envs = num_cpu
        #model.rollout_buffer.reset()
    else:
        print('initializing new policy')
        model = PPO('CnnPolicy',
                    env,
                    verbose=1,
                    n_steps=ep_length,
                    batch_size=512,
                    n_epochs=1,
                    gamma=0.999,
                    )

    model.learn(total_timesteps=(ep_length)*num_cpu,
                callback=checkpoint_callback,
                )


if __name__ == '__main__':
    run_save(sys.argv[1])

#    all_saves = list(Path('session_4da05e87').glob('*.zip'))
#    selected_saves = [Path('session_4da05e87/init')] + all_saves[:10] + all_saves[10:120:5] + all_saves[120:420:10]
#    len(selected_saves)

#    for idx, save in enumerate(selected_saves):

