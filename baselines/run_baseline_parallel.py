from os.path import exists
from pathlib import Path
import uuid
import sys
import pokemonred_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback


if __name__ == '__main__':
    ep_length = 2048 * 8
    sess_path = Path(f'session_{str(uuid.uuid4())[:8]}')
    num_cpu = 44 #64 #46  # Also sets the number of episodes per training iteration
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
    #env_checker.check_env(env)
    learn_steps = 40
    file_name = 'session_e41c9eff/poke_38207488_steps' #'session_e41c9eff/poke_250871808_steps'

    #'session_bfdca25a/poke_42532864_steps' #'session_d3033abb/poke_47579136_steps' #'session_a17cc1f5/poke_33546240_steps' #'session_e4bdca71/poke_8945664_steps' #'session_eb21989e/poke_40255488_steps' #'session_80f70ab4/poke_58982400_steps'
    if exists(file_name + '.zip'):
        print('\nloading checkpoint')
        model = PPO.load(path=file_name,
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
        model = PPO(policy='CnnPolicy',
                    env=env,
                    verbose=1,
                    n_steps=ep_length,
                    batch_size=512,
                    n_epochs=1,
                    gamma=0.999,
                    )

    for i in range(learn_steps):
        model.learn(total_timesteps=(ep_length)*num_cpu*1000,
                    callback=checkpoint_callback)
