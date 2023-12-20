from pathlib import Path
import uuid
import pokemonred_env
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from pyboy.utils import WindowEvent

if __name__ == '__main__':

    sess_path = Path(f'session_{str(uuid.uuid4())[:8]}')
    ep_length = 2**23

    num_cpu = 1 #64 #46  # Also sets the number of episodes per training iteration
    env = make_vec_env(env_id="PokeRed-v0",
                       n_envs=num_cpu,
                       seed=None,
                       vec_env_cls=SubprocVecEnv,
                       env_kwargs=dict(headless=False,
                                       pyboy_bequiet='--quiet' in sys.argv,
                                       gb_path=Path("../PokemonRed.gb"),
                                       init_state=Path('../has_pokedex_nballs.state'),
                                       session_path=sess_path,
                                       max_steps=ep_length,
                                       )
                       )

    file_name = 'session_4da05e87_main_good/poke_439746560_steps'

    print('\nloading checkpoint')
    model = PPO.load(path=file_name,
                     env=env,
                     custom_objects={'lr_schedule': 0,
                                     'clip_range': 0,
                                     },
                     force_reset=True,
                     )

    #keyboard.on_press_key("M", toggle_agent)
    obs, info = env.reset()
    while True:
        action = WindowEvent.PASS
        try:
            with open("agent_enabled.txt", "r") as f:
                agent_enabled = f.readlines()[0].startswith("yes")
        except:
            agent_enabled = False
        if agent_enabled:
            action, _states = model.predict(obs,
                                            deterministic=False,
                                            )
        obs, rewards, terminated, truncated, info = env.step(action)
        env.render()
        if truncated:
            break
    env.close()
