from pathlib import Path
import pandas as pd
import numpy as np
import sys
import pokemonred_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

def run_recorded_actions_on_emulator_and_save_video(sess_id,
                                                    instance_id,
                                                    run_index,
                                                    ):
    sess_path = Path(f'session_{sess_id}')
    tdf = pd.read_csv(f"session_{sess_id}/agent_stats_{instance_id}.csv.gz",
                      compression='gzip',
                      )
    tdf = tdf[tdf['map'] != 'map'] # remove unused
    action_arrays = np.array_split(tdf, np.array((tdf["step"].astype(int) == 0).sum()))
    action_list = [int(x) for x in list(action_arrays[run_index]["last_action"])]
    max_steps = len(action_list) - 1
    env = make_vec_env(env_id="PokeRed-v0",
                       n_envs=1,
                       seed=None,
                       vec_env_cls=SubprocVecEnv,
                       env_kwargs=dict(pyboy_bequiet='--quiet' in sys.argv,
                                       gb_path=Path("../PokemonRed.gb"),
                                       init_state=Path('../has_pokedex_nballs.state'),
                                       session_path=sess_path,
                                       max_steps=max_steps,
                                       print_rewards=False,
                                       save_video=True,
                                       fast_video=False,
                                       instance_id=f'{instance_id}_recorded',
                                       )
                       )

    env.set_attr(attr_name="reset_count",
                 value=run_index,
                 indicies=[0],
                 )

    obs = env.reset()
    for action in action_list:
        obs, rewards, term, trunc, info = env.step(action)
        env.render()
