from gymnasium.envs.registration import register
from gymnasium.wrappers import AutoResetWrapper

register(
     id="pokemonred_env/PokeRed-v0",
     entry_point="pokemonred_env.envs:PokeRedEnv",
     max_episode_steps=20480,
     additional_wrappers=(AutoResetWrapper),
)