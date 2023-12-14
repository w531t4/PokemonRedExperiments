from gymnasium.envs.registration import register

register(
     id="pokemonred_env/PokeRed-v0",
     entry_point="pokemonred_env.envs:PokeRedEnv",
     max_episode_steps=20480,
     autoreset=True,
)