from gymnasium.envs.registration import register

register(
     id="PokeRed-v0",
     entry_point="pokemonred_env.envs:PokeRedEnv",
     max_episode_steps=20480,
)

hyperparams = {"PokeRed-v0": dict(frame_stack=3,
                                  policy="CnnPolicy",
                                  n_envs=4,
                                  n_steps=20480,
                                  n_epochs=4,
                                  batch_size=128,
                                  n_timesteps=0.0000001,
                                  learning_rate=0.0003,
                                  clip_range=0.2,
                                  vf_coef=0.5,
                                  ent_coef=0.0,
                                  )
               }
