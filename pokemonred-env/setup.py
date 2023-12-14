from setuptools import setup

setup(
    name="pokemonred_env",
    version="0.0.1",
    install_requires=[
                      "matplotlib==3.7.1",
                      "hnswlib",
                      "numpy==1.25.0",
                      "einops==0.6.1",
                      "matplotlib==3.7.1",
                      "scikit-image==0.21.0",
                      "pyboy==1.6.9",
                      "mediapy @ git+https://github.com/PWhiddy/mediapy.git@45101800d4f6adeffe814cad93de1db67c1bd614",
                      "pandas==2.0.2",
                      "gymnasium>=0.26.0",
                      ],
)