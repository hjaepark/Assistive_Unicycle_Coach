from setuptools import setup, find_packages

setup(
    name="Assistive_Unicycle_Coach",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'gymnasium',
        'stable-baselines3',
        'numpy',
        'scipy',
        'mujoco'
    ]
)