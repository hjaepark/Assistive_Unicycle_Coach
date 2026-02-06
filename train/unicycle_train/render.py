# train/unicycle_train/render.py

import sys
from pathlib import Path
import numpy as np

import gymnasium as gym
from stable_baselines3 import SAC

import envs.unicycle

# Identical camera config to user study
DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": -1,
    "lookat": np.array((2.5, 0.0, 0.0)),
    "distance": 5,
    "azimuth": 45.0,
    "elevation": -15.0,
}

def main() -> None:
    if len(sys.argv) < 2:
        sys.stderr.write("Usage: python render.py /path/to/model[.zip]\n")
        sys.exit(1)

    raw_path = Path(sys.argv[1]).expanduser()
    model_path = raw_path.resolve()

    if not model_path.exists() and model_path.suffix == "" and model_path.with_suffix(".zip").exists():
        model_path = model_path.with_suffix(".zip")

    if not model_path.exists():
        sys.stderr.write(f"Model file not found: {raw_path}\n")
        sys.exit(1)

    env = gym.make("UnicyclePendulumPD-v0", render_mode="human", default_camera_config=DEFAULT_CAMERA_CONFIG)

    model = SAC.load(model_path.as_posix(), env=env, device="auto")

    try:
        while True:
            obs, info = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()