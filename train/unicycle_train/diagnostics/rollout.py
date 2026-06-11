# train/unicycle_train/diagnostics/rollout.py
import numpy as np


def rollout(policy_fn, env, seed):
    # policy_fn(obs, prev_reward) -> (action, step_info dict)
    obs, info = env.reset(seed=seed)
    done = False
    ep_return = 0.0
    length = 0
    prev_reward = 0.0
    extras = []

    while not done:
        action, step_info = policy_fn(obs, prev_reward)
        obs, reward, terminated, truncated, info = env.step(action)
        prev_reward = reward
        ep_return += reward
        length += 1
        extras.append(step_info)
        done = terminated or truncated

    out = {
        "return": ep_return,
        "goal_reached": bool(info["goal_reached"]),
        "is_fallen": bool(info["is_fallen"]),
        "length": length,
        "final_x": float(info["x_position"]),
    }
    if extras[0]:
        for k in extras[0]:
            out[k] = float(np.mean([e[k] for e in extras]))
    return out