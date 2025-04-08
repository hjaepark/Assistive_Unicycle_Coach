import gymnasium as gym
import numpy as np
import dearpygui.dearpygui as dpg
import time

# Initialize environment
env = gym.make("HalfCheetah-v5", render_mode="rgb_array")
observation, info = env.reset()

# Initialize GUI
dpg.create_context()
dpg.create_viewport(title="RL Visualization", width=1200, height=800)

# Setup texture for environment rendering
with dpg.texture_registry():
    dpg.add_raw_texture(
        width=480,
        height=480,
        default_value=np.zeros((480, 480, 3), dtype=np.float32),
        format=dpg.mvFormat_Float_rgb,
        tag="env_texture",
    )

# Create windows and plots
with dpg.window(label="Environment", width=820, height=620):
    dpg.add_image("env_texture")

with dpg.window(label="Performance", pos=(830, 10), width=360, height=300):
    with dpg.plot(label="Reward", height=250):
        dpg.add_plot_axis(dpg.mvXAxis, label="steps")
        dpg.add_plot_axis(dpg.mvYAxis, label="reward")
        dpg.add_line_series([], [], parent=dpg.last_item(), tag="reward_plot")

with dpg.window(label="Actions", pos=(830, 320), width=360, height=300):
    with dpg.plot(label="Actions", height=250):
        dpg.add_plot_axis(dpg.mvXAxis, label="steps")
        dpg.add_plot_axis(dpg.mvYAxis, label="action values")
        dpg.add_line_series([], [], parent=dpg.last_item(), tag="action_plot")

# Setup data storage
rewards = []
actions = []
x_data = []

# Setup and show GUI
dpg.setup_dearpygui()
dpg.show_viewport()

# Main loop - combines RL training with GUI updates
step_count = 0
episode_count = 0
episode_reward = 0
training_active = True

try:
    while dpg.is_dearpygui_running():
        # Run RL step if training is active
        if training_active:
            # Replace with your agent's policy
            action = env.action_space.sample()

            # Step environment
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            # Store data for visualization
            step_count += 1
            x_data.append(step_count)
            rewards.append(episode_reward)
            actions.append(action)

            # Update environment rendering
            render_frame = env.render()
            render_frame = np.ascontiguousarray(render_frame / 255, dtype=np.float32)
            dpg.set_value("env_texture", render_frame)

            # Update plots
            dpg.set_value("reward_plot", [x_data, rewards])

            # Plot first dimension of actions
            action_values = [a[0] for a in actions[-100:]]  # Show just recent actions
            action_steps = x_data[-len(action_values) :]
            dpg.set_value("action_plot", [action_steps, action_values])

            # Reset if episode is done
            if terminated or truncated:
                observation, info = env.reset()
                episode_count += 1
                episode_reward = 0
                print(f"Episode {episode_count} completed")

                # Optional: stop after certain number of episodes
                if episode_count >= 10:
                    training_active = False

        # Render GUI frame
        dpg.render_dearpygui_frame()

        # Control simulation speed
        if training_active:
            time.sleep(0.01)  # Adjust to control simulation speed

except KeyboardInterrupt:
    pass
finally:
    dpg.destroy_context()
