# import pyglet
# window = pyglet.window.Window(visible=False)

from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import RecordVideo
from .env import make_gym_env


class VideoRecordingCallback(BaseCallback):
    def __init__(
        self,
        simulator_kwargs={},
        video_folder: str = "videos",
        video_length: int = 1000,
        video_prefix: str = "evaluation",
        save_freq: int = 1024,
        verbose=0,
    ):
        super(VideoRecordingCallback, self).__init__(verbose)
        self.simulator_kwargs = simulator_kwargs
        self.video_folder = video_folder
        self.video_length = video_length
        self.video_prefix = video_prefix
        self.save_freq = save_freq
        self.steps_since_last_video = 0
        print("video callback called!")

    def _on_step(self) -> bool:
        """
        This method will be called at every step during training.
        It will trigger video recording every `self.save_freq` steps.
        """
        # Increase the count of steps
        self.steps_since_last_video += 1

        # Check if we've reached the interval for recording a video
        if self.steps_since_last_video >= self.save_freq:
            self.steps_since_last_video = 0  # Reset the counter

            # Record video for this episode
            self.env = RecordVideo(
                make_gym_env(self.simulator_kwargs),
                video_folder=self.video_folder,
                video_length=self.video_length,
                name_prefix=self.video_prefix + str(self.num_timesteps),
                episode_trigger=lambda x: True,  # Record every episode
            )

            # Make sure the PPO model is used to take actions during video recording
            print(f"Recording video at step {self.n_calls}!")

            # Simulate an episode with the PPO model controlling the agent
            obs, info = self.env.reset()
            episode_over = False
            while not episode_over:
                action, _states = self.model.predict(
                    obs
                )  # Use PPO model to predict action
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_over = terminated or truncated

            # Close the environment after the episode
            self.env.close()

        return True
