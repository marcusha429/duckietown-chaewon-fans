from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium.wrappers import RecordVideo
import pyglet

# Hide pyglet window
window = pyglet.window.Window(visible=False)


class VideoRecordingCallback(BaseCallback):
    def __init__(
        self,
        video_folder: str = "videos",
        video_length: int = 200,
        video_prefix: str = "evaluation",
        save_freq: int = 1024,
        verbose=0,
    ):
        super(VideoRecordingCallback, self).__init__(verbose)
        self.video_folder = video_folder
        self.video_length = video_length
        self.video_prefix = video_prefix
        self.save_freq = save_freq
        self.steps_since_last_video = 0

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
                self.env,
                video_folder=self.video_folder,
                video_length=self.video_length,
                name_prefix=self.video_prefix,
                episode_trigger=lambda x: True,  # Record every episode
            )
            print(f"Recording video at step {self.n_calls}!")

        return True
