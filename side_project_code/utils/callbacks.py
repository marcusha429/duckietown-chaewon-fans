import os
from gym.wrappers import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from env import make_gym_env


# The VideoRecorderCallback uses your make_gym_env function to create a new environment
# instance for recording a single episode video. This ensures that the training environment
# (especially if it is vectorized) is not affected by video recording.
class VideoRecorderCallback(BaseCallback):
    """
    Callback for periodically recording a video of the agent.

    This callback creates a new (non-vectorized) environment instance using your
    `make_gym_env(simulator_kwargs)` function, wraps it with Gym's Monitor to capture a video,
    and runs one episode using the current policy.

    Args:
        video_folder (str): Directory where video files will be saved.
        video_length (int): Maximum number of steps to record per video.
        trigger_freq (int): How many calls to _on_step before triggering a recording.
        simulator_kwargs (dict): Keyword arguments to pass to your environment creator.
        verbose (int): Verbosity level.
    """

    def __init__(
        self,
        video_folder: str,
        video_length: int = 1000,
        trigger_freq: int = 10000,
        simulator_kwargs=None,
        verbose=1,
    ):
        super(VideoRecorderCallback, self).__init__(verbose)
        self.video_folder = video_folder
        self.video_length = video_length
        self.trigger_freq = trigger_freq
        self.simulator_kwargs = simulator_kwargs if simulator_kwargs is not None else {}
        self.video_count = 0

    def _init_callback(self) -> None:
        # Create the directory for saving videos if it does not exist.
        if not os.path.exists(self.video_folder):
            os.makedirs(self.video_folder)

    def _on_step(self) -> bool:
        # Check if it's time to record a video based on the trigger frequency.
        if self.n_calls % self.trigger_freq == 0:
            self._record_video()
        return True

    def _record_video(self):
        """
        Records a video by running one episode using the current model's policy.
        """
        # Create a fresh environment for video recording.
        # This uses your utility function, so the environment configuration stays consistent.
        env = make_gym_env(self.simulator_kwargs)

        # Wrap the environment with Monitor to handle video recording.
        # The lambda always returns True so that the video is recorded regardless of the episode number.
        env = Monitor(
            env,
            directory=self.video_folder,
            video_callable=lambda episode_id: True,
            force=True,
        )

        obs = env.reset()
        done = False
        step_count = 0

        # Run an episode up to video_length steps (or until the episode ends)
        while not done and step_count < self.video_length:
            # Use the current policy in a deterministic way
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            step_count += 1

        env.close()
        self.video_count += 1
        if self.verbose > 0:
            print(f"Recorded video {self.video_count} at step {self.n_calls}.")


# When starting training, simply pass the callback (or add it to a list along with others)
model.learn(
    total_timesteps=total_timesteps,
    tb_log_name=model_name,
    callback=[checkpoint_callback, video_callback],
    reset_num_timesteps=False,
)
