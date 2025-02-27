from stable_baselines3.common.callbacks import BaseCallback


# Create a custom callback to render the environment
class RenderCallback(BaseCallback):
    def __init__(self, render_freq: int = 1, verbose: int = 1):
        super(RenderCallback, self).__init__()
        self.render_freq = render_freq
        self.verbose = verbose

    def _on_step(self) -> bool:
        """
        This method is called at every step of training.
        We will render the environment based on `render_freq`.
        """
        if self.n_calls % self.render_freq == 0:  # Render every `render_freq` steps
            # Access the environment and render it
            env = self.locals["env"]
            env.render(mode="human")

        return True  # Always return True to continue training
