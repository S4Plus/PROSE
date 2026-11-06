import abc


class Subscriber(abc.ABC):
    """Subscribes to events in the runtime.

    Subscribers can be used to save data, visualize, etc.
    """

    @abc.abstractmethod
    def on_episode_start(self) -> None:
        """Called when an episode starts."""

    @abc.abstractmethod
    def on_step(self, observation: dict, action: dict) -> None:
        """Append a step to the episode."""

    @abc.abstractmethod
    def on_episode_end(self, metrics: dict | None = None) -> None:
        """Called when an episode ends.

        Args:
            metrics: Optional per-episode metrics such as success flags.
        """
