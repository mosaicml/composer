import random
import math
from abc import abstractmethod

from composer.core.time import TimeUnit

def get_max_duration_as_steps(state):
    """Returns the max duration of the state in steps

    Args:
        state (:class:`State`): The current state.
    Returns:
        int: Number of steps will run
    """
    if state.max_duration.unit == TimeUnit.EPOCH:
        max_dur = state.max_duration.value * state.steps_per_epoch
    elif state.max_duration.unit == TimeUnit.BATCH:
        # in correct form
        max_dur = state.max_duration.value
    else:
        raise ValueError(f"Can't infer number of steps from max duration unit {state.max_duration.unit}")

    return max_dur

class SAMInterval():
    """Abstract class for determinining if SAM should be run for each interval
    or not. Each implementation of this should overrride run_check."""

    def __init__(self, T, **kwargs):
        """
        Args:
            T (int): The total number of steps that will be taken
        """
        self.T = T
        if self.T is None and self.requires_max_steps:
            raise ValueError("This SAMInterval algorithm requires that the number of max_steps not be None")
        elif self.T < 0 and self.requires_max_steps:
            raise ValueError("This SAMInterval algorithm requires that the number of max_steps be greater than 0")

    @abstractmethod
    def run_check(self, t):
        """Checks if should run SAM for this iteration. Can either use a
        self-maintained global counter (t) or the tracker in State (or both).

        Args:
            t (int): The global step
        Returns:
            bool: True if SAM should be run
        """

    def requires_max_steps(self):
        """If the interval checker requires knowledge of the number of max
        steps. Default is False."""
        return False

class SAM_FixedInterval(SAMInterval):
    """Will run SAM at a fixed interval (once every t steps)."""

    def __init__(self, T, interval_num, **kwargs):
        super(SAM_FixedInterval, self).__init__(T)
        self.interval_num = interval_num

    def run_check(self, t):
        return (t + 1) %  self.interval_num == 0


class SAM_ConstantRandom(SAMInterval):
    """Will run SAM with a constant probability.
    (`Zhao et al, 2020 <https://arxiv.org/abs/2203.09962>`_)."""

    def __init__(self, T, prob, **kwargs):
        super(SAM_ConstantRandom, self).__init__(T)
        self.prob = prob

    def run_check(self, t):
        return random.random() < self.prob

class SAM_Piecewise(SAMInterval):
    """Will run SAM with piecewise probability
    (`Zhao et al, 2020 <https://arxiv.org/abs/2203.09962>`_)."""

    def __init__(self, T, prob, boundary_frac, **kwargs):
        super(SAM_Piecewise, self).__init__(T)
        self.prob = prob
        self.boundary_point = int(boundary_frac * self.T)

    def run_check(self, t):
        if t <= self.boundary_point:
            return random.random() < self.prob
        else:
            return random.random() < (1 - self.prob)

    def requires_max_steps(self):
        return True


class SAM_CosProb(SAMInterval):
    """Will run SAM with cosine scheduling function
    (`Zhao et al, 2020 <https://arxiv.org/abs/2203.09962>`_)."""

    def run_check(self, t):
        return random.random() < (0.5 + 0.5 * math.cos( t / self.T * math.pi))

    def requires_max_steps(self):
        return True
