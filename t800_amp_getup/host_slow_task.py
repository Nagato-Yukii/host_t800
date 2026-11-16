"""HoST warm-resume curriculum: 20% at 3000, minus 5 points every 500 iterations."""
from .host_task import T800HoSTTask, assistance_at_iteration


def slow_assistance_at_iteration(iteration: int) -> float:
    if iteration < 3000:
        return assistance_at_iteration(iteration)
    return max(0, 4 - (int(iteration) - 3000) // 500) / 20


class T800HoSTSlowTask(T800HoSTTask):
    curriculum_version = 'host_3000_20pct_minus5_each500_v1'

    def set_training_iteration(self, iteration):
        super().set_training_iteration(iteration)
        self.assistance_fraction = slow_assistance_at_iteration(iteration)
