"""HoST get-up task with constant assistance at 80% of body weight."""

from .host_task import T800HoSTTask


class T800Test80Task(T800HoSTTask):
    curriculum_version = "fixed_weight_80_constant_v1"

    def set_training_iteration(self, iteration):
        super().set_training_iteration(iteration)
        self.assistance_fraction = 0.8
