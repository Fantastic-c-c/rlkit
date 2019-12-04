from robel.dclaw.pose import BaseDClawPose

from . import register_env


@register_env('dclaw-pose-sim')
class DClawPoseEnv(BaseDClawPose):
    def __init__(self,
                 randomize_tasks=True,
                 n_tasks=5,
                 **kwargs
                 ):
        BaseDClawPose.__init__(self, **kwargs)
        if randomize_tasks:
            self.goals = [self._make_random_pose() for _ in range(n_tasks)]
        else:
            self.goals = [
                [0.33216093, -0.34648382, 1.05373163, 0.3457113, -0.75016362, 0.01385098,
                 0.04234343, -0.35791852, -0.56737231],
                [0.21918838, -0.08618355, -0.55813989, -0.46893439, -0.77759893, 0.71962147,
                 0.23718581, -0.93691183, -1.3277795]
            ]
        print(self.goals)
        assert len(self.goals) == n_tasks, "The number of goals should equal the number of tasks"
        self.reset_task(0)  # should this be random?

    def get_all_task_idx(self):
        return range(len(self.goals))

    def get_goal(self):
        return self._desired_pos

    def reset_task(self, idx):
        self._desired_pos = self.goals[idx]
        self.reset()

    def get_tasks(self):
        return self.goals
