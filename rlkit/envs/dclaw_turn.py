from robel.dclaw.turn import BaseDClawTurn

import numpy as np
from . import register_env


@register_env('dclaw-turn')
class DClawTurnEnv(BaseDClawTurn):
    def __init__(self,
                 randomize_tasks=True,
                 n_tasks=5,
                 **kwargs
                 ):
        BaseDClawTurn.__init__(self, **kwargs)
        print("RANDOMIZE TASKS? " + str(randomize_tasks))
        if randomize_tasks:
            self.goals = (np.pi + np.random.uniform(low=-np.pi / 3, high=np.pi / 3, size=(2,))).tolist()
        else:
            self.goals = [3.1963582825190135, 4.152947458951554]
        self.goals = self.goals[:n_tasks]
        print(np.array(self.goals).tolist())
        assert len(self.goals) == n_tasks, "The number of goals should equal the number of tasks"
        self.reset_task(0)  # should this be random?

    def get_all_task_idx(self):
        return range(len(self.goals))

    def get_goal(self):
        return self._target_object_pos

    def reset_task(self, idx):
        self._set_target_object_pos(self.goals[idx])
        self.reset()

    def get_tasks(self):
        return self.goals
