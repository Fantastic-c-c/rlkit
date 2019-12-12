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
            self.goals = (np.pi + np.random.uniform(low=-np.pi / 3, high=np.pi / 3, size=(n_tasks,))).tolist()
        else:
            self.goals = [4.05348225976433, 3.0027595301945875, 3.672103499187853, 2.604160714509009, 3.9190760751537157, 2.67673452523618, 2.8075936237925787, 3.8781358460543327, 2.5002439422059877, 
            2.870680469029778, 3.6338332538150313, 2.2865661785793785, 3.313270394077212, 4.077422222067832, 3.243763743452335, 4.096120764635373, 2.303972182267003, 4.018452341694367, 
            3.957028748768969, 3.9590421409028345, 3.2247066098634023, 2.2712006213049545, 3.463491943393195, 3.079749259182855, 3.949866732469301, 3.7648670230124415, 2.339450870589328,
            2.402419843037275, 2.384067786238313, 3.4882450975838726, 3.543568290399147, 2.328435057935333, 2.578644350398771, 2.5560015826465614, 3.441739283973496, 3.083268425629527, 
            2.712910496444433, 2.2380670322282725, 2.83802164774973, 2.1498965922609075]

        self.goals = self.goals[:n_tasks]
        print(np.array(self.goals).tolist())
        assert len(self.goals) == n_tasks, "The number of goals should equal the number of tasks"
        self.reset_task(0)  # should this be random?

        self._desired_claw_pos = np.array([0, -0.4, 0.4] * 3)
        print("Desired pose: " + str(self._desired_claw_pos))

    def get_all_task_idx(self):
        return range(len(self.goals))

    def get_goal(self):
        return self._target_object_pos

    def reset_task(self, idx):
        self._set_target_object_pos(self.goals[idx])
        self.reset()

    def get_tasks(self):
        return self.goals
