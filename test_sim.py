import numpy as np
import time
import os
import shutil

from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv

env = NormalizedBoxEnv(ENVS['torque-peg-insert'](environment_kwargs={'control_timestep':.05}))

all_dists = {}
video_frames = []
for i in range(7):
    for j in range(2):
        dists = []
        action = np.zeros(7)
        action[i] = 2 * j - 1  # action is either -1 or 1
        true = 10
        for reps in range(10):
            env.reset()
            # measure starting position
            #init_pos = env.task.get_observation(env.physics)['ee_position']
            init_pos = env.task.get_observation(env.physics)['angles'][i]
            for s in range(5):
                env.step(action)
                video_frames.append(env.get_image())
            time.sleep(1)
            #final_pos = env.task.get_observation(env.physics)['ee_position']
            final_pos = env.task.get_observation(env.physics)['angles'][i]
            dists.append(np.linalg.norm(init_pos - final_pos))
        key_str = "joint{}_direction_{}_true{}".format(i, j, true)
        all_dists[key_str] = np.mean(np.asarray(dists))


# save frames to file temporarily
temp_dir = 'temp'
os.makedirs(temp_dir, exist_ok=True)
for i, frm in enumerate(video_frames):
    frm.save(os.path.join(temp_dir, '%06d.jpg' % i))

video_filename='analysis_video.mp4'
# run ffmpeg to make the video
os.system('ffmpeg -r {} -i {}/%06d.jpg -vcodec mpeg4 {}'.format(int(env.frame_rate), temp_dir, video_filename))
# delete the frames
shutil.rmtree(temp_dir)

print("All dists: " + str(all_dists))
