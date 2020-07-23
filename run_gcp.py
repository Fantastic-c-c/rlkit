import doodad
from doodad.launch.launch_api import run_python, run_command
from doodad import mode, mount

import argparse

"""
Run PEARL with Docker and doodad on GCP
"""


"""
gsutil rsync -r -x ".*\.data*|.*\.index" gs://meld/jan26_batchmode_reacher/logs/my_gcp_logs/gym .

gsutil cp -r gs://meld/testing/logs/my_gcp_logs/gym/**events\.* .
^ use this one to avoid replay buffer checkpoints, etc.

"""

##########################################################################################
##########################################################################################


def run_job(args):

  # Log will be created in GCS under: bucket_name/gcp_log_path/doodad...
  # Results will be stored in GCS under: bucket_name/gcp_log_path/logs/my_logs/root_dir/...

  # don't copy these files to remote host because permissions
  ignore_exts = ('.pyc', '.log', '.git', '.mp4', '.viminfo', '.bash_history', '.python_history')

  project_name = 'mslac-anusha-kate-tony'
  bucket_name = 'mslac'
  gcp_image = 'last-chance-texaco'

  # in this case, we must manually specify the code mount, because doodad cannot infer it automatically from a general bash command
  code_mnt = mount.MountLocal(local_dir='/home/rakelly/rlkit', mount_point='/root/code', output=False, filter_ext=ignore_exts, pythonpath=True)

  # output will be stored in Google Cloud Storage under bucket_name/gcp_log_path/logs/my_gcp_logs
  output_mnt = mount.MountGCP(gcp_path='my_gcp_logs', mount_point='/root/rlkit/output', output=True)
  remote = mode.GCPMode(gcp_project=project_name, \
          gcp_bucket=bucket_name, \
          gcp_log_path=args.gcp_log_path, \
          gcp_image=gcp_image, \
          gcp_image_project=project_name, \
          terminate_on_end=True, \
          preemptible=False, \
          zone=args.zone, \
          instance_type='n1-standard-32', \
          gpu_kwargs=dict(gpu_model='nvidia-tesla-p100', num_gpu=2)) # do not change this, our quota is only for this type of gpu, and we need 2 gpus

  command = 'python /root/code/_home_rakelly_rlkit/launch_experiment.py ./configs/' + args.config
  # launch the job! the docker-image will be automatically downloaded from the Docker Hub
  run_command(command, mode=remote, mounts=[code_mnt, output_mnt], verbose=True, docker_image='rakelly/pearl-mj200')


##########################################################################################
##########################################################################################

def main():

  #####################
  # training args
  #####################

  parser = argparse.ArgumentParser(
      # Show default value in the help doc.
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,)

  parser.add_argument('--config', type=str, default='')
  parser.add_argument('--gcp_log_path', type=str, default='pearl')
  # specify where to run on GCP
  parser.add_argument('--zone', type=str, default='us-west1-b')

  args = parser.parse_args()
  run_job(args)

if __name__ == '__main__':
  main()
