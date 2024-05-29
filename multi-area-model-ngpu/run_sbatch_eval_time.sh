#!/bin/bash -x
#SBATCH --account=jinb33
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:10:00
#SBATCH --partition=dc-gpu
#SBATCH --gres=gpu:4
#SBATCH --output=/p/project/cjinb33/tiddia1/mam_2024/multi-area-model-ngpu/logfiles/test_mam_eval_time_out.%j
#SBATCH --error=/p/project/cjinb33/tiddia1/mam_2024/multi-area-model-ngpu/logfiles/test_mam_eval_time_err.%j
# *** start of job script ***
# Note: The current working directory at this point is
# the directory where sbatch was executed.

#source ~/mam_debugging/config.sh

#export OMP_NUM_THREADS=32

if [ "$#" -ne 1 ]; then
    seed=12345
else
    seed=$1
fi

cat run_eval_time.templ | sed "s/__seed__/$seed/g" > run_eval_time.py
srun python3 run_eval_time.py
