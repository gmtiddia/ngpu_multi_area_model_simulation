#!/bin/bash -x
#SBATCH --account=ACCOUNTNAME
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:10:00
#SBATCH --partition=gpus
#SBATCH --output=PATH/test_mam_eval_time_out.%j
#SBATCH --error=PATH/test_mam_eval_time_err.%j
# *** start of job script ***
# Note: The current working directory at this point is
# the directory where sbatch was executed.

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

if [ "$#" -ne 1 ]; then
    seed=12345
else
    seed=$1
fi

cat run_eval_time.templ | sed "s/__seed__/$seed/g" > run_eval_time.py
srun python3 run_eval_time.py
