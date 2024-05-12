#!/bin/bash -x
#SBATCH --account=None
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH --time=10:00:00
#SBATCH --partition=None
#SBATCH --output=/path/test_mam_eval_time_out.%j
#SBATCH --error=/path/test_mam_eval_time_err.%j
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
srun -n 32 python3 run_eval_time.py
