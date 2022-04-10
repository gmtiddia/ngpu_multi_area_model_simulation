#!/bin/bash -x
#SBATCH --account=icei-hbp-2020-0007
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH --time=20:00:00
#SBATCH --partition=gpus
#SBATCH --output=/p/scratch/icei-hbp-2020-0007/mam_ngpu1/distributions/test_fr_out.%j
#SBATCH --error=/p/scratch/icei-hbp-2020-0007/mam_ngpu1/distributions/test_fr_err.%j
# *** start of job script ***
# Note: The current working directory at this point is
# the directory where sbatch was executed.

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
srun python firing_rate.py
