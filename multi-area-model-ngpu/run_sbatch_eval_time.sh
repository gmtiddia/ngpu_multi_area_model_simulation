#!/bin/bash -x
#SBATCH --account=icei-hbp-2020-0007
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --partition=gpus
#SBATCH --output=/p/project1/icei-hbp-2020-0007/tiddia/mam_stage2024/mam_mpi_comm_benchmarking_2024/multi-area-model-ngpu/logfiles/test_mam_eval_time_out.%j
#SBATCH --error=/p/project1/icei-hbp-2020-0007/tiddia/mam_stage2024/mam_mpi_comm_benchmarking_2024/multi-area-model-ngpu/logfiles/test_mam_eval_time_err.%j
# *** start of job script ***
# Note: The current working directory at this point is
# the directory where sbatch was executed.

if [ "$#" -ne 1 ]; then
    seed=12345
else
    seed=$1
fi

cat run_eval_time.templ | sed "s/__seed__/$seed/g" > run_eval_time.py
srun python3 run_eval_time.py
