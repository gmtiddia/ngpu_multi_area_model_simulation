# Absolute path of repository
base_path = '/home/tiddia/mam_debugging/multi-area-model-ngpu'
# Place to store simulations
data_path = '/home/tiddia/mam_debugging/multi-area-model-ngpu/simulations'
# Template for job scripts
jobscript_template = '''#!/bin/bash -x
#SBATCH --account=icei-hbp-2020-0007
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH --time=02:00:00
#SBATCH --partition=gpus
#SBATCH --output=path/test_mam_out.%j
#SBATCH --error=path/test_mam_err.%j
# *** start of job script ***
# Note: The current working directory at this point is
# the directory where sbatch was executed.

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

srun -n 32 python3 run_simulation.py
#srun -n 32 python {base_path}/run_simulation.py {label} {network_label}
'''

# Command to submit jobs on the local cluster
submit_cmd = 'sbatch'
