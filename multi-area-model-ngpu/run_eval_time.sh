#!/bin/bash
for i in $(seq 1 10); do
    sbatch run_sbatch_eval_time.sh ${i}12345
done
