#!/bin/bash
for i in $(seq 1 10); do
    sbatch run_sbatch.sh ${i}12345
done
