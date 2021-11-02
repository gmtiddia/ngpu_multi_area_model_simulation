#!/bin/bash
for i in $(seq 2 10); do
    sbatch run_sbatch.sh ${i}12345
done
