#!/bin/bash
for i in $(seq 1 19); do
    sbatch run_sbatch.sh ${i}31415
done
