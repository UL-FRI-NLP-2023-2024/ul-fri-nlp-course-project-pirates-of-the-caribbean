#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=01:30:00
#SBATCH --output=logs/nlp-benchmark-%J.out
#SBATCH --error=logs/nlp-benchmark-%J.err
#SBATCH --job-name="NLP DP Benchmark"

echo "$2"
singularity exec --nv ../containers/nlp_benchmark.sif python3 "$1" "$2"
