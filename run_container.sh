#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --reservation=fri
#SBATCH --time=01:30:00
#SBATCH --output=logs/nlp-benchmark-DP-ner-%J.out
#SBATCH --error=logs/nlp-benchmark-DP-%J.err
#SBATCH --job-name="NLP DP Benchmark"

srun singularity exec --nv ../containers/nlp_benchmark.sif python \
    "nli_benchmark.py"