#!/usr/bin/bash

#SBATCH --job-name=pii-ddp-transformer
#SBATCH --output=slurm_output/%J.out
#SBATCH --error=slurm_output/%J.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00

srun --unbuffered --output slurm_output/%j_%2t.out --error slurm_output/%j_%2t.err ddp_job.sh $1
