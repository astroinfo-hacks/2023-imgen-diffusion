#!/bin/bash
#SBATCH --job-name=stats
#SBATCH --output=stats-%J.out
#SBATCH --error=stats-%J.err
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=skylake
#SBATCH --mem=10GB

# List the available gpu cards

cd ~/users4astroinfo2023/jimenez/hackathon/2023-imgen-diffusion

module load userspace/all

eval "$(micromamba shell hook --shell bash)"

micromamba activate tpml_tuesday

python stats/check_header.py
PWD=$(pwd)
echo "Procesing from directory $PWD"


