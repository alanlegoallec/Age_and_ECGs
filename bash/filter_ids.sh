#!/bin/bash
#SBATCH -p short
#SBATCH --open-mode=truncate
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=alanlegoallec@g.harvard.edu

module load gcc/6.2.0
module load python/3.6.0
source ~/python_3.6.0/bin/activate
python ../scripts/TS03_filter_ids.py $1 $2

