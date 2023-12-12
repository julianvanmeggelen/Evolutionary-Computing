#!/bin/bash -l


#SBATCH --job-name=robot_bodybrain_ea

#SBATCH --nodes=1

#SBATCH --time=0-5:00:00

#SBATCH --partition defq

# How much memory you need.
# --mem will define memory per node and
# --mem-per-cpu will define memory per CPU/core. Choose one of those.
##SBATCH --mem-per-cpu=1500MB
##SBATCH --mem=5GB    # this one is not in effect, due to the double hash

# Turn on mail notification. There are many possible self-explaining values:
# NONE, BEGIN, END, FAIL, ALL (including all aforementioned)
# For more values, check "man sbatch"
##SBATCH --mail-type=END,FAIL

source $HOME/.bashrc
conda activate EC
NRUNS=20 NGEN=100 python main.py

mv ./meanmax.png $SLURM_SUBMIT_DIR/result.png


# Finish the script
exit 0