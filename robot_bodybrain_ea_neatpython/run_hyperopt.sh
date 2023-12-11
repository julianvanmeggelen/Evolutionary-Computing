#!/bin/bash -l


#SBATCH --job-name=robot_bodybrain_ea

#SBATCH --nodes=1

#SBATCH --time=0-08:00:00

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
python hyperopt.py --timeout 4200

# Specify the directory you want to copy
source_directory="./results"

# Specify the destination directory (in this case, the SLURM job's working directory)
destination_directory="$SLURM_SUBMIT_DIR/results_$SLURM_JOB_ID"

# Create the destination directory if it doesn't exist
mkdir -p $destination_directory

# Copy the entire directory to the destination
cp -r $source_directory/* $destination_directory/

# Finish the script
exit 0