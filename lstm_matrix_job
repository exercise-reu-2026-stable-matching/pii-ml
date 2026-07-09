#!/usr/bin/bash

#SBATCH --job-name=pii-state-ml
#SBATCH --output=slurm_output/%A_%4a.out
#SBATCH --error=output/%A_%4a.err
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --array=0-0

echo "Running job on ${SLURMD_NODENAME}"
cd matrix_models

# Create TMP directory
mkdir -p /home/reujpasternak/tmp
export TMPDIR=/home/reujpasternak/tmp/

# Obtain the current location
HDIR=$(pwd)
INPUT_DIR=matrix_data
PLOT_DIR=lstm_matrix_plots
PLOT_DATA_DIR=lstm_matrix_plot_data

# Create a temporary working directory on the node
WDIR=/home/reujpasternak/tmp/pii-state-ml
mkdir -p ${WDIR}/${INPUT_DIR}
mkdir -p ${WDIR}/${PLOT_DIR}
mkdir -p ${WDIR}/${PLOT_DATA_DIR}
cd ${WDIR}

# Copy set of input files to the working directory
cp ${HDIR}/${INPUT_DIR}/*.csv ${WDIR}/${INPUT_DIR}

# Setup PyTorch environment and run
python3 -m venv ${WDIR}/venv
source ${WDIR}/venv/bin/activate

python3 -m pip install matplotlib numpy scipy pandas # TODO: Use requirements.txt
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

echo "[JOB] About to run python script"
python3 -u ${HDIR}/lstm_matrix.py

# Copy the set of output files back to the original folder
cp ${WDIR}/${PLOT_DIR}/*      ${HDIR}/${PLOT_DIR}/
cp ${WDIR}/${PLOT_DATA_DIR}/* ${HDIR}/${PLOT_DATA_DIR}/

# Tidy up local files
rm -rf ${WDIR}/*

# TRIAL_SIZE=1000
# N_SIZES=(20 30 40 50 60 70 80 90 100)
# N_PROGRAMS=10

# N_INDEX=$(($SLURM_ARRAY_TASK_ID / $N_PROGRAMS))
# N_SIZE=${N_SIZES[$N_INDEX]}

# PROGRAM_ID=$(($SLURM_ARRAY_TASK_ID % $N_PROGRAMS))

# java Main $TRIAL_SIZE $N_SIZE data/stateData_2000_${N_SIZE}_ID${PROGRAM_ID}.csv $PROGRAM_ID