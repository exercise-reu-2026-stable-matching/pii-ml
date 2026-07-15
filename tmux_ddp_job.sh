#!/usr/bin/bash

# Check for fourth argument
if [ -z "$4" ]
  then
    echo "Missing required argument 4: job id"
    exit 1
fi

echo "[JOB] Running job on $(hostname)"

REPO_DIR=/mnt/linuxlab/home/reujpasternak/pii-state-data-ml
cd $REPO_DIR

# Create TMP directory
mkdir -p /home/reujpasternak/tmp
export TMPDIR=/home/reujpasternak/tmp/

# Obtain the matrix_models home directory
HDIR=${REPO_DIR}/matrix_models
cd $HDIR

# Specify configuration file
CONFIG_FILE=${1}
echo "[JOB] Using configuration file ${1}"

# Specify number of nodes and main node
NUM_NODES=${2}
MAIN_NODE=${3}
export SLURM_JOB_ID=${4}
echo "[JOB] Running DDP on $NUM_NODES nodes with $MAIN_NODE as main"

# List directory names
CONFIG_DIR=transformer_configs
INPUT_DIR=matrix_data
SAVED_MODELS_DIR=saved_transformer_models
PLOT_DIR=transformer_matrix_plot_data
PLOT_DATA_DIR=transformer_matrix_plots

# Create a temporary working directory on the node
WDIR=/home/reujpasternak/tmp/pii-state-ml
mkdir -p ${WDIR}
cd ${WDIR}

# Tidy up prior job files, except the virtual environment
echo "[JOB] Cleaning up any existing local files"
ls ${WDIR} | grep -xvF torch_venv | xargs rm -vrf --

# Create subdirectories within working directory
mkdir -p ${WDIR}/${CONFIG_DIR}
mkdir -p ${WDIR}/${INPUT_DIR}
mkdir -p ${WDIR}/${SAVED_MODELS_DIR}
mkdir -p ${WDIR}/${PLOT_DIR}
mkdir -p ${WDIR}/${PLOT_DATA_DIR}

# Copy set of input files to the working directory
echo "[JOB] Copying input files to working directory..."
cp    ${HDIR}/${CONFIG_DIR}/*.json     ${WDIR}/${CONFIG_DIR}
cp    ${HDIR}/${INPUT_DIR}/*.csv.gz    ${WDIR}/${INPUT_DIR}
cp -a ${HDIR}/${SAVED_MODELS_DIR}/*    ${WDIR}/${SAVED_MODELS_DIR}

# Unzip input data
echo "[JOB] Unzipping input data..."
gunzip -v ${WDIR}/${INPUT_DIR}/*.csv.gz

# Setup PyTorch environment and run
echo "[JOB] Creating PyTorch virtual environment..."

export PYTHONOPTIMIZE=2 # Disable assertions and docstrings
python3 -m venv ${WDIR}/torch_venv
source ${WDIR}/torch_venv/bin/activate

python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
python3 -m pip install -r ${REPO_DIR}/requirements.txt

echo "[JOB] About to run python script"
python3 -m torch.distributed.run \
  --nnodes=${NUM_NODES} \
  --nproc_per_node=1 \
  --rdzv_id=${SLURM_JOB_ID} \
  --rdzv_backend=c10d \
  --rdzv_conf="join_timeout=600" \
  --rdzv_endpoint=${MAIN_NODE}:29400 \
  ${HDIR}/ddp_transformer_matrix.py \
  $CONFIG_DIR/$CONFIG_FILE

# TODO: If this is the main node
# Copy the set of output files back to the original folder
echo "[JOB] Copying output files back to home directory..."
cp -a ${WDIR}/${PLOT_DIR}/*         ${HDIR}/${PLOT_DIR}/
cp -a ${WDIR}/${PLOT_DATA_DIR}/*    ${HDIR}/${PLOT_DATA_DIR}/
cp -a ${WDIR}/${SAVED_MODELS_DIR}/* ${HDIR}/${SAVED_MODELS_DIR}/

echo "[JOB] Cleaning up local files"
# Tidy up local files, except the virtual environment (save for future use)
ls ${WDIR} | grep -xvF torch_venv | xargs rm -vrf --

echo "[JOB] Job ${SLURM_JOB_ID} complete!"