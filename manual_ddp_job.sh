#!/usr/bin/bash

# Check for second argument
if [ -z "$2" ]
  then
    echo "Missing required argument 2: node number"
    exit 1
fi

REPO_DIR=/mnt/linuxlab/home/reujpasternak/pii-state-data-ml

tmux new -d -s ddp${2}
tmux send-keys -t ddp${2} "ssh -t hpcl${2} 'bash ${REPO_DIR}/tmux_ddp_job.sh ${1} $NUM_NODES $MAIN_NODE $SLURM_JOB_ID; bash'" C-m
