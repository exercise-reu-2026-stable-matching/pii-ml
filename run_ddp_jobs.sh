#!/usr/bin/bash

if [ -z "$2" ]
  then
    echo "Missing required argument 2: number of nodes"
    exit 1
fi

CONFIG_FILE=${1}
NUM_NODES=${2}

PARTITION="hslinux"
# If arg 3 is not empty, then use it as the partition
if [ ! -z "$3" ]
  then
    PARTITION=${3}
fi

# If arg 4 is not empty, then use it as the node list
if [ ! -z "$4" ]
  then
    NODE_LIST_ARG="--nodelist ${4}"
else
    NODE_LIST_ARG=""
fi

JOB_OUTPUT=$(sbatch --nodes=${NUM_NODES} --partition ${PARTITION} ${NODE_LIST_ARG} sbatch_ddp_jobs.sh ${CONFIG_FILE})
echo $JOB_OUTPUT

# Send job ID to monitor script
JOB_ID=$(echo $JOB_OUTPUT | grep -E -o -a -m 1 -h "[0-9]+")
bash monitor $JOB_ID

echo "watch -n 2 tail -n 20 slurm_output/${JOB_ID}_00.out"
watch -n 2 "tail -n 20 slurm_output/${JOB_ID}_00.out"
