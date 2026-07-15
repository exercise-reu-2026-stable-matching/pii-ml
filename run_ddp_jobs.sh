#!/usr/bin/bash

# Check for first argument
if [ -z "$1" ]
  then
    echo "Missing required argument 1: configuration file"
    exit 1
fi

HASH=$(date +%s | md5sum | awk '{print $1}')
export SLURM_JOB_ID=$(python3 -c "print(f\"{int('$HASH', 16) % 1000000:06}\")")

export MAIN_NODE="hpcl2-1"
export NUM_NODES=8
printf "%s\0" 2-1 2-2 2-3 2-4 2-5 2-6 4-1 4-2 | xargs -0 -I @ -P 4 bash manual_ddp_job.sh $1 @
