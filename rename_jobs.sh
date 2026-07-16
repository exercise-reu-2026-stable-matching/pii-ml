#!/usr/bin/bash

if [ -z "$2" ]
  then
    echo "Missing required argument 2: append string"
    exit 1
fi

JOB_ID=$1
APPEND_STRING=$2

TARGET_DIR="."
if [ ! -z "$3" ]
  then
    TARGET_DIR=$3
fi

find $3 -name "*${1}*" -exec rename -v "${1}" "${2}" {} \;
