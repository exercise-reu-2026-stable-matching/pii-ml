#!/usr/bin/bash

cd data

CSV_HEADERS=$(cat "$(ls | grep -E -o -a -m 1 -h "stateData_.+_.+_ID.+\.csv" | head -n 1)" | head -n 1)

# 20 30 40 50 60 70 80 90 100
for i in 20 30 40 50 60 70 80 90 100
do
    # Combine all the files of each problem size
    tail -qn +2 stateData_2000_${i}_ID[0-99].csv > stateData_20000_${i}.csv

    # Prepend the CSV headers to the combined file
    sed -i "1i ${CSV_HEADERS}" stateData_20000_${i}.csv
done
