#!/bin/bash

# Path to the counts file
input_file="./fast5_counts.txt"

# Check if the file exists
if [ ! -f "$input_file" ]; then
    echo "File $input_file not found!"
    exit 1
fi

# Sum up the counts
total=$(awk -F ': ' '{sum += $2} END {print sum}' "$input_file")

echo "Total .fast5 files: $total"
