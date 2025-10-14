#!/bin/bash

# Change this to your actual script's name
PYTHON_SCRIPT="run_analysis.py"

# Directory containing the .txt files, "." means current directory
INPUT_DIR="../mem_access_traces"

for csvfile in "$INPUT_DIR"/*.csv
do
    # Get the base name without the extension
    base=$(basename "$csvfile" .csv)
    echo "$csvfile"
    python3 "$PYTHON_SCRIPT" "$csvfile"
    echo "completed for $csvfile"
done
