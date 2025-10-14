#!/bin/bash

# Change this to your actual script's name
PYTHON_SCRIPT="ghb_pipeline2_strided.py"

# Directory containing the .txt files, "." means current directory
INPUT_DIR="traces"
OUTPUT_DIR="mem_access_traces"

for txtfile in "$INPUT_DIR"/*.txt
do
    # Get the base name without the extension
    base=$(basename "$txtfile" .txt)
    csvfile="$OUTPUT_DIR/$base.csv"
    python3 "$PYTHON_SCRIPT" "$txtfile" "$csvfile" 8
    echo "completed for $txtfile"
done

