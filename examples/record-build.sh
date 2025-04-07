#!/bin/bash

# Script to run build.sh and save its output to a text file

# Path to the build.sh script
BUILD_SCRIPT="./build.sh"

# Output file
OUTPUT_FILE="./build_output.txt"

# Run the build script and redirect both stdout and stderr to the output file
echo "Running build.sh and saving output to $OUTPUT_FILE..."
bash $BUILD_SCRIPT > $OUTPUT_FILE 2>&1

echo "Build completed. Output saved to $OUTPUT_FILE"