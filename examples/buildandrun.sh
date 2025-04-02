#!/usr/bin/env bash

# Optional: Exit immediately if a command exits with a non-zero status
# set -e

# Run build.sh
echo "Running build.sh..."
./build.sh

# If build.sh succeeded, run run.sh
echo "Running run.sh..."
./run.sh

echo "All scripts have been executed."