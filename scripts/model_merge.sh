#!/bin/bash
set -ux
CHECKPOINTS_DIR=""

find ${CHECKPOINTS_DIR} -maxdepth 2 -mindepth 2 -type d | while read BASE_DIR; do
    for global_step_dir in ${BASE_DIR}/global_step_*; do
        if [ -d "$global_step_dir" ]; then
            step_name=$(basename "$global_step_dir")
            echo "Processing: $step_name"

            if [ -d "$global_step_dir/hf" ]; then
                echo "Skipping: $global_step_dir/hf already exists"
                rm -rf $global_step_dir/actor
            elif [ -d "$global_step_dir/actor" ]; then
                echo "Merging $global_step_dir/actor -> $global_step_dir/hf"
                
                python model_merger.py merge \
                    --backend fsdp \
                    --local_dir "$global_step_dir/actor" \
                    --target_dir "$global_step_dir/hf"
            else
                echo "Warning: actor directory not found in $global_step_dir"
            fi

            echo "----------------------------------------"
        fi
    done
done
