#!/bin/bash
set -eux
# PREFIX="occlusion"
PREFIX="first"
#PREFIX="no_overlap"
CLASSES=$1
CONFIG_PATH="config/shapenet_iclr2022.yaml"
# CONFIG_PATH="${2:-"config/shapenet_iclr2022.yaml"}"


PROJECT_ROOT="${PROJECT_ROOT:-"/home/mil/kawana/workspace/moving_primitives"}"
MODEL_ROOT="${PROJECT_ROOT}/external/neural_parts"
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT:$MODEL_ROOT
cd $MODEL_ROOT
    MM_CONFIG_PATH="${PROJECT_ROOT}/config_files/baselines/neural_parts_8.yaml"
    #MM_CONFIG_PATH="${PROJECT_ROOT}/config_files/baselines/nsd.yaml"
    ROOT_ARTIFACT_DIR="${PROJECT_ROOT}/artifacts/experiments/baselines/neural_parts"
    OUT_DIR="${ROOT_ARTIFACT_DIR}"
    python3 scripts/train_network.py \
        "${CONFIG_PATH}" \
        "${OUT_DIR}" \
        --experiment_tag "${PREFIX}_${CLASSES}" \
        --moving_primitive_config_path "${MM_CONFIG_PATH}" --classes "${CLASSES}"

cd ..

