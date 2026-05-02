#!/bin/bash

#SBATCH --partition=h100
#SBATCH --job-name=text_gradcam
#SBATCH --nodes=1                         
#SBATCH --ntasks=1                         
#SBATCH --gpus-per-task=1                 
#SBATCH --cpus-per-task=10                  
#SBATCH --mem=32G
#SBATCH --time=0
#SBATCH --exclude=node002

# ==========================================
# Environment Variables
# ==========================================
export PROJECT_ROOT="/MDS/gradient_feature" 
export DATA_ROOT="/MDS/data"

DATASET_NAME="quantity"
MODELS_ROOT=""
MODEL_PATH="${MODELS_ROOT}/Qwen3-VL-8B-Instruct"
SAMPLE_INDEX_FILE="${PROJECT_ROOT}/extract_result/${DATASET_NAME}/sample_index.json"

# Run text (text) and caption-based (caption) gradient analysis
MODES=("text" "caption")

for TARGET_MODE in "${MODES[@]}"
do
    echo "=========================================="
    echo "Processing MODEL: $MODEL_PATH | Mode: $TARGET_MODE"
    echo "=========================================="
    
    echo "--- Step 1: Running Text Gradient EXTRACTION ---"
    
    time python -m gradient_feature.extract_text_gradient \
        --dataset-name "$DATASET_NAME" \
        --model-name "$MODEL_PATH" \
        --mode "$TARGET_MODE" \
        --sample-index "$SAMPLE_INDEX_FILE"

    echo "--- Step 2: Running Sequential PLOTTING ---"
    
    time python -m gradient_feature.analysis.plot_text_gradient \
        --dataset-name "$DATASET_NAME" \
        --model-name "$MODEL_PATH" \
        --mode "$TARGET_MODE"
        
done

# Only iterate 8B model
for MODEL_NAME in "$MODEL_PATH"
do
    echo "=========================================="
    echo "Processing MODEL: $MODEL_NAME | Mode: Image Grad-CAM"
    echo "=========================================="
    
    MODE="image"
    
    echo "--- Step 1: Running Distributed EXTRACTION (Rank distributed) ---"
    
    time python -m gradient_feature.extract_gradcam \
        --dataset-name "$DATASET_NAME" \
        --model-name "$MODEL_NAME" \
        --sample-index "$SAMPLE_INDEX_FILE"

    echo "--- Step 2: Running Sequential PLOTTING (Viz on master task) ---"
    

    time python -m gradient_feature.analysis.plot_gradcam_map \
        --dataset-name "$DATASET_NAME" \
        --model-name "$MODEL_NAME"
        
done

echo "=========================================="
echo "Pipeline FINISHED."
echo "=========================================="