#!/bin/bash
# ConnecToMind2 Environment Setup
# Usage: bash setup.sh

set -e

VENV_DIR="venv"
PYTHON="python3"

# ============ 1. Python version check ============
echo "Checking Python version..."
$PYTHON --version
PYTHON_VERSION=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [[ "$PYTHON_VERSION" < "3.10" ]]; then
    echo "Error: Python 3.10+ required (found $PYTHON_VERSION)"
    exit 1
fi

# ============ 2. Create venv (isolated) ============
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
    source $VENV_DIR/bin/activate
    TORCH_LIB="$VENV_DIR/lib/python3.10/site-packages/torch/lib"
    export LD_LIBRARY_PATH="$TORCH_LIB:$LD_LIBRARY_PATH"
    echo "Activated: $(which python)"
    echo "To recreate, run: rm -rf $VENV_DIR && bash setup.sh"
else
    echo "Creating virtual environment..."
    $PYTHON -m venv $VENV_DIR

    source $VENV_DIR/bin/activate

    # Override LD_LIBRARY_PATH to use venv's torch libs (prevents NGC system lib conflicts)
    TORCH_LIB="$VENV_DIR/lib/python3.10/site-packages/torch/lib"
    export LD_LIBRARY_PATH="$TORCH_LIB:$LD_LIBRARY_PATH"

    echo "Activated: $(which python)"

    # ============ 3. Install dependencies (order matters) ============
    echo ""
    echo "Upgrading pip..."
    pip install --upgrade pip

    # 0. PyTorch + torchvision (CUDA 12.1)
    echo "[0/7] Installing PyTorch..."
    pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

    # 1. numpy (pin first to avoid conflicts)
    echo "[1/7] Installing numpy..."
    pip install numpy==1.26.4

    # 2. Core dependencies (version conflict-sensitive)
    echo "[2/7] Installing core dependencies..."
    pip install tokenizers==0.14.1
    pip install safetensors==0.7.0
    pip install transformers==4.34.1
    pip install diffusers==0.14.0
    pip install accelerate==1.12.0
    pip uninstall huggingface_hub -y 2>/dev/null || true
    pip install huggingface_hub==0.21.4

    # 3. DeepSpeed
    echo "[3/7] Installing DeepSpeed..."
    pip install xgboost==1.7.5
    pip install deepspeed==0.14.0

    # 4. Image/Vision
    echo "[4/7] Installing image/vision libraries..."
    pip install scikit-image==0.21.0
    pip install Pillow==10.4.0

    # 5. CLIP
    echo "[5/7] Installing CLIP..."
    pip install ftfy==6.3.1
    pip install regex==2024.5.15
    pip install git+https://github.com/openai/CLIP.git

    # 6. Data/Experiment tools
    echo "[6/7] Installing data/experiment tools..."
    pip install nibabel==5.3.3
    pip install wandb==0.24.0
    pip install pandas==2.2.1
    pip install scipy==1.13.1
    pip install tqdm==4.66.4
    pip install timm==1.0.24
    pip install natsort

    # 7. Matplotlib
    echo "[7/7] Installing matplotlib..."
    pip install matplotlib==3.8.0

    # ============ 4. Verify installation ============
    echo ""
    echo "============ Verification ============"
    python -c "
import torch
print(f'  PyTorch:      {torch.__version__}')
print(f'  CUDA:         {torch.cuda.is_available()} ({torch.version.cuda})')
print(f'  GPUs:         {torch.cuda.device_count()}')
import transformers, diffusers, accelerate, deepspeed, clip
print(f'  transformers: {transformers.__version__}')
print(f'  diffusers:    {diffusers.__version__}')
print(f'  accelerate:   {accelerate.__version__}')
print(f'  deepspeed:    {deepspeed.__version__}')
print(f'  CLIP:         OK')
print()
print('All dependencies installed successfully!')
"
fi

# ============ 5. Run experiment ============
echo ""
echo "============ Running Experiment ============"
for SUB in sub-01
do
    NCCL_P2P_DISABLE=1 accelerate launch \
        --config_file configs/accelerate_config.yaml \
        main_accelerate.py \
        --experiment_name connectomind2_${SUB}_vis-only \
        --model_name connectomind2_${SUB}_vis-only \
        --recon_name recon_${SUB}_vis-only \
        --metrics_name metrics_${SUB}_vis-only \
        --root_dir /workspace/connectomind2 \
        --cache_dir /workspace/connectomind2/5-code/cache \
        --output_dir /workspace/connectomind2/5-code/output-reb \
        --roi_suffix schaefer100_vis \
        --seq_len 17 \
        --num_query_tokens 17 \
        --val_size 130 \
        --batch_size 32 \
        --inference_batch_size 32 \
        --num_epochs 151 \
        --fir_weight 100.0 \
        --fic_weight 0.1 \
        --fim_weight 1.0 \
        --lowlevel_weight 1.0 \
        --connectivity_type SC \
        --subjects ${SUB} \
        --input_dim 2088 \
        --resume_checkpoint /workspace/connectomind2/5-code/output-reb/connectomind2_sub-01_vis-only/connectomind2_sub-01_vis-only_epoch120/connectomind2_sub-01_vis-only_epoch120.pt
done