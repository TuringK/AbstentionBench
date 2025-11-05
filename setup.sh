# Create and activate the new environment as per the environment.yml file
# mamba env create -f environment.yml

# mamba activate abstention-bench

# Install VLLM and PyTorch using pip, because we need specific CUDA-compatible versions
# pip install tiktoken sentencepiece --only-binary :all:
pip install xformers==0.0.29.post2
pip install vllm==0.8.5.post1 --extra-index-url https://download.pytorch.org/whl/cu124
 
if [[ $OSTYPE == "darwin"* ]]; then
  pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 -U 
else
  pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 -U --index-url https://download.pytorch.org/whl/cu124
fi 

pip install -e .

# Test that PyTorch is installed correctly
# pytorch_version_output=`python -c "import torch; print(torch.__version__)"`
# if [[ $pytorch_version_output == *"121"* ]]; then
#   echo "PyTorch is installed with cuda 12.1!"
# else
#   echo "PyTorch installation missing cuda 12.1. Please install with pip using official instructions"
# fi
