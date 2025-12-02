# Create and activate the new environment as per the environment.yml file
mamba env create -f environment.yml

mamba activate abstention-bench

# Install uv for faster and clearer dependacy management
pip install uv

# We need PyTorch fist to build vllm dependancies
if [[ $OSTYPE == "darwin"* ]]; then
  uv pip install torch==2.6.0 
else
  uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
fi 

uv pip install tiktoken sentencepiece --only-binary :all:
uv pip install xformers==0.0.29.post2 --no-build-isolation # build will use already installed Pytorch
uv pip install vllm==0.8.5.post1 --torch-backend=cu124 # --extra-index-url https://download.pytorch.org/whl/cu124

uv pip install -e .

# Test that PyTorch is installed correctly
pytorch_version_output=`python -c "import torch; print(torch.__version__)"`
if [[ $pytorch_version_output == *"124"* ]]; then
  echo "PyTorch is installed with cuda 12.4!"
else
  echo "PyTorch installation missing cuda 12.4. Please install with pip using official instructions"
fi
