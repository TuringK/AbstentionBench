# Mamba HPC Installation

## Prepare Shell
1. `module load GCC/12.3.0`
2. `module load CUDA/12.1.1`
No GPU needed for setting up the environment.

## Install Mamba
1. Go to miniforge3 releases (`https://github.com/conda-forge/miniforge/releases`) and download the `Miniforge3-25.3.1-0-Linux-x86_64.sh` script. Do this by `wget` or an FTP server.
2. Give execution rights (`chmod +x ./Miniforge3-25.3.1-0-Linux-x86_64.sh`) and run the script `./Miniforge3-25.3.1-0-Linux-x86_64.sh`

Mamba will ask for a default install directory, I would NOT recommend setting it to your home `/~` (default) to prevent hitting the file quota. I just put mine in on parscratch `/mnt/parscratch/users/[YOUR_USERNAME]/private`

## Add to PATH
Whichever directory you installed Mamba to, you need to add the `/bin` directory to your path. The easiest way of doing this is ediitng your `~/.bashrc`. At the very bottom of the file add the following line:
```bash
PATH="[MAMBA_INSTALLATION_DIR]/bin:$PATH"
```

Then reload the shell by running `source ~/.bashrc` or logging out and then back in onto Stanage.

## Installation Complete
You'll have Mamba installed on your HPC account so you can continue with the Abstention Dataset environment. You'll probably need to rerun `module load GCC/12.3.0` and `module load CUDA/12.1.1` to install the correct versions of pytorch or it'll probably spit out some errors
