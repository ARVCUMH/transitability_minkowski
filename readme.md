# Installation
## Requirements
    Ubuntu >= 14.04
    CUDA >= 10.1.243 and the same CUDA version used for pytorch (e.g. if you use conda cudatoolkit=11.1, use CUDA=11.1 for MinkowskiEngine compilation)
    pytorch >= 1.7 You must match the CUDA version pytorch uses and CUDA version used for Minkowski Engine installation.
    python >= 3.6
    ninja (for installation)
    GCC >= 7.4.0
    
## Minkowski Engine installation
The **Minkowski Engine** is distributed via PyPI MinkowskiEngine which can be installed simply with **pip**. First, install pytorch following the instruction. Next, install openblas.

    sudo apt install build-essential python3-dev libopenblas-dev
    pip install torch ninja
    pip install -U MinkowskiEngine --install-option="--blas=openblas" -v --no-deps
    
    For pip installation from the latest source
    pip install -U git+https://github.com/NVIDIA/MinkowskiEngine --no-deps

## Quick Start
- First of all you must fill in the **config.yaml** file with the parameters that it contains.
- Afterwards the **dataset.py** file read and prepare the dato for the network. If you wish change the features which we are training the network you should change this script.
- Once the data is ready and its clear the neural network input, the training process is described in **training.py**.
-     python3 training.py
