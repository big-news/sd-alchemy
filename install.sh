#!/bin/bash

source ./utils.sh

check_python_version() {
    which python >/dev/null 2>&1 || fatalln "Python is not installed. Please install Python 3.10.6 or higher."

    python_version=$(python --version | cut -d " " -f 2)
    major_version=$(echo $python_version | cut -d "." -f 1)
    minor_version=$(echo $python_version | cut -d "." -f 2)
    patch_version=$(echo $python_version | cut -d "." -f 3)
    if [[ $major_version -ne 3 || $minor_version -lt 10 || $patch_version -lt 6 ]]; then
        fatalln "Python version '${python_version}' is smaller than 3.10.6 and thus not supported. Please install Python 3.10.6 or higher."
    fi
    successln "Python version: '${python_version}'"
}

create_venv() {
    infoln "Creating virtual environment..."
    python -m venv venv --upgrade-deps
    source venv/bin/activate
}

install_dependencies() {
    pushd sd-scripts
    infoln "Installing dependencies..."
    
    pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116 || fatalln "Failed to install PyTorch. Please install PyTorch 1.12.1+cu116."
    pip install --upgrade -r requirements.txt || fatalln "Failed to install dependencies. Please install the dependencies from requirements.txt."
    pip install -U -I --no-deps https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/linux/xformers-0.0.14.dev0-cp310-cp310-linux_x86_64.whl || fatalln "Failed to install xformers. Please install xformers 0.0.14.dev0."
    pip install lion-pytorch || fatalln "Failed to install lion-pytorch. Please install lion-pytorch."
    pip install dadaptation || fatalln "Failed to install dadaptation. Please install dadaptation."
    pip install tensorboard || fatalln "Failed to install tensorboard. Please install tensorboard."

    successln "Installation finished."
    popd
}

check_python_version
create_venv
install_dependencies