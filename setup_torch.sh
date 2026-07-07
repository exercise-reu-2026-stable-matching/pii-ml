#!/usr/bin/bash

# mkdir -p /home/reujpasternak/tmp
# export TMPDIR=/home/reujpasternak/tmp/

# Setup Python installation
python3 -m venv ${0}
source ${0}/bin/activate

python3 -m pip install matplotlib numpy scipy pandas
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
