#!/bin/bash
# Download Home Credit Default Risk dataset from Kaggle.
# Usage: export KAGGLE_API_TOKEN=your_token && bash download_data.sh
set -e
pip install kaggle -q
mkdir -p data/raw
kaggle competitions download -c home-credit-default-risk -p data/raw
unzip -q data/raw/home-credit-default-risk.zip -d data/raw
echo "Done — files in data/raw:"
ls -lh data/raw/*.csv
