#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/raw

kaggle datasets download \
  -d sakshigoyal7/credit-card-customers \
  -p data/raw \
  --unzip

echo "Dataset descargado en data/raw"
