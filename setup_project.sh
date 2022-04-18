#!/bin/bash

mkdir auxiliar_files

mkdir config
touch config/variables.json

mkdir -p data/raw
mkdir -p data/structured

mkdir images

mkdir models

mkdir -p notebooks/eda
mkdir -p notebooks/evaluation
mkdir -p notebooks/modeling

mkdir -p src/feature_engineering
mkdir -p src/metrics
mkdir -p src/modeling
mkdir -p src/modeling/nn
mkdir -p src/processing
mkdir -p src/visualization


touch DataDumper.py
touch DataLoader.py
touch utils.py

touch requirements.txt
