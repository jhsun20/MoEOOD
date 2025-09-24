# Mixture-of-Experts (MoE) for OOD Graph Learning

This repository implements a Mixture-of-Experts (MoE) architecture for improving Out-of-Distribution (OOD) generalization in graph learning tasks, with a focus on graph classification. We will be using the GOOD benchmark for all training and evaluation.

# Replication Instructions

```bash
python3 -m venv moeoodenv
source moeoodenv/bin/activate
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118 pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html torch-geometric==2.5.0 pyyaml wandb numpy==1.26.4 gdown munch dive-into-graphs entmax optuna plotly rdkit-pypi
python src/main.py --config config/config_hiv_size.yaml