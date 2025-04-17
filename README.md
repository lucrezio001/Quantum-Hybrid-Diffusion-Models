# Quantum-Machine-Learning-Hybrid-Vertex-U-net
Implementazione pytorch del paper "Quantum Hybrid Diffusion Models for Image Synthesis" per l'esame di Quantum Machine Learning, Luca Capece

## Quickstart

[Note in tools.train_ddpm.py e tools.train_ddpm_finetune.py Different type of U-net must be changed in the toggling on and off comment for the selected U-net]
[All other Hyperparameter can be configured using the default.yaml file]

* Start Train from scratch of the selected net
* ```python -m tools.train_ddpm```
* Start finetuning train of selected net (adapted to fine tune quantum for classical you can use both train_ddpm)
* ```python -m tools.train_ddpm_finetune```
* Start Sampling process from the selected U-Net 
* ```python -m tools.sample_ddpm``` 
* Return The metric FID KID for existing experiment 
* ```python -m tools.fid```

