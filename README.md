# VisionEngine
Repository for code presented in A DEEP GENERATIVE FRAMEWORK FOR INVESTIGATING ANIMAL COLORATION PATTERNS

# Getting started

# Training a model from scratch
Start training with a config file from 'configs/', e.g. python main.py -c configs/vae_config.json

# Repository Structure
```bash
.
├── checkpoints
├── data
│   ├── processed
│   ├── raw
│   └── vae.py
├── LICENSE
├── logs
├── notebooks
│   ├── ColorPatternSpace.ipynb
│   ├── DRAFTColorPatternSpace_Flows-Copy1.ipynb
│   ├── DRAFTColorPatternSpace_Flows.ipynb
│   ├── GANotebook.ipynb
│   ├── Generator.ipynb
│   ├── SuperResolution.ipynb
│   └── TorchVAEFlows.ipynb
├── README.md
├── report
│   └── figures
├── requirements.txt
├── setup.py
└── src
    ├── base
    │   ├── base_data_loader.py
    │   ├── base_model.py
    │   └── base_trainer.py
    ├── configs
    │   └── vae_config.json
    ├── data_loader
    │   └── vae_data_loader.py
    ├── main.py
    ├── mains
    │   ├── evolve_samples.py
    │   └── vae.py
    ├── models
    │   └── vae_model.py
    ├── trainers
    │   └── vae_trainer.py
    ├── utils
    │   ├── args.py
    │   ├── config.py
    │   ├── dirs.py
    │   └── factory.py
    └── visualization
        └── visualize.py

```