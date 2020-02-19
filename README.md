# VisionEngine
Code repository for the framework presented in VISIONENGINE, INVESTIGATING NATURAL COLOR-PATTERNS WITH MACHINE LEARNING. Read the [report](https://github.com/ietheredge/VisionEngine/tree/master/report/VisionEngine.pdf)

# Getting setup
We recommend creating a virtual environment to run VisionEngine, e.g. [Anaconda](https://docs.anaconda.com/anaconda/user-guide/getting-started/?gclid=EAIaIQobChMIi5mM5-Hd5wIVhsjeCh1B_AheEAAYASAAEgJ-8PD_BwE).

Once Anaconda is installed, run:
```bash
conda env -n VisionEngine tensorflow-gpu
conda activate VisionEngine
```
Then, to install VisionEngine, run: 
```bash 
git clone https://github.com/ietheredge/VisionEngine
cd VisionEngine
conda install --yes --file requirements.txt
python setup.py install
```
If you do not want to install VisionEngine locally, feel free to run the provided [Jupyter notebooks](https://github.com/ietheredge/VisionEngine/tree/master/notebooks) via [google colab](https://colab.research.google.com/notebooks/intro.ipynb) by clicking at the link at the top of each notebook.

# Raw data availability
The raw files and processed outputs can be accessed [here](https://owncloud.gwdg.de/index.php/s/6lpgoCEDpxlOuUq). For training a model from scratch, the required datasets will be automatically downloaded via the dataloaders, you do not need to do this yourself. Any missing data files required for Jupyter Notebooks will be downloaded automatically via the dataloaders. 

# Training a model from scratch
To start training with one of the [config files](https://github.com/ietheredge/VisionEngine/tree/master/VisionEngine/configs') run, e.g.: 
```bash
python main.py -c configs/guppy_vae_config.json
```
If you'd like to use your own data, create a new config file and make the appropriate changes [here](https://github.com/ietheredge/VisionEngine/tree/master/VisionEngine/configs') and make a custom dataloader for your dataset [here](https://github.com/ietheredge/VisionEngine/tree/master/VisionEngine/data_loaders'), dataset python files should be place in [this folder](https://github.com/ietheredge/VisionEngine/tree/master/VisionEngine/data_loaders/datasets') along with any required local data. 

# Repository Structure
```bash
.
├── checkpoints
├── data
│   └── processed
├── LICENSE
├── logs
├── notebooks
├── README.md
├── report
│   ├── figures
│   └── VisionEngine.pdf
├── requirements.txt
├── setup.py
└── VisionEngine
    ├── base
    │   ├── base_data_loader.py
    │   ├── base_model.py
    │   └── base_trainer.py
    ├── configs
    │   ├── butterfly_vae_config.json
    │   └── guppy_vae_config.json
    ├── data_loaders
    │   ├── butterfly_gan_data_loader.py
    │   ├── butterfly_vae_data_loader.py
    │   ├── guppy_gan_data_loader.py
    │   ├── guppy_vae_data_loader.py
    │   └── __init__.py
    ├── __init__.py
    ├── logs
    ├── main.py
    ├── models
    │   ├── gan_model.py
    │   └── vae_model.py
    ├── trainers
    │   ├── gan_trainer.py
    │   └── vae_trainer.py
    ├── utils
    │   ├── args.py
    │   ├── config.py
    │   ├── dirs.py
    │   ├── factory.py
    │   └── utils.py
    └── visualization
        └── visualize.py
```
