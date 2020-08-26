# Decontextualized learning for interpretable hierarchical representations of visual patterns



# DHRL/VisionEngine

![](../assets/Overview.png?raw=true)

Welcome to the repository for the framework presented in **VisionEngine, investigating natural color-patterns with machine learning**. Read the complete report [here](https://github.com/ietheredge/VisionEngine/tree/master/report/VisionEngine.pdf).

# Getting setup
We recommend creating a virtual environment to run VisionEngine using [Anaconda](https://docs.anaconda.com/anaconda/user-guide/getting-started/?gclid=EAIaIQobChMIi5mM5-Hd5wIVhsjeCh1B_AheEAAYASAAEgJ-8PD_BwE).

Once Anaconda is installed, run:
```bash
conda env -n VisionEngine tensorflow-gpu
```
Then, to get VisionEngine, run: 
```bash
conda activate VisionEngine
git clone https://github.com/ietheredge/VisionEngine
cd VisionEngine
conda install --yes --file requirements.txt
python setup.py install
```
If you do not wish to install VisionEngine locally, feel free to run the provided [notebooks](https://github.com/ietheredge/VisionEngine/tree/master/notebooks) via google colab by clicking the link at the top of each notebook.

# Raw data availability
The raw files and processed outputs can be accessed [here](https://owncloud.gwdg.de/index.php/s/6lpgoCEDpxlOuUq). For training a model from scratch and using the provided notebook the required datasets will be automatically downloaded via the dataloaders, so you do not need to do anything.

# Training a model from scratch
To start training with one of the [config files](https://github.com/ietheredge/VisionEngine/tree/master/VisionEngine/configs) run, e.g.: 
```bash
python main.py -c configs/guppy_vae_config.json
```
If you'd like to use your own data, create a new config file and make the appropriate changes [here](https://github.com/ietheredge/VisionEngine/tree/master/VisionEngine/configs) and make a custom dataloader for your dataset [here](https://github.com/ietheredge/VisionEngine/tree/master/VisionEngine/data_loaders), dataset python files should be place in [this folder](https://github.com/ietheredge/VisionEngine/tree/master/VisionEngine/data_loaders/datasets) along with any required local data. Consult the datasets [README](https://github.com/ietheredge/VisionEngine/tree/master/VisionEngine/data_loaders/datasets/README.md) for setting up your own datasets. 

# Repository Structure
```bash
.
├── LICENSE
├── README.md
├── VisionEngine
│   ├── __init__.py
│   ├── base
│   │   ├── __init__.py
│   │   ├── base_data_loader.py
│   │   ├── base_model.py
│   │   └── base_trainer.py
│   ├── configs
│   │   ├── butterfly_vae_config.json
│   │   ├── celeba_vae_config.json
│   │   └── guppy_vae_config.json
│   ├── data_loaders
│   │   ├── __init__.py
│   │   ├── dataset
│   │   ├── datasets
│   │   │   ├── README.md
│   │   │   ├── __init__.py
│   │   │   ├── butterflies.py
│   │   │   ├── celeba.py
│   │   │   └── guppies.py
│   │   └── vae_data_loader.py
│   ├── extensions
│   │   ├── feature_attribution.py
│   │   └── latent_evolution.py
│   ├── layers
│   │   ├── __init__.py
│   │   ├── noise_layer.py
│   │   ├── perceptual_loss_layer.py
│   │   ├── spectral_normalization_wrapper.py
│   │   ├── squeeze_excite_layer.py
│   │   └── variational_layer.py
│   ├── main.py
│   ├── tests
│   │   └── variational_layer_test.py
│   ├── trainers
│   │   ├── __init__.py
│   │   ├── gan_trainer.py
│   │   └── vae_trainer.py
│   └── utils
│       ├── __init__.py
│       ├── args.py
│       ├── config.py
│       ├── dirs.py
│       ├── disentanglement_score.py
│       ├── eval.py
│       ├── factory.py
│       ├── perceptual_loss.py
│       └── plotting.py
├── checkpoints
├── environment.yml
├── logs
├── notebooks
│   ├── EvaluateAndCompareModels.ipynb
│   ├── EvolveSamples.ipynb
│   └── FeatureAttribution.ipynb
├── requirements.txt
└── setup.py
```
