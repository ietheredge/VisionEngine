# DHRL-VisionEngine
[![Travis (.com) branch](https://img.shields.io/travis/com/ietheredge/VisionEngine/master?logo=travis)](https://travis-ci.com/ietheredge/VisionEngine)
[![Latest PyPI version](https://img.shields.io/pypi/v/VisionEngine?color=blue&logo=pypi)](https://pypi.org/project/VisionEngine)
[![Conda (channel only)](https://img.shields.io/conda/vn/ietheredge/visionengine?color=blue&label=Anaconda%20Cloud&logo=Anaconda)](https://anaconda.org/ietheredge/visionengine)

The repository for the framework presented in [**Decontextualized learning for interpretable hierarchical representations of visual patterns**](https://doi.org/10.1016/j.patter.2020.100193).

![](../assets/Overview.png?raw=true)`

# Getting setup
We recommend creating a virtual environment to run VisionEngine using [anaconda](https://docs.anaconda.com/anaconda/user-guide/getting-started/?gclid=EAIaIQobChMIi5mM5-Hd5wIVhsjeCh1B_AheEAAYASAAEgJ-8PD_BwE).

Once anaconda/miniconda is installed, download VisionEngine and enter the HOME directory:

```bash
$ git clone https://github.com/ietheredge/VisionEngine
$ cd VisionEngine
```

Set up the environment: 
- export an .env file
- create a new conda environment (will install VisionEngine and all dependencies)

```bash
$ echo "VISIONENGINE_HOME = $(pwd)" >> .env
$ conda env create -f environment.yml
$ conda activate visionengine
```

# Raw data and trained models

All neceassary data files and trained models can be accessed [here](https://owncloud.gwdg.de/index.php/s/u6RQq20x1MHePl3).
*Note:* You do not need to download raw data for evaluation, this is done automatically by the dataloaders but you *will* need to put the downloaded model checkpoints [here](https://github.com/ietheredge/VisionEngine/tree/master/checkpoints) to use a pretrained model (e.g., to verify results).

# Evaluation

We provide three notebooks to evaluate trained models, visualize feature attributions and perform an evolutionary experiment [here](https://github.com/ietheredge/VisionEngine/tree/master/notebooks).

# Training a model from scratch

To start training a model, use one of the [config files](https://github.com/ietheredge/VisionEngine/tree/master/VisionEngine/configs), e.g.: 

```bash
$ python main.py -c configs/guppy_vae_config.json
```

Use tensorboard to observe training progress:

```bash
$ tensorboard --logdir logs --bind_all
```

# Using your own data

If you'd like to use your own data, make a custom dataloader for your dataset [here](https://github.com/ietheredge/VisionEngine/tree/master/VisionEngine/data_loaders), dataset should be placed in [this folder](https://github.com/ietheredge/VisionEngine/tree/master/VisionEngine/data_loaders/datasets). Consult the datasets [README](https://github.com/ietheredge/VisionEngine/tree/master/VisionEngine/data_loaders/datasets/README.md). Then create a new config file and make the appropriate changes [here](https://github.com/ietheredge/VisionEngine/tree/master/VisionEngine/configs). Finally, reinstall VisionEngine.

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
│   │   ├── _example_config.json
│   │   ├── butterflies_config.json
│   │   ├── generated_guppies_config.json
│   │   ├── guppies_config.json
│   │   └── guppies_DHRL_config.json
│   ├── data_loaders
│   │   ├── __init__.py
│   │   ├── dataset
│   │   ├── datasets
│   │   │   ├── README.md
│   │   │   ├── __init__.py
│   │   │   ├── butterflies.py
│   │   │   └── guppies.py
│   │   └── data_loader.py
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
│   │   └── trainer.py
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
└── setup.py
```

# Citation

If you use DHRL-VisionEngine in your own research, feel free to reachout to Ian Etheredge via [email](mailto:rianetheredge@gmail.com?subject=DHRL-VisionEngine) or on [twitter](https://twitter.com/ian_etheredge) and please cite our [paper](https://doi.org/10.1016/j.patter.2020.100193):

```
@article{ETHEREDGE2021100193,
title = "Decontextualized learning for interpretable hierarchical representations of visual patterns",
journal = "Patterns",
pages = "100193",
year = "2021",
issn = "2666-3899",
doi = "https://doi.org/10.1016/j.patter.2020.100193",
url = "http://www.sciencedirect.com/science/article/pii/S2666389920302634",
author = "Robert Ian Etheredge and Manfred Schartl and Alex Jordan",
keywords = "generative modeling, interpretable AI, disentangled representation learning, hierarchical features, image analysis, small data, decontextualized learning, feature attribution, latent evolution",
abstract = "Summary
Apart from discriminative modeling, the application of deep convolutional neural networks to basic research utilizing natural imaging data faces unique hurdles. Here, we present decontextualized hierarchical representation learning (DHRL), designed specifically to overcome these limitations. DHRL enables the broader use of small datasets, which are typical in most studies. It also captures spatial relationships between features, provides novel tools for investigating latent variables, and achieves state-of-the-art disentanglement scores on small datasets. DHRL is enabled by a novel preprocessing technique inspired by generative model chaining and an improved ladder network architecture and regularization scheme. More than an analytical tool, DHRL enables novel capabilities for virtual experiments performed directly on a latent representation, which may transform the way we perform investigations of natural image features, directly integrating analytical, empirical, and theoretical approaches."
}
```
