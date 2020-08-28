# DHRL-VisionEngine

The repository for the framework presented in **Decontextualized learning for interpretable hierarchical representations of visual patterns** [10.1101/2020.08.25.266593](https://www.biorxiv.org/content/10.1101/2020.08.25.266593v1).

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
- create a new environment
- install the VisionEngine package in the environment

```bash
$ VISIONENGINE_HOME=$(pwd); echo VISIONENGINE_HOME = $VISIONENGINE_HOME > .env
$ conda env create -f environment.yml
$ conda activate visionengine
$ python setup.py install
```

# Raw data and trained models

All neceassary data files and trained models can be accessed [here](https://owncloud.gwdg.de/index.php/s/u6RQq20x1MHePl3).
*Note:* You do not need to download raw data for evaluation, this is done automatically by the dataloaders but you *will* need to put the downloaded model checkpoints [here](https://github.com/ietheredge/VisionEngine/tree/master/checkpoints) to verify results.

# Evaluation

We provide three notebooks to evaluate trained models, visualize feature attributions and perform an evolutionary experiment [here](https://github.com/ietheredge/VisionEngine/tree/master/notebooks).

# Training a model from scratch

To start training a model, use one of the [config files](https://github.com/ietheredge/VisionEngine/tree/master/VisionEngine/configs): 

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

If you use DHRL-VisionEngine in your own research, feel free to reachout to Ian Etheredge via [email](mailto:ietheredge@ab.mpg.de?subject=DHRL-VisionEngine) or [twitter](https://twitter.com/ian_etheredge) and please cite our [preprint](https://www.biorxiv.org/content/10.1101/2020.08.25.266593v1):

```
@article {etheredge2020dhrl,
	author = {Etheredge, R. Ian and Schartl, Manfred and Jordan, Alex},
	title = {Decontextualized learning for interpretable hierarchical representations of visual patterns},
	elocation-id = {2020.08.25.266593},
	year = {2020},
	doi = {10.1101/2020.08.25.266593},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2020/08/25/2020.08.25.266593},
	eprint = {https://www.biorxiv.org/content/early/2020/08/25/2020.08.25.266593.full.pdf},
	journal = {bioRxiv}
}
```
