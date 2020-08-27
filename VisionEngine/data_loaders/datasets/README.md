*Provided datasets*

We provide three datasets to evaluate our approach + one synthetic dataset generated as part of the DHRL pipeline. We recommend downloading these datasets using the provided python scripts (e.g. "guppies.py") by opening a terminal and running:
```
$ python guppies.py
```
They can also be accessed here: 

*Creating your own dataset*

We use tensorflow datasets to handle generation, preprocessing, caching and batch size. Adjust vae_data_loader.py lines 86-152 and 222-259

```python
# butterfly dataset
if self.config.data_loader.dataset == 'butterflies':
    if self.config.data_loader.use_real is True:
        if self.config.data_loader.use_generated is True:
            raise NotImplementedError
        else:
            list_data = tf.data.Dataset.list_files(str(self.data_dir/'*/*'), shuffle=False, seed=42) 
    else:
        raise NotImplementedError

    ds = list_data.map(preprocess_input, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# guppy dataset
elif self.config.data_loader.dataset == 'guppies':
    if self.config.data_loader.use_real is True:
        if self.config.data_loader.use_generated is True:
            list_data = tf.data.Dataset.list_files(str(self.data_dir/'*/*'), shuffle=False, seed=42)
        else:
            list_data = tf.data.Dataset.list_files(str(self.data_dir/'*_*/*'), shuffle=False, seed=42) 
    else:
        list_data = tf.data.Dataset.list_files(str(self.data_dir/'[!a-z][!a-z]/*'), shuffle=False, seed=42)

    ds = list_data.map(preprocess_input, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# celeba dataset
elif self.config.data_loader.dataset == 'celeba':
    if self.config.data_loader.use_real is True:
        if self.config.data_loader.use_generated is True:
            raise NotImplementedError
        else:
            list_data = tf.data.Dataset.list_files(str(self.data_dir/'*/*'), shuffle=False, seed=42) 
    else:
        raise NotImplementedError

    ds = list_data.map(preprocess_input_celeba, num_parallel_calls=tf.data.experimental.AUTOTUNE)

else:
    raise NotImplementedError
```

adding another control statement

```python
# your dataset
elif self.config.data_loader.dataset == 'your_dataset':
    if self.config.data_loader.use_real is True:
        if self.config.data_loader.use_generated is True:
            raise NotImplementedError
        else:
            list_data = tf.data.Dataset.list_files(str(self.data_dir/'*/*'), shuffle=False, seed=42) 
    else:
        raise NotImplementedError

    ds = list_data.map(preprocess_input_your_dataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)
```
you will need to reinstall the package by running
```
$ python setup.py install
```
 from the repository home directory

 you will also need to create a new config file for your data/experiment.
 using default_nouveau_config.json as a template, replacing the dataset value.
