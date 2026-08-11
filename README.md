# PII Cycle Prediction

PII Cycle Prediction contains a variety of machine learning models used to predict whether a trial of PII will cycle. Models are split into two different categories: mean and matrix, based on what kind of input data they use.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install everything in the `requirements.txt` file:

```bash
pip install -r requirements.txt
```
If running the models through a .ipynb file or running the mean models, `ipykernel` must also be installed.

## Mean Model Usage

The mean models do not support running inference on a saved model or saving models in general. There is only the option to train a model on given input data.

To generate your own mean state data, use the `java-sequential-solver` repository. Use the [`mean-state-data-collection`](https://github.com/exercise-reu-2026-stable-matching/java-sequential-solver/tree/mean-state-data-collection) branch to collect mean state data. To use already existing data, untar and/or unzip the corresponding file in `mean_models/mean_data`.

To extract the csv from a .tar.gz file:
```bash
tar -xzf file.tar.gz
```
To extract csv from a .gz file:
```bash
gunzip -k file.gz
```

To train a mean model on input data, there is the option to use two separate files for training and testing, or use one file and specify a train/test split. 
```python
separateFileData = False # set to True for separate files for train and test
```
If train and test data are on two separate files, the value of n and names of the csv files must be specified.
```python
# Example using stateData_2000_20_ID0_itr0.csv as the training file and stateData_2000_20_ID1_itr0.csv as the testing file
n = 20
dataset_str = f"stateData_2000_{n}_ID0_itr0"
test_dataset_str = f"stateData_2000_{n}_ID1_itr0"
```

If train and test data are taken from the same file, the data length and train ratio must also be specified.
```python
# Example using 75% of stateData_20000_10_itr1.csv for training and 25% for testing
n = 10
data_len = 20000
dataset_str = f"stateData_{data_len}_{n}_itr1"
train_ratio = 0.75
```

Graphs detailing the loss and accuracy as the model trains will be output in either `mean_models/basic_nn_mean_plots` or `mean_models/lstm_mean_plots` based on which type of model was run.

## Matrix Model Usage

Matrix models do not support inference using an already trained model, but the transformer model does support saving checkpoints that can be used to continue training on a previously trained model.

### Matrix State Data

To generate your own matrix state data, use the `java-sequential-solver` repository. Use the [`matrix-state-data-collection`](https://github.com/exercise-reu-2026-stable-matching/java-sequential-solver/tree/matrix-state-data-collection) branch to collect matrix state data. To use already existing data, unzip the corresponding file in `mean_models/mean_data`. Both the iteration and trial files are needed.

```bash
gunzip -k file_iter.csv.gz
gunzip -k file_trial.csv.gz
```

### LSTM Matrix Model

To train the LSTM matrix model on input data, the data length, iteration number, and train/test split must be provided.
```python
# Example using matrixStateData_200000_10_iter.csv and matrixStateData_200000_10_trial.csv as input data
# The model will only train and test on iteration 1 with a train/test split of 75/25
n = 10
data_len = 200000
iter_index = 1
data_name = f"matrixStateData_{data_len}_{n}"
train_ratio = 0.75
test_ratio = 0.25
```

Graphs detailing the loss and accuracy as the LSTM model trains will be output in `matrix_models/lstm_matrix_plots`. The csvs used to generate these plots are found in `matrix_models/lstm_matrix_plot_data`.

### Transformer

To run the transformer model, a configuration file found in `matrix_models/transformer_configs` must be specified. If starting training from scratch, `weight_file` must be null and `start_epoch` must be 0.
```json
{
    // Sample configuration file
    "batch_size": 64,
    "learning_rate": 0.001,
    "epochs": 250,
    "n": 10,
    "buffer_size": 1048576, // How many bytes to write to a csv file at a time
    "csv_chunk_size": 1000000, // Number of lines read from data csvs at once
    "partition_size": 200000, // The number of trials in each partition
    "data_len": 2000000, // Number of trials in the dataset
    "iter_index": 0, // Specifies which iteration to train and test on
    "train_ratio": 0.75,
    "test_ratio": 0.25,
    "shuffle_rand_seed": 1, // Seed used for shuffling data
    "torch_rand_seed": 42, // Seed used any PyTorch randomness
    "output_size": 2,
    "hidden_size": 64,
    "num_layers": 4,
    "num_heads": 8,
    "start_epoch": 0, // Used to specify what epoch to start from when using a pre-trained checkpoint
    "weight_file": null, // Used when starting training from a pre-trained checkpoint
    "print_freq": 1, // How often accuracy and loss information is output
    "checkpoint_freq": 10 // How often a checkpoint is saved for the model
}
```

If continuing training on an already trained transformer model, the `weight_file` and `start_epoch` configurations must be set.
```json
{
    // Example configuration that starts from epoch 100 on an already trained model
    "start_epoch": 100,
    "weight_file": "matrix_models/saved_transformer_models/matrixStateData_2000000_10_iter0_hs64/matrixStateData_2000000_10_iter0_ID105100_n10_hs64_bs64_ep0100",
}
```

Graphs detailing the loss and accuracy as the LSTM model trains will be output in `matrix_models/transformer_matrix_plots`. The csvs used to generate these plots are found in `matrix_models/transformer_matrix_plot_data`. A saved checkpoint file will be generated every `checkpoint_freq` epochs. Saved checkpoint files are found in `saved_transformer_models` which are then sorted by `matrixStateData_data_len_n_iter_index_hidden_size`.

To run the transformer model, use the following command. The first argument specifies the configuration file that will be used to set up the input data and hyperparameters for the model.

```bash 
# Example using a specified configuration file
python transformer_matrix.py transformer_configs/matrixStateData_2000000_20_iter0_hs64.json
```

## File Manifest

- `matrix_models`
  - `lstm_matrix_plot_data`: directory that contains csvs with loss and accuracy data from training the LSTM
  - `lstm_matrix_plots`: directory that contains loss and accuracy graphs from training the LSTM
  - `matrix_data`: directory that contains matrix state data used as input
  - `saved_transformer_models`: directory that contains saved checkpoint files from trained transformer models
  - `transformer_configs`: directory that contains configuration files for the transformer
  - `transformer_matrix_plot_data`: directory that contains csvs with loss and accuracy data from training the transformer
  - `transformer_matrix_plots`: directory that contains loss and accuracy graphs from training the transformer
  - `lstm_matrix.py`: file used to train an LSTM model on matrix state data
  - `transformer_matrix.py`: file used to train a transformer model on matrix state data
  - `plot_from_datafile.ipynb`: file used to graph model accuracy data from multiple csvs
- `mean_models`
  - `basic_nn_mean_plots`: directory that contains loss and accuracy graphs from training the feed-forward model
  - `lstm_mean_plots`: directory that contains loss and accuracy graphs from training the LSTM model
  - `mean_data`: directory that contains mean state data used as input
  - `basic_nn_mean.py`: file used to train a feed-forward model on mean state data
  - `lstm_mean.py`: file used to train an LSTM model on mean state data

## Known Bugs

Support for using Slurm Workload Manager to run the matrix models and support for using a transformer with PyTorch's Distributed Data Parallel are included.

Job scripts for running Slurm jobs on all models have directories hard coded within script variables and must be changed to be used properly. In particular, `reujpasternak` is often hardcoded as the Linux username.

Within this repository, Distributed Data Parallel and Slurm support is only intended for 1 GPU per node, but this can be adjusted if multiple GPUs are available per node.

## Contact
Matthew Goldman  
mgoldman5@binghamton.edu

Juniper Pasternak  
juniper.pasternak23@kzoo.edu

## Acknowledgements

Thank you to William Bradley and Jeffrey Xu for their Java sequential solver implementation of PII.

Parts of our code are based on [2D Positional Encoding](https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py) (used with permission) and [Pytorch Tutorials](https://github.com/LukeDitria/pytorch_tutorials/blob/main/section14_transformers/solutions/Pytorch1_Transformer_Text_Classification_Multi_Block.ipynb) (used under the MIT license).

## License

[MIT](https://choosealicense.com/licenses/mit/)
