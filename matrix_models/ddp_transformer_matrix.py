# %%
# %reset -f

# %%
import torch
# import torch.accelerator
import torch.distributed as dist
from torch import nn
from torch.nn.utils.rnn import pack_sequence, unpack_sequence, PackedSequence
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

import os
import sys
import math
import timeit
import datetime
import json
import logging
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
try:
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
except AttributeError:
    device = "cpu"
logging.info("Running Torch on device: %s", device)

# %%
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)-8s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
    force=True,
)

# %%
# Gather environment variables
JOB_ID = os.environ.get("SLURM_JOB_ID", "x")

# %%
# Simulate sys.argv in IPYNB
try:
    get_ipython() # pyright: ignore[reportUndefinedVariable]
    sys.argv = ["transformer_matrix.ipynb"]
    # sys.argv = ["transformer_matrix.ipynb", "transformer_configs/matrixStateData_2000000_20_iter0_hs64.json"]
except NameError:
    pass

# %%
# IF CONFIG FILE PROVIDED, use its parameters
if len(sys.argv) > 1:
    input_file = sys.argv[1]

    with open(input_file) as f:
        config = json.load(f)

    assert isinstance(config, dict)

    # --- HYPER PARAMETERS ---
    BATCH_SIZE = config["batch_size"]
    LEARNING_RATE = config["learning_rate"]
    EPOCHS = config["epochs"]
    START_EPOCH = config["start_epoch"]


    # --- DATA PREPROCESSING PARAMETERS ---
    BUFFER_SIZE:    int = config.get("buffer_size", 2 << 19)
    CSV_CHUNK_SIZE: int = config.get("csv_chunk_size", 1000000)
    PARTITION_SIZE: int = config["partition_size"]


    # --- DATASET PARAMETERS ---
    N = config["n"]
    DATA_LEN = config["data_len"]
    ITER_INDEX = config["iter_index"]
    DATA_NAME = config.get("data_name", f"matrixStateData_{DATA_LEN}_{N}")

    # Scalar multiple for ratios to get 200k exactly divisible by 64: 0.99968
    TRAIN_RATIO = config["train_ratio"]
    TEST_RATIO = config["test_ratio"]

    SHUFFLE_RAND_SEED = config["shuffle_rand_seed"]


    # --- MODEL HYPER PARAMETERS ---
    OUTPUT_SIZE = config["output_size"]
    HIDDEN_SIZE = config["hidden_size"]
    NUM_LAYERS = config["num_layers"]
    NUM_HEADS = config["num_heads"]
    SAVE_MODEL_SUBDIR = config.get("save_model_subdir", f"{DATA_NAME}_iter{ITER_INDEX}_hs{HIDDEN_SIZE}")
    TORCH_RAND_SEED = config.get("torch_rand_seed", torch.seed())

    # Parameters for saved weights
    WEIGHT_FILE_JOB_ID = config.get("weight_file_job_id")
    WEIGHT_FILE = config.get("weight_file")

    if WEIGHT_FILE is None and WEIGHT_FILE_JOB_ID is not None:
        WEIGHT_FILE = f"saved_transformer_models/{DATA_NAME}_iter{ITER_INDEX}_hs{HIDDEN_SIZE}/{DATA_NAME}_iter{ITER_INDEX}_ID{WEIGHT_FILE_JOB_ID}_ep{START_EPOCH:04d}.pt"

    # --- TRAINING PARAMETERS ---
    PRINT_FREQ = config["print_freq"]
    CHECKPOINT_FREQ = config["checkpoint_freq"]

# OTHERWISE, use hard-coded parameters
else:
    # --- HYPER PARAMETERS ---
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    EPOCHS = 25
    START_EPOCH = 0


    # --- DATA PREPROCESSING PARAMETERS ---
    BUFFER_SIZE    = 2 << 19 # 1 MB
    CSV_CHUNK_SIZE = 1000000
    PARTITION_SIZE = 20000


    # --- DATASET PARAMETERS ---
    N = 20
    DATA_LEN = 200000
    ITER_INDEX = 0
    DATA_NAME = f"matrixStateData_{DATA_LEN}_{N}"

    # Scalar multiple for ratios to get 200k exactly divisible by 64: 0.99968
    TRAIN_RATIO = 0.75
    TEST_RATIO = 0.25

    SHUFFLE_RAND_SEED = 1

    # --- MODEL HYPER PARAMETERS ---
    OUTPUT_SIZE = 2
    HIDDEN_SIZE = 128
    NUM_LAYERS = 4
    NUM_HEADS = 8
    SAVE_MODEL_SUBDIR = f"{DATA_NAME}_iter{ITER_INDEX}_hs{HIDDEN_SIZE}"
    TORCH_RAND_SEED = torch.seed()

    # Parameters for saved weights
    WEIGHT_FILE = None
    WEIGHT_FILE_JOB_ID = "x"
    WEIGHT_FILE = f"saved_transformer_models/{DATA_NAME}_iter{ITER_INDEX}_hs{HIDDEN_SIZE}/{DATA_NAME}_iter{ITER_INDEX}_ID{WEIGHT_FILE_JOB_ID}_ep{START_EPOCH:04d}.pt"


    # --- TRAINING PARAMETERS ---
    PRINT_FREQ = 1
    CHECKPOINT_FREQ = 10

NP_RAND_GEN = np.random.default_rng(SHUFFLE_RAND_SEED)

# %%
os.makedirs(os.path.join("saved_transformer_models", SAVE_MODEL_SUBDIR), exist_ok=True)

# Save all hyper parameters to a JSON file
with open(f"saved_transformer_models/{SAVE_MODEL_SUBDIR}/{DATA_NAME}_iter{ITER_INDEX}_ID{JOB_ID}.json", "w") as f:
    json.dump(
        {
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            
            "buffer_size": BUFFER_SIZE,
            "csv_chunk_size": CSV_CHUNK_SIZE,
            "partition_size": PARTITION_SIZE,

            "n": N,
            "data_len": DATA_LEN,
            "iter_index": ITER_INDEX,
            "data_name": DATA_NAME,
            "train_ratio": TRAIN_RATIO,
            "test_ratio": TEST_RATIO,
            "shuffle_rand_seed": SHUFFLE_RAND_SEED,

            "output_size": OUTPUT_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "num_heads": NUM_HEADS,
            "save_model_subdir": SAVE_MODEL_SUBDIR,
            "torch_rand_seed": TORCH_RAND_SEED,
            "start_epoch": START_EPOCH,
            "weight_file_job_id": WEIGHT_FILE_JOB_ID,
            "weight_file": WEIGHT_FILE,

            "print_freq": PRINT_FREQ,
            "checkpoint_freq": CHECKPOINT_FREQ,
        },
        f
    )

# %%
logging.info("Initializing process group")
dist.init_process_group("gloo")
RANK = dist.get_rank()
WORLD_SIZE = dist.get_world_size()
logging.info(f"Process group initialized. Rank: {RANK}/{WORLD_SIZE - 1}")

# %%
def list_from_str(str_list: str, fn_apply_to_items: Callable) -> list:
    if str_list in ["", "[]"]:
        return []

    elems = str_list.strip("[]\"").split(", ")
    return list(map(fn_apply_to_items, elems))

# %%
def unzip(zipped_list: list[tuple], output_length: int = 1) -> tuple:
    if len(zipped_list) == 0:
        return tuple([] for _ in range(output_length))

    return tuple(map(list, zip(*zipped_list, strict=True)))

# %%
def sample_iter_df(df: pd.DataFrame, iter_index: int) -> pd.DataFrame:
    if iter_index >= 0:
        return df[df.iterationIndex == iter_index]
    if iter_index == -1:
        return df.iloc[df[df.iterationIndex == 0].index - 1]

    raise ValueError("iter_index must be at least -1")

# %%
def slice_columns(df: pd.DataFrame, start_str: str, end_str: str) -> pd.DataFrame:
    start_idx, end_idx = df.columns.slice_locs(start_str, end_str)
    return df.iloc[:, start_idx:end_idx]

# %%
pd.set_option('display.max_columns', 500)
# pd.reset_option('display.max_columns')

torch.set_printoptions(edgeitems=7)
# torch.set_printoptions(edgeitems=3)

# %%
def preprocess_sampling(data_file_name: str, iter_index: int) -> str:
    iter_file = f"{data_file_name}_iter.csv"
    new_file = f"{data_file_name}_iter_{iter_index}_sample.csv"

    with open(new_file, "w", buffering=BUFFER_SIZE) as new_fp:
        # Write the column headers to the new file
        header_df = pd.read_csv(iter_file, nrows=0)
        header_df.to_csv(new_fp, header=True, index=False)

        # Write the data in chunks to the new file
        df_iter = pd.read_csv(iter_file, chunksize=CSV_CHUNK_SIZE)
        for chunk in df_iter:
            sample_iter_df(chunk, iter_index).to_csv(new_fp, header=False, index=False)

    return new_file

# %%
def preprocess_joining(data_file_name: str, sample_iter_file: str, iter_index: int) -> str:
    trial_file = f"{data_file_name}_trial.csv"
    new_file = f"{data_file_name}_combine_iter_{iter_index}_unshuffled.csv"

    with open(new_file, "w", buffering=BUFFER_SIZE) as new_fp:
        # Write the column headers to the new file
        header_df_iter = pd.read_csv(sample_iter_file, nrows=0)
        header_df_trial = pd.read_csv(trial_file, nrows=0)
        header_df = pd.merge(header_df_iter, header_df_trial, on=["programIndex", "trialIndex"])
        header_df.to_csv(new_fp, header=True, index=False)

        # Write the data in chunks to the new file
        df_iter = pd.read_csv(sample_iter_file, chunksize=CSV_CHUNK_SIZE)
        df_trial = pd.read_csv(trial_file, chunksize=CSV_CHUNK_SIZE)

        for chunk_iter, chunk_trial in zip(df_iter, df_trial, strict=True):
            merged_df = pd.merge(chunk_iter, chunk_trial, on=["programIndex", "trialIndex"])
            merged_df.to_csv(new_fp, header=False, index=False)

    os.remove(sample_iter_file)
    return new_file

# %%
def preprocess_shuffling(data_file_name: str, combined_file: str, iter_index: int):
    new_file = f"{data_file_name}_combine_iter_{iter_index}.csv"

    with open(combined_file, "rb") as f:
        row_count = sum(1 for _ in f) - 1

    with open(new_file, "w", buffering=BUFFER_SIZE) as new_fp:
        # Write the column headers to the new file
        header_df = pd.read_csv(combined_file, nrows=0)
        header_df.to_csv(new_fp, header=True, index=False)

        # Write the data in chunks to the new file
        df_iter_1 = pd.read_csv(combined_file, nrows=row_count // 2, chunksize=CSV_CHUNK_SIZE)
        df_iter_2 = pd.read_csv(combined_file, skiprows=range(1, row_count // 2 + 1), chunksize=CSV_CHUNK_SIZE)

        for chunk_1, chunk_2 in zip(df_iter_1, df_iter_2, strict=True):
            # Combine, shuffle, and write the chunks
            combined_chunks = pd.concat((chunk_1, chunk_2))
            combined_chunks = combined_chunks.sample(frac=1, random_state=NP_RAND_GEN).reset_index(drop=True)
            combined_chunks.to_csv(new_fp, header=False, index=False)

    os.remove(combined_file)
    return new_file

# %%
class PIIStateDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        # Obtain the preference matrix and unflatten preferences into n^2 x 2 for each iteration
        pref_matrix = slice_columns(df, "l0", f"r{N*N - 1}")
        pref_tensor = torch.from_numpy(pref_matrix.values).float()
        seq_prefs = pref_tensor.unflatten(1, (-1, 2))

        # Normalize preference ratings by dividing by n
        # seq_prefs.div_(n)

        # Add 5 zero columns (bit flags) to each l and r value of the preferences
        bit_flags_size = seq_prefs.size()[:-1] + torch.Size([5])
        self.seq_features = torch.cat((seq_prefs, torch.zeros(bit_flags_size)), dim=2)

        pairs_df = slice_columns(df, "matchIndices", "nm2Indices")
        flattened_seq_features = self.seq_features.flatten(0, 1)

        # TODO: Only need to check empty indices for nm2
        for pair_type_index, col_name in enumerate(pairs_df):
            col = df[col_name]
            col = col.str.strip("[]")
            empty_indices = col.index[col.apply(len) == 0]
            col = col.str.split(", ")
            col.iloc[empty_indices] = [[]] * len(empty_indices)
            col = col.apply(np.int64)

            indices = (col.index * N**2 + col)
            indices.drop(indices.index[empty_indices], inplace=True)

            flattened_pair_indices = indices.explode().to_numpy(np.int64)

            flattened_seq_features[flattened_pair_indices, pair_type_index + 2] = 1

        self.seq_features = flattened_seq_features.unflatten(0, (-1, N**2))

        # Convert the 0/1 converge labels to one hots
        converges = torch.tensor(df["converges"], dtype=torch.long)
        self.convergesOneHot = nn.functional.one_hot(converges, 2).float()

    def __len__(self):
        return len(self.convergesOneHot)

    def __getitem__(self, idx) -> tuple:
        return self.seq_features[idx], self.convergesOneHot[idx]

    def __getitems__(self, indices) -> list[tuple]:
        return [(self.seq_features[i], self.convergesOneHot[i]) for i in indices]

    def get_batch(self, indices) -> tuple:
        return self.seq_features[indices], self.convergesOneHot[indices]

# %%
class DataLoaderWrapper:
    def __init__(self, data_file: str, partition_size: int, batch_size: int, data_ratio: float, offset_ratio: float, is_training: bool):
        self.data_file = data_file
        self.batch_size = batch_size
        self.is_training = is_training

        with open(self.data_file, "rb") as f:
            full_data_len = sum(1 for _ in f) - 1

        if data_ratio + offset_ratio > 1:
            raise ValueError("provided data and offset ratios are out of bounds")

        # Get the appropriate percentage of the data
        self.data_len = int(data_ratio * full_data_len)
        self.start_row = int(offset_ratio * full_data_len)

        self.partition_size = min(partition_size, self.data_len)
        self.num_partitions = math.ceil(self.data_len / self.partition_size)

        os.makedirs("saved_transformer_partitions", exist_ok=True)
        self.train_test_str = "train" if self.is_training else "test"

        for i in range(self.num_partitions):
            dataset = self._create_partition(i)
            partition_file = f"saved_transformer_partitions/{JOB_ID}_{self.train_test_str}_{i}.ds"
            with open(partition_file, "wb", buffering=BUFFER_SIZE) as f:
                torch.save(dataset, f)

    def __len__(self):
        return math.ceil(self.partition_size / self.batch_size) * self.num_partitions

    def get_data_len(self):
        return self.data_len

    def get_iterator(self):
        if self.is_training:
            partition_order = NP_RAND_GEN.permutation(self.num_partitions)
        else:
            partition_order = np.arange(self.num_partitions)

        for partition_idx in partition_order:
            dataset = self._get_partition(partition_idx)

            if self.is_training:
                sample_order = NP_RAND_GEN.permutation(len(dataset))
            else:
                sample_order = np.arange(len(dataset))

            for batch_idx in range(len(dataset) // self.batch_size):
                sample_start_idx = batch_idx * self.batch_size
                sample_end_idx = min(sample_start_idx + self.batch_size, len(dataset))
                yield dataset.get_batch(sample_order[sample_start_idx : sample_end_idx])

            del dataset # TODO: Consider triggering garbage collection after this?

    def _get_partition(self, partition_idx):
        partition_file = f"saved_transformer_partitions/{JOB_ID}_{self.train_test_str}_{partition_idx}.ds"
        with open(partition_file, "rb", buffering=BUFFER_SIZE) as f:
            dataloader: PIIStateDataset = torch.load(f, weights_only=False)
        return dataloader

    def _create_partition(self, partition_idx):
        nrows = self.partition_size
        if partition_idx == self.num_partitions - 1:
            nrows = self.data_len - partition_idx * self.partition_size

        df = pd.read_csv(
            self.data_file,
            skiprows=range(1, self.start_row + self.partition_size * partition_idx + 1),
            nrows=nrows
        )
        df = df.reset_index(drop=True)

        return PIIStateDataset(df)

# %%
logging.info(f"Preprocessing n={N} data, of length {DATA_LEN}, at iteration index {ITER_INDEX}")

data_file_name = f"matrix_data/{DATA_NAME}"
preprocessed_file = f"{data_file_name}_combine_iter_{ITER_INDEX}.csv"

if not os.path.exists(preprocessed_file):
    logging.info(f"Sampling iteration index from data")
    sample_iter_file = preprocess_sampling(data_file_name, ITER_INDEX)
    logging.info(f"Joining iteration and trial data files")
    combined_file = preprocess_joining(data_file_name, sample_iter_file, ITER_INDEX)
    logging.info(f"Shuffling data")
    preprocessed_file = preprocess_shuffling(data_file_name, combined_file, ITER_INDEX)


dist_batch_size = BATCH_SIZE // WORLD_SIZE
dist_ratio = 1 / WORLD_SIZE

dist_train_ratio = TRAIN_RATIO * dist_ratio
dist_train_offset = RANK * dist_train_ratio

dist_test_ratio = TEST_RATIO * dist_ratio
dist_test_offset = TRAIN_RATIO + RANK * dist_test_ratio

logging.info(
    "Generating partitions using " +
    f"Train({dist_train_offset * 100:>0.1f}% -> +{dist_train_ratio * 100:>0.1f}%), " +
    f"Test({dist_test_offset * 100:>0.1f}% -> +{dist_test_ratio * 100:>0.1f}%)"
)

train_dataloader = DataLoaderWrapper(preprocessed_file, PARTITION_SIZE, dist_batch_size, dist_train_ratio, dist_train_offset, is_training=True)
test_dataloader = DataLoaderWrapper(preprocessed_file, PARTITION_SIZE, dist_batch_size, dist_test_ratio, dist_test_offset, is_training=False)

logging.info("Preprocessing done!")

# train_dataloader = CustomDataLoader(training_data, batch_size)
# test_dataloader = CustomDataLoader(test_data, batch_size)

# %%
# BENCHMARKING: Dataset
# df = pd.read_csv("matrix_data/matrixStateData_200000_20_combine_iter_0.csv")
# num_trials = 30

# start_wtime = timeit.default_timer()

# for _ in range(num_trials):
#     dataset = PIIStateDataset(df)

# end_wtime = timeit.default_timer()

# print(end_wtime - start_wtime)
# print((end_wtime - start_wtime) / num_trials)

# # ORIGINAL
# # 30x: 354.7033492710325
# # Avg: 11.823444975701083

# # Pandas string series to list series
# # 30x: 192.64214746496873
# # Avg: 6.421404915498957

# # Flatten -> unflatten
# # 10x: 54.93283691990655
# # Avg: 5.493283691990655

# # Only find empty indices once
# # 30x: 157.72345656598918
# # Avg: 5.257448552199639

# %%
# # BENCHMARKING: Dataloader (n=20, size=200000, partition_size=20000)
# num_trials = 30
# start_wtime = timeit.default_timer()

# for _ in range(num_trials):
#     for _ in train_dataloader.get_iterator():
#         pass

# end_wtime = timeit.default_timer()
# print(end_wtime - start_wtime)
# print((end_wtime - start_wtime) / num_trials)

# # ORIGINAL
# # 30x: 327.79994397005066
# # Avg: 10.926664799001689

# # Pre-generate and pickle
# # 30x: 25.349475977011025
# # Avg: 0.8449825325670342

# # Use custom get_batch over DataLoader
# # 30x: 18.63490589801222
# # Avg: 0.6211635299337407

# %%
# %reset_selective -f "(^model$|^loss_fn$|^optimizer$|^scheduler$)"

# %%
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# %%
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, dim):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_seq_len, dim)
        
    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).expand(x.size(0), -1)
        position_embeddings = self.position_embeddings(positions)
        return x + position_embeddings

# %%
class Sinusoidal2dPosEnc(nn.Module):
    # https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py

    def __init__(self, encoding_dim):
        super().__init__()
        self.encoding_dim = encoding_dim

    def forward(self, height, width, device):
        """
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        dim = self.encoding_dim

        if dim % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                            "odd dimension (got dim={:d})".format(dim))
        pe = torch.zeros(dim, height, width, device=device)
        # Each dimension use half of d_model
        dim = int(dim / 2)
        div_term = torch.exp(torch.arange(0., dim, 2, device=device) *
                            -(math.log(10000.0) / dim))
        pos_w = torch.arange(0., width, device=device).unsqueeze(1)
        pos_h = torch.arange(0., height, device=device).unsqueeze(1)
        pe[0:dim:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:dim:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[dim::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[dim + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

        return pe.permute([1,2,0])

# %%
class TransformerBlock(nn.Module):
    # https://github.com/LukeDitria/pytorch_tutorials/blob/main/section14_transformers/solutions/Pytorch1_Transformer_Text_Classification_Multi_Block.ipynb

    def __init__(self, hidden_size=128, num_heads=4):
        super().__init__()
        
        # Layer normalization for the input
        self.norm1 = nn.LayerNorm(hidden_size)
        
        # Multi-head self-attention mechanism
        self.multihead_attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, 
                                                    batch_first=True, dropout=0.25)
        
        # Layer normalization for the output of the attention mechanism
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # Feed-forward neural network layer
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),  # Linear transformation
            nn.LayerNorm(hidden_size),  # Layer normalization
            nn.ELU(),  # Activation function (ELU)
            nn.Linear(hidden_size, hidden_size)  # Linear transformation
        )

    def forward(self, x, key_padding_mask):
        # Layer normalization for the input
        norm_x = self.norm1(x)
        
        # Multi-head self-attention mechanism
        # [0] selects the attention output
        attn_output = self.multihead_attn(norm_x, 
                                          norm_x, 
                                          norm_x, 
                                          key_padding_mask=key_padding_mask)[0]
        
        # Residual connection and layer normalization for the attention output
        x = attn_output + x
        norm_x = self.norm2(x)
        
        # Feed-forward neural network layer
        mlp_output = self.mlp(norm_x)
        
        # Residual connection and output of the TransformerBlock
        output = mlp_output + x
        return output

# %%
class Transformer(nn.Module):
    # https://github.com/LukeDitria/pytorch_tutorials/blob/main/section14_transformers/solutions/Pytorch1_Transformer_Text_Classification_Multi_Block.ipynb

    """
    Transformer model consisting of an embedding layer, positional embeddings, 
    multiple Transformer blocks, and a final output layer.
    
    Args:
        input_size (int): Dimensionality of the input.
        output_size (int): Dimensionlogging.info("Training done!")ality of the output.
        hidden_size (int): Dimensionality of the hidden layers.
        num_layers (int): Number of Transformer blocks.
        num_heads (int): Number of attention heads.
    """
    def __init__(self, input_size, output_size, hidden_size=128, num_layers=3, num_heads=4):
        super(Transformer, self).__init__()

        self.embed_ff = nn.Linear(input_size, hidden_size)

        self.pos_emb = Sinusoidal2dPosEnc(hidden_size)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads) for _ in range(num_layers)
        ])

        self.out_vec = nn.Parameter(torch.zeros(1, 1, hidden_size))

        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq):
        """
        Forward pass through the Transformer model.
        
        Args:
            input_seq (Tensor): Input sequence tensor with shape (batch_size, sequence_length, feature_length).
        
        Returns:logging.info("Training done!")
            Tensor: Output tensor with shape (batch_size, output_size).
        """
        bs = input_seq.size(0)

        key_padding_mask = None

        input_embs = self.embed_ff(input_seq)

        # Add a unique embedding to each token embedding depending on its position in the sequence
        seq_len = torch.sqrt(torch.tensor(input_embs.size(1))).int().item()
        pos_emb = self.pos_emb(seq_len, seq_len, input_embs.get_device()).flatten(0,1)
        embs = input_embs + pos_emb

        # Concatenate a learnable output vector to the embeddings
        embs = torch.cat((self.out_vec.expand(bs, 1, -1), embs), dim=1)

        # Pass the embeddings through each Transformer block
        for block in self.blocks:
            embs = block(embs, key_padding_mask)

        # Pass the first embedding in the sequence to the final linear layer to get the output
        return self.fc_out(embs[:, 0])

# %%
# Set the random seed
torch.manual_seed(TORCH_RAND_SEED)

SEQ_FEAT_LENGTH = 7

# Create model
model = Transformer(SEQ_FEAT_LENGTH, output_size=OUTPUT_SIZE, hidden_size=HIDDEN_SIZE, 
                            num_layers=NUM_LAYERS, num_heads=NUM_HEADS).to(device)

ddp_model = DDP(model, device_ids=[device])

if WEIGHT_FILE is not None:
    model.load_state_dict(torch.load(WEIGHT_FILE))

logging.info(
f"""Created Transformer model
    output_size = {OUTPUT_SIZE}
    hidden_size = {HIDDEN_SIZE}
    num_layers = {NUM_LAYERS}
    num_heads = {NUM_HEADS}"""
)

# %%
# loss_fn = nn.CrossEntropyLoss()
# loss_fn = nn.BCELoss()
loss_fn = nn.BCEWithLogitsLoss()

# optimizer = torch.optim.SGD(ddp_model.parameters(), lr=learning_rate)
# optimizer = torch.optim.ASGD(ddp_model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(ddp_model.parameters(), lr=LEARNING_RATE)
# optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-3)

# %%
# Let's see how many Parameters our Model has!
num_params = sum(p.numel() for p in ddp_model.parameters())

logging.info("This model has %.1fk parameters!", num_params / 1e3)

# %%
def train_loop(dataloader: DataLoaderWrapper, model, loss_fn, optimizer):
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()

    data_len = dataloader.get_data_len()
    num_batches = len(dataloader)
    train_loss, correct = 0, 0

    for sequential_X, y in dataloader.get_iterator():
        sequential_X = sequential_X.to(device)
        y = y.to(device)

        # Compute prediction and loss
        pred = model(sequential_X)
        # pred = model(singletons_X, sequential_X)

        loss = loss_fn(pred, y)
        train_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        correct += pred.argmax(1).eq(y.argmax(1)).sum().item()

    correct /= data_len
    train_loss /= num_batches

    return train_loss, correct


def test_loop(dataloader: DataLoaderWrapper, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()

    data_len = dataloader.get_data_len()
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for sequential_X, y in dataloader.get_iterator():
            sequential_X = sequential_X.to(device)
            y = y.to(device)

            pred = model(sequential_X)
            # pred = model(singletons_X, sequential_X)

            loss = loss_fn(pred, y)
            test_loss += loss.item()

            correct += pred.argmax(1).eq(y.argmax(1)).sum().item()

    test_loss /= num_batches
    correct /= data_len

    # Adjust the learning rate with the scheduler
    # scheduler.step(test_loss)

    return test_loss, correct

# %%
def plot_data_to_csv(learning_rates, train_losses, test_losses, train_accuracies, test_accuracies, local_epoch, start_epoch):
    df = pd.DataFrame({
        "epochs": np.arange(start_epoch, start_epoch + local_epoch),
        "learning_rate": learning_rates[0:local_epoch],
        "train_loss": train_losses[0:local_epoch],
        "test_loss": test_losses[0:local_epoch],
        "train_accuracy": train_accuracies[0:local_epoch],
        "test_accuracy": test_accuracies[0:local_epoch],
    })

    df.to_csv(f"transformer_matrix_plot_data/{DATA_NAME}_iter{ITER_INDEX}_ID{JOB_ID}.csv", index=False)

def save_model(model: DDP, epoch: int, sub_directory: str):
    file_name = f"{DATA_NAME}_iter{ITER_INDEX}_ID{JOB_ID}_ep{epoch:04d}"
    torch.save(model.state_dict(), f"saved_transformer_models/{sub_directory}/{file_name}.pt")

def format_seconds(n):
    return str(datetime.timedelta(seconds=n))

# %%
if RANK == 0:
    learning_rates = np.zeros(EPOCHS, dtype=np.float32)
    train_losses, train_accuracies = np.zeros(EPOCHS, dtype=np.float32), np.zeros(EPOCHS, dtype=np.float32)
    test_losses, test_accuracies = np.zeros(EPOCHS, dtype=np.float32), np.zeros(EPOCHS, dtype=np.float32)

logging.info("Starting training...")
start_wtime = timeit.default_timer()

for local_epoch in range(EPOCHS):
    train_loss, train_accuracy = train_loop(train_dataloader, ddp_model, loss_fn, optimizer)
    test_loss, test_accuracy = test_loop(test_dataloader, ddp_model, loss_fn)

    # Sum the losses and accuracies in rank 0
    loss_acc_tensor = torch.tensor([train_loss, train_accuracy, test_loss, test_accuracy], dtype=torch.float)
    dist.reduce(loss_acc_tensor, dst=0, op=dist.ReduceOp.SUM)

    if RANK == 0:
        # Take average by dividing summed tensor
        loss_acc_tensor.div_(WORLD_SIZE)
        train_loss, train_accuracy, test_loss, test_accuracy = loss_acc_tensor

        # Epoch including start epoch
        global_epoch = START_EPOCH + local_epoch

        last_lr = scheduler.get_last_lr()
        learning_rates[local_epoch] = last_lr[0]       # pyright: ignore[reportPossiblyUnboundVariable]

        train_losses[local_epoch] = train_loss         # pyright: ignore[reportPossiblyUnboundVariable]
        train_accuracies[local_epoch] = train_accuracy # pyright: ignore[reportPossiblyUnboundVariable]
        test_losses[local_epoch] = test_loss           # pyright: ignore[reportPossiblyUnboundVariable]
        test_accuracies[local_epoch] = test_accuracy   # pyright: ignore[reportPossiblyUnboundVariable]

        if local_epoch % PRINT_FREQ == 0:
                logging.info(
                    f"Epoch {global_epoch + 1}  |   lr={last_lr}\n-------------------------------\n" +
                    f"Train Error: \n Accuracy: {(100*train_accuracy):>0.1f}%, Avg loss: {train_loss:>8f} \n\n" +
                    f"Test Error: \n Accuracy: {(100*test_accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n",
                )

        if (global_epoch + 1) % CHECKPOINT_FREQ == 0:
            plot_data_to_csv(learning_rates, train_losses, test_losses, train_accuracies, test_accuracies, local_epoch, START_EPOCH) # pyright: ignore[reportPossiblyUnboundVariable]
            save_model(ddp_model, global_epoch + 1, SAVE_MODEL_SUBDIR)

if RANK == 0:
    plot_data_to_csv(learning_rates, train_losses, test_losses, train_accuracies, test_accuracies, EPOCHS, START_EPOCH) # pyright: ignore[reportPossiblyUnboundVariable]
    save_model(ddp_model, START_EPOCH + EPOCHS, SAVE_MODEL_SUBDIR)

stop_wtime = timeit.default_timer()
total_wtime = round(stop_wtime - start_wtime)
wtime_per_epoch = round(total_wtime / EPOCHS)

logging.info(f"Training done in {format_seconds(total_wtime)}!")
logging.info(f"Average epoch runtime: {format_seconds(wtime_per_epoch)}")

# %%
def plot_data(
        x, ys: np.ndarray, labels: list[str] | None = None, title: str = "", ylabel: str = ""
    ) -> tuple:
    fig, ax = plt.subplots()

    if len(ys.shape) > 1:
        assert isinstance(labels, list)
        for y in ys:
            ax.plot(x, y)
    else:
        line = ax.plot(x, ys)
        ax.legend(handles=line)

    ax.set(xlabel="Epoch", ylabel=ylabel, title=title)
    ax.grid()

    if labels != None:
        ax.legend(labels)

    return fig, ax

# %%
if RANK == 0:
    x = np.arange(START_EPOCH, START_EPOCH + EPOCHS)

    losses = np.vstack((train_losses, test_losses))
    loss_fig, loss_ax = plot_data(x, losses, ["Training", "Testing"], f"Loss vs. Epoch ({DATA_NAME})", "Loss")
    plt.savefig(f"transformer_matrix_plots/{DATA_NAME}_iter{ITER_INDEX}_ID{JOB_ID}_loss")

    accuracies = np.vstack((train_accuracies, test_accuracies)) * 100
    acc_fig, acc_ax = plot_data(x, accuracies, ["Training", "Testing"], f"Accuracy vs. Epoch ({DATA_NAME})", "Accuracy (%)")
    plt.savefig(f"transformer_matrix_plots/{DATA_NAME}_iter{ITER_INDEX}_ID{JOB_ID}_acc")

    lr_fig, lr_ax = plot_data(x, learning_rates, title=f"Learning Rate vs. Epoch ({DATA_NAME})", ylabel="Learning Rate")
    # plt.savefig(f"transformer_matrix_plots/{data_name}_iter{iter_index}_ID{JOB_ID}_lr")

# %%
# Clean up process group
dist.destroy_process_group()


