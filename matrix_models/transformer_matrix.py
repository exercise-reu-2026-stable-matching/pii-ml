# %%
# %reset -f

# %%
import torch
# import torch.accelerator
from torch import nn
from torch.nn.utils.rnn import pack_sequence, unpack_sequence, PackedSequence
from torch.utils.data import Dataset, DataLoader

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
    # sys.argv = ["transformer_matrix.ipynb"]
    sys.argv = ["transformer_matrix.ipynb", "transformer_configs/matrixStateData_2000000_10_iter0_hs64.json"]
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
    batch_size = config["batch_size"]
    learning_rate = config["learning_rate"]
    epochs = config["epochs"]
    start_epoch = config["start_epoch"]


    # --- DATA PREPROCESSING PARAMETERS ---
    BUFFER_SIZE:    int = config.get("buffer_size", 2 << 19)
    CSV_CHUNK_SIZE: int = config.get("csv_chunk_size", 1000000)
    PARTITION_SIZE: int = config["partition_size"]


    # --- DATASET PARAMETERS ---
    n = config["n"]
    data_len = config["data_len"]
    iter_index = config["iter_index"]
    data_name = config.get("data_name", f"matrixStateData_{data_len}_{n}")

    # Scalar multiple for ratios to get 200k exactly divisible by 64: 0.99968
    train_ratio = config["train_ratio"]
    test_ratio = config["test_ratio"]

    shuffle_rand_seed = config["shuffle_rand_seed"]


    # --- MODEL HYPER PARAMETERS ---
    output_size = config["output_size"]
    hidden_size = config["hidden_size"]
    num_layers = config["num_layers"]
    num_heads = config["num_heads"]
    save_model_subdir = config.get("save_model_subdir", f"{data_name}_iter{iter_index}_hs{hidden_size}")
    torch_rand_seed = config.get("torch_rand_seed", torch.seed())

    # Parameters for saved weights
    weight_file_job_id = config.get("weight_file_job_id")
    weight_file = config.get("weight_file")

    if weight_file is None and weight_file_job_id is not None:
        weight_file = f"saved_transformer_models/{data_name}_iter{iter_index}_hs{hidden_size}/{data_name}_iter{iter_index}_ID{weight_file_job_id}_ep{start_epoch:04d}.pt"

    # --- TRAINING PARAMETERS ---
    print_freq = config["print_freq"]
    checkpoint_freq = config["checkpoint_freq"]

# OTHERWISE, use hard-coded parameters
else:
    # --- HYPER PARAMETERS ---
    batch_size = 64
    learning_rate = 1e-3
    epochs = 25
    start_epoch = 0


    # --- DATA PREPROCESSING PARAMETERS ---
    BUFFER_SIZE    = 2 << 19 # 1 MB
    CSV_CHUNK_SIZE = 1000000
    PARTITION_SIZE = 200000


    # --- DATASET PARAMETERS ---
    n = 20
    data_len = 20000
    iter_index = 0
    data_name = f"matrixStateData_{data_len}_{n}"

    # Scalar multiple for ratios to get 200k exactly divisible by 64: 0.99968
    train_ratio = 0.75
    test_ratio = 0.25

    shuffle_rand_seed = 1

    # --- MODEL HYPER PARAMETERS ---
    output_size = 2
    hidden_size = 128
    num_layers = 4
    num_heads = 8
    save_model_subdir = f"{data_name}_iter{iter_index}_hs{hidden_size}"
    torch_rand_seed = torch.seed()

    # Parameters for saved weights
    weight_file = None
    weight_file_job_id = "x"
    weight_file = f"saved_transformer_models/{data_name}_iter{iter_index}_hs{hidden_size}/{data_name}_iter{iter_index}_ID{weight_file_job_id}_ep{start_epoch:04d}.pt"


    # --- TRAINING PARAMETERS ---
    print_freq = 1
    checkpoint_freq = 10

NP_RAND_GEN = np.random.default_rng(shuffle_rand_seed)

# %%
os.makedirs(os.path.join("saved_transformer_models", save_model_subdir), exist_ok=True)

# Save all hyper parameters to a JSON file
with open(f"saved_transformer_models/{save_model_subdir}/{data_name}_iter{iter_index}_ID{JOB_ID}.json", "w") as f:
    json.dump(
        {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            
            "buffer_size": BUFFER_SIZE,
            "csv_chunk_size": CSV_CHUNK_SIZE,
            "partition_size": PARTITION_SIZE,

            "n": n,
            "data_len": data_len,
            "iter_index": iter_index,
            "data_name": data_name,
            "train_ratio": train_ratio,
            "test_ratio": test_ratio,
            "shuffle_rand_seed": shuffle_rand_seed,

            "output_size": output_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "save_model_subdir": save_model_subdir,
            "torch_rand_seed": torch_rand_seed,
            "start_epoch": start_epoch,
            "weight_file_job_id": weight_file_job_id,
            "weight_file": weight_file,

            "print_freq": print_freq,
            "checkpoint_freq": checkpoint_freq,
        },
        f
    )

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

    if not os.path.exists(new_file):
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

    if not os.path.exists(new_file):
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

    if os.path.exists(sample_iter_file):
        os.remove(sample_iter_file)
    return new_file

# %%
def preprocess_shuffling(data_file_name: str, combined_file: str, iter_index: int):
    new_file = f"{data_file_name}_combine_iter_{iter_index}.csv"

    with open(combined_file, "rb") as f:
        row_count = sum(1 for _ in f) - 1

    if not os.path.exists(new_file):
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

    if os.path.exists(combined_file):
        os.remove(combined_file)
    return new_file

# %%
class PIIStateDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        # Obtain the preference matrix and unflatten preferences into n^2 x 2 for each iteration
        pref_matrix = slice_columns(df, "l0", f"r{n*n - 1}")
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

            indices = (col.index * n**2 + col)
            indices.drop(indices.index[empty_indices], inplace=True)

            flattened_pair_indices = indices.explode().to_numpy(np.int64)

            flattened_seq_features[flattened_pair_indices, pair_type_index + 2] = 1

        self.seq_features = flattened_seq_features.unflatten(0, (-1, n**2))

        # Convert the 0/1 converge labels to one hots
        converges = torch.tensor(df["converges"], dtype=torch.long)
        self.convergesOneHot = nn.functional.one_hot(converges, 2).float()

    def __len__(self):
        return len(self.convergesOneHot)

    def __getitem__(self, idx) -> tuple:
        return self.seq_features[idx], self.convergesOneHot[idx]

    def __getitems__(self, indices) -> list[tuple]:
        return [(self.seq_features[i], self.convergesOneHot[i]) for i in indices]

    # TODO: Try using this over dataloader
    def get_batch(self, indices) -> tuple:
        return self.seq_features[indices], self.convergesOneHot[indices]

# %%
class DataLoaderWrapper:
    def __init__(self, data_file: str, partition_size: int, batch_size: int, data_ratio: float, offset_ratio: float, shuffle: bool):
        self.data_file = data_file
        self.batch_size = batch_size
        self.shuffle = shuffle

        with open(self.data_file, "rb") as f:
            full_data_len = sum(1 for _ in f) - 1

        if data_ratio + offset_ratio > 1:
            raise ValueError("provided data and offset ratios are out of bounds")

        # Get the appropriate percentage of the data
        self.data_len = int(data_ratio * full_data_len)
        self.start_row = int(offset_ratio * full_data_len)

        self.partition_size = min(partition_size, self.data_len)
        self.num_partitions = math.ceil(self.data_len / self.partition_size)

    def __len__(self):
        return math.ceil(self.partition_size / self.batch_size) * self.num_partitions

    def get_data_len(self):
        return self.data_len

    def get_iterator(self):
        partition_order = NP_RAND_GEN.permutation(self.num_partitions)

        for partition_idx in partition_order:
            data_loader = self._get_partition(partition_idx)
            # return data_loader
            for batch in data_loader:
                yield batch
            del data_loader # TODO: Consider triggering garbage collection after this?

    def _get_partition(self, partition_idx):
        nrows = self.partition_size
        if partition_idx == self.num_partitions - 1:
            nrows = self.data_len - partition_idx * self.partition_size

        df = pd.read_csv(
            self.data_file,
            skiprows=range(1, self.start_row + self.partition_size * partition_idx + 1),
            nrows=nrows
        )
        df = df.reset_index(drop=True)

        dataset = PIIStateDataset(df)
        # display(dataset[0:65])
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)

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
logging.info(f"Preprocessing n={n} data, of length {data_len}, at iteration index {iter_index}")

data_file_name = f"matrix_data/{data_name}"
preprocessed_file = f"{data_file_name}_combine_iter_{iter_index}.csv"

if not os.path.exists(preprocessed_file):
    logging.info(f"Sampling iteration index from data")
    sample_iter_file = preprocess_sampling(data_file_name, iter_index)
    logging.info(f"Joining iteration and trial data files")
    combined_file = preprocess_joining(data_file_name, sample_iter_file, iter_index)
    logging.info(f"Shuffling data")
    preprocessed_file = preprocess_shuffling(data_file_name, combined_file, iter_index)

train_dataloader = DataLoaderWrapper(preprocessed_file, PARTITION_SIZE, batch_size, train_ratio, 0, shuffle=True)
test_dataloader = DataLoaderWrapper(preprocessed_file, PARTITION_SIZE, batch_size, test_ratio, train_ratio, shuffle=False)

logging.info(f"Preprocessing done!")

# train_dataloader = CustomDataLoader(training_data, batch_size)
# test_dataloader = CustomDataLoader(test_data, batch_size)

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
torch.manual_seed(torch_rand_seed)

SEQ_FEAT_LENGTH = 7

# Create model
model = Transformer(SEQ_FEAT_LENGTH, output_size=output_size, hidden_size=hidden_size, 
                            num_layers=num_layers, num_heads=num_heads).to(device)

if weight_file is not None:
    model.load_state_dict(torch.load(weight_file))

logging.info(
f"""Created Transformer model
    output_size = {output_size}
    hidden_size = {hidden_size}
    num_layers = {num_layers}
    num_heads = {num_heads}"""
)

# %%
# loss_fn = nn.CrossEntropyLoss()
# loss_fn = nn.BCELoss()
loss_fn = nn.BCEWithLogitsLoss()

# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.ASGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-3)

# %%
# Let's see how many Parameters our Model has!
num_params = sum(p.numel() for p in model.parameters())

logging.info("This model has %.1fk parameters!", num_params / 1e3)

# %%
def train_loop(dataloader: DataLoaderWrapper, model, loss_fn, optimizer):
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()

    data_len = dataloader.get_data_len()
    num_batches = len(dataloader)
    train_loss, correct = 0, 0

    batch_num = 0
    data_num = 0
    for sequential_X, y in dataloader.get_iterator():
        batch_num += 1
        data_num += sequential_X.size(0)

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
    scheduler.step(test_loss)

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

    df.to_csv(f"transformer_matrix_plot_data/{data_name}_iter{iter_index}_ID{JOB_ID}.csv", index=False)

def save_model(epoch: int, sub_directory: str):
    file_name = f"{data_name}_iter{iter_index}_ID{JOB_ID}_ep{epoch:04d}"
    torch.save(model.state_dict(), f"saved_transformer_models/{sub_directory}/{file_name}.pt")

def format_seconds(n):
    return str(datetime.timedelta(seconds=n))

# %%
learning_rates = np.zeros(epochs, dtype=np.float32)
train_losses, train_accuracies = np.zeros(epochs, dtype=np.float32), np.zeros(epochs, dtype=np.float32)
test_losses, test_accuracies = np.zeros(epochs, dtype=np.float32), np.zeros(epochs, dtype=np.float32)

logging.info("Starting training...")
start_wtime = timeit.default_timer()

for local_epoch in range(epochs):
    # Epoch including start epoch
    global_epoch = start_epoch + local_epoch

    last_lr = scheduler.get_last_lr()
    learning_rates[local_epoch] = last_lr[0]

    train_loss, train_accuracy = train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loss, test_accuracy = test_loop(test_dataloader, model, loss_fn)

    train_losses[local_epoch] = train_loss
    train_accuracies[local_epoch] = train_accuracy
    test_losses[local_epoch] = test_loss
    test_accuracies[local_epoch] = test_accuracy

    if local_epoch % print_freq == 0:
        logging.info(
            f"Epoch {global_epoch + 1}  |   lr={last_lr}\n-------------------------------\n" +
            f"Train Error: \n Accuracy: {(100*train_accuracy):>0.1f}%, Avg loss: {train_loss:>8f} \n\n" +
            f"Test Error: \n Accuracy: {(100*test_accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n",
        )

    if (global_epoch + 1) % checkpoint_freq == 0:
        plot_data_to_csv(learning_rates, train_losses, test_losses, train_accuracies, test_accuracies, local_epoch, start_epoch)
        save_model(global_epoch + 1, save_model_subdir)

plot_data_to_csv(learning_rates, train_losses, test_losses, train_accuracies, test_accuracies, epochs, start_epoch)
save_model(start_epoch + epochs, save_model_subdir)

stop_wtime = timeit.default_timer()
total_wtime = round(stop_wtime - start_wtime)
wtime_per_epoch = round(total_wtime / epochs)

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
x = np.arange(start_epoch, start_epoch + epochs)

losses = np.vstack((train_losses, test_losses))
loss_fig, loss_ax = plot_data(x, losses, ["Training", "Testing"], f"Loss vs. Epoch ({data_name})", "Loss")
plt.savefig(f"transformer_matrix_plots/{data_name}_iter{iter_index}_ID{JOB_ID}_loss")

accuracies = np.vstack((train_accuracies, test_accuracies)) * 100
acc_fig, acc_ax = plot_data(x, accuracies, ["Training", "Testing"], f"Accuracy vs. Epoch ({data_name})", "Accuracy (%)")
plt.savefig(f"transformer_matrix_plots/{data_name}_iter{iter_index}_ID{JOB_ID}_acc")

lr_fig, lr_acc = plot_data(x, learning_rates, title=f"Learning Rate vs. Epoch ({data_name})", ylabel="Learning Rate")
# plt.savefig(f"transformer_matrix_plots/{data_name}_iter{iter_index}_ID{JOB_ID}_lr")


