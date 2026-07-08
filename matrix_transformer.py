# %%
import torch
# import torch.accelerator
from torch import nn
from torch.nn.utils.rnn import pack_sequence, unpack_sequence, PackedSequence
from torch.utils.data import Dataset, DataLoader

import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import sys
import pandas as pd
from typing import Callable

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
# TODO: Change this to read from a file / environment variables / etc

# --- HYPER PARAMETERS ---
batch_size = 64
learning_rate = 1e-3
epochs = 650

# --- DATASET PARAMETERS ---
n = 10
data_len = 2000000
iter_index = 0
data_name = f"matrixStateData_{data_len}_{n}"

# Scalar multiple for ratios to get 200k exactly divisible by 64: 0.99968
train_ratio = 0.75
test_ratio = 0.25

shuffle_rand_seed = 1


# --- MODEL HYPER PARAMETERS ---
output_size = 2
hidden_size = 64
num_layers = 4
num_heads = 8

# Parameters for saved weights
start_epoch = 250
# weight_file = None
weight_file_job_id = 105100
weight_file = f"savedModels/{data_name}_iter{iter_index}_hs{hidden_size}/{data_name}_iter{iter_index}_ID{weight_file_job_id}_ep{start_epoch:04d}.pt"


# --- TRAINING PARAMETERS ---
print_freq = 1
checkpoint_freq = 10

# %%
# Save all hyper parameters to a JSON file
with open(f"savedModels/{data_name}_iter{iter_index}_ID{JOB_ID}.json", "w") as f:
    json.dump(
        {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            
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
class PIIStateDataset(Dataset):
    def __init__(self, iter_df: pd.DataFrame, trial_df: pd.DataFrame, iter_index: int, data_ratio: float, offset_ratio: float) -> None:
        if data_ratio + offset_ratio > 1:
            raise ValueError("provided data and offset ratios are out of bounds")

        # Get the appropriate percentage of the data
        start_row = int(offset_ratio * len(iter_df))
        end_row = int((data_ratio + offset_ratio) * len(iter_df))

        iter_df = iter_df.iloc[start_row : end_row]

        # Join iteration data with trial data based on the unique program+trial index
        df = pd.merge(iter_df, trial_df, on=["programIndex", "trialIndex"])

        # Obtain singleton features from each iteration
        self.singleton_feature_dict = {}
        for col in slice_columns(df, "numUnstable", "avgCycleLen").columns:
            self.singleton_feature_dict[col] = torch.tensor(df[col], dtype=torch.float)

        # Normalize (dividing by n or n^2) and stack features together
        self.singleton_feature_dict["numUnstable"].div_(n)
        self.singleton_features = torch.stack(tuple(self.singleton_feature_dict.values()), dim=1)
        self.singleton_features.div_(n)

        # Obtain the preference matrix and unflatten preferences into n^2 x 2 for each iteration
        pref_matrix = slice_columns(df, "l0", f"r{n*n - 1}")
        pref_tensor = torch.from_numpy(pref_matrix.values).float()
        seq_prefs = pref_tensor.unflatten(1, (-1, 2))

        # Normalize preference ratings by dividing by n
        # seq_prefs.div_(n)

        # Add 5 zero columns (bit flags) to each l and r value of the preferences
        bit_flags_size = seq_prefs.size()[:-1] + torch.Size([5])
        self.seq_features = torch.cat((seq_prefs, torch.zeros(bit_flags_size)), dim=2)

        # For each iteration, get each list of indices,
        # and set the flag to 1 at the respective column
        for row, pair_indices_lists in slice_columns(df, "matchIndices", "nm2Indices").iterrows():
            assert isinstance(row, int)
            for col, pair_indices_strs in enumerate(pair_indices_lists):
                pair_indices: list[int] = list_from_str(pair_indices_strs, int)
                self.seq_features[row, pair_indices, col + 2] = 1

        # Convert the 0/1 converge labels to one hots
        self.converges = torch.tensor(df["converges"], dtype=torch.long)
        self.convergesOneHot = nn.functional.one_hot(self.converges, 2).float()

    def __len__(self):
        return len(self.converges)

    def __getitem__(self, idx) -> tuple:
        X = (self.singleton_features[idx], self.seq_features[idx])

        return X, self.convergesOneHot[idx]

# %%
class CustomDataLoader:
    def __init__(self, dataset: PIIStateDataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)
    
    def get_iterator(self):
        return self._custom_data_loader()

    def _custom_data_loader(self):
        for start_idx in range(0, len(self.dataset), self.batch_size):
            yield self._get_batch(start_idx)

    def _get_batch(self, start_idx: int):
        # batch_singletons = torch.zeros((batch_size, dataset[0][0][0].size(0)), dtype=dataset[0][0][0].dtype)
        # batch_y = torch.zeros((batch_size, dataset[0][1].size(0)), dtype=dataset[0][1].dtype)
        batch_singletons = []
        batch_y = []
        batch_mean_lists = []

        end_idx = min(start_idx + self.batch_size, len(self.dataset))
        for idx in range(start_idx, end_idx):
            (singletons, mean_lists), y = self.dataset[idx]
            batch_singletons.append(singletons)
            batch_y.append(y)
            batch_mean_lists.append(mean_lists)

        batch_singletons = torch.stack(batch_singletons, dim=0)
        batch_y = torch.stack(batch_y, dim=0)

        # 4-tuple of batch length list of 2D tensors
        batch_means = unzip(batch_mean_lists)
        packed_batch_means = []

        for means in batch_means:
            packed_batch_means.append(pack_sequence(means, enforce_sorted=False))

        return (batch_singletons, packed_batch_means), batch_y

# %%
logging.info(f"Importing n={n} data, of length {data_len}, at iteration index {iter_index}")

# Sample random single iterations from the iteration data
iter_df = pd.read_csv(f"matrix_data/{data_name}_iter.csv")
iter_df = iter_df.sample(frac=1, random_state=shuffle_rand_seed).reset_index(drop=True)
iter_df = sample_iter_df(iter_df, iter_index)

trial_df = pd.read_csv(f"matrix_data/{data_name}_trial.csv")

training_data = PIIStateDataset(
    iter_df, trial_df,
    iter_index=iter_index,
    data_ratio=train_ratio,
    offset_ratio=0
)

test_data = PIIStateDataset(
    iter_df, trial_df,
    iter_index=iter_index,
    data_ratio=test_ratio,
    offset_ratio=train_ratio
)

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# train_dataloader = CustomDataLoader(training_data, batch_size)
# test_dataloader = CustomDataLoader(test_data, batch_size)


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
        input_size (int): TODO.
        output_size (int): Dimensionality of the output.
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
        
        Returns:
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
(singleton_feats, seq_feats) ,_y = training_data[0]

# Create model
model = Transformer(seq_feats.size(1), output_size=output_size, hidden_size=hidden_size, 
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
def train_loop(dataloader, model, loss_fn, optimizer):
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()

    data_len = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss, correct = 0, 0

    for (singletons_X, sequential_X), y in dataloader:
        singletons_X = singletons_X.to(device)
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


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()

    data_len = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for (singletons_X, sequential_X), y in dataloader:
            singletons_X = singletons_X.to(device)
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

    df.to_csv(f"transformerPlotData/{data_name}_iter{iter_index}_ID{JOB_ID}.csv", index=False)

# %%
logging.info("Starting training...")

learning_rates = np.zeros(epochs, dtype=np.float32)
train_losses, train_accuracies = np.zeros(epochs, dtype=np.float32), np.zeros(epochs, dtype=np.float32)
test_losses, test_accuracies = np.zeros(epochs, dtype=np.float32), np.zeros(epochs, dtype=np.float32)

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
        torch.save(model.state_dict(), f"savedModels/{data_name}_iter{iter_index}_ID{JOB_ID}_ep{global_epoch + 1:04d}.pt")

plot_data_to_csv(learning_rates, train_losses, test_losses, train_accuracies, test_accuracies, epochs, start_epoch)
torch.save(model.state_dict(), f"savedModels/{data_name}_iter{iter_index}_ID{JOB_ID}_ep{start_epoch + epochs:04d}.pt")
logging.info("Training done!")

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
plt.savefig(f"transformerPlots/{data_name}_iter{iter_index}_ID{JOB_ID}_loss")

accuracies = np.vstack((train_accuracies, test_accuracies)) * 100
acc_fig, acc_ax = plot_data(x, accuracies, ["Training", "Testing"], f"Accuracy vs. Epoch ({data_name})", "Accuracy (%)")
plt.savefig(f"transformerPlots/{data_name}_iter{iter_index}_ID{JOB_ID}_acc")

lr_fig, lr_acc = plot_data(x, learning_rates, title=f"Learning Rate vs. Epoch ({data_name})", ylabel="Learning Rate")
# plt.savefig(f"transformerPlots/{data_name}_iter{iter_index}_ID{JOB_ID}_lr")


