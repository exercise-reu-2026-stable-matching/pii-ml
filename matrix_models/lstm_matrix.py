# %%


# %%
import torch
# import torch.accelerator
from torch import nn
from torch.nn.utils.rnn import pack_sequence, unpack_sequence, PackedSequence
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import pandas as pd
from typing import Callable

# %%
try:
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
except AttributeError:
    device = "cpu"
device

# %%
# Hyper parameters
batch_size = 64
learning_rate = 1e-3
epochs = 100

n = 10

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
data_len = 2000000
iter_index = 1

data_name = f"matrixStateData_{data_len}_{n}"

# Scalar multiple for ratios to get 200k exactly divisible by 64: 0.99968
train_ratio = 0.75
test_ratio = 0.25

# Sample random single iterations from the iteration data
rand_seed = 1
iter_df = pd.read_csv(f"matrix_data/{data_name}_iter.csv")
iter_df = iter_df.sample(frac=1, random_state=rand_seed).reset_index(drop=True)
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
# training_data[0][0].shape

print(len(training_data), len(test_data))

# %%
print(
    torch.stack([y for _, y in training_data])[:, 1].sum().item() / len(training_data) * 100,
    torch.stack([y for _, y in test_data])[:, 1].sum().item() / len(test_data) * 100,
)

# %%
for i in range(2):
    print(i)
    print(training_data[i])

# %%
for i in range(2):
    print(i)
    print(test_data[i])

# %%
next(iter(train_dataloader))

# %%

# %%
"""
The following class `Sinudosial2dPosEnc` was modified, with permission,
from code by Zelun Wang and Jyh-Charn Liu as a part of the following publication:
Wang, Zelun, and Jyh-Charn Liu.
"Translating math formula images to LaTeX sequences using deep neural networks with sequence-level training."
International Journal on Document Analysis and Recognition (IJDAR) (2020): 1-13.

Orginal source: https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py
"""

class Sinusoidal2dPosEnc(nn.Module):
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
"""
More complicated things to try:
- layer norm / batch norm / other norm / divide by n
- bidirectional LSTM
- linear layer *before* / after
- positional encoding
- other embedding of the "words"
- dropout
"""

class LSTMModel(nn.Module):
    def __init__(
            self, input_size: int, output_size: int, hidden_size=64, num_layers=1, bidirectional=False, batch_first=True
        ) -> None:
        super().__init__()

        D = 2 if bidirectional else 1

        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=bidirectional, dropout=0.2, batch_first=batch_first)
        # self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers, bidirectional=bidirectional, batch_first=batch_first)
        self.ff_stack = nn.Sequential(
            nn.Linear(D * hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, output_size),
        )

    def forward(self, x):
        out, _ = self.lstm1(x)
        # out, _ = self.lstm2(out)

        out = out[:, -1, :]
        out = self.ff_stack(out)

        return out

# %%
class HybridModel(nn.Module):
    def __init__(
        self, singleton_input_size: int, seq_input_size: int, output_size: int, hidden_size=64, num_layers=1, dropout=0., bidirectional=False, batch_first=True
    ):
        super().__init__()
        self.batch_first = batch_first

        # if batch_first:
        #     self.seq_layer_norm = nn.LayerNorm([n**2, seq_input_size])
        # else:
        #     self.seq_layer_norm = nn.LayerNorm([batch_size, seq_input_size])

        self.pre_ff = nn.Sequential(
            # nn.Linear(seq_input_size, hidden_size),
            nn.Linear(seq_input_size, 16),
            # nn.ReLU(),
            # nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, hidden_size),
        )

        self.positional_encoding = Sinusoidal2dPosEnc(hidden_size)
        # self.positional_encoding = LearnedPositionalEncoding(n**2, hidden_size)

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=batch_first)

        self.ff_stack = nn.Sequential(
            # nn.LayerNorm(singleton_input_size + hidden_size),

            # nn.Linear(singleton_input_size + hidden_size, 16),
            # nn.ReLU(),
            # # nn.Linear(16, 16),
            # # nn.ReLU(),
            # nn.Linear(16, output_size),
            nn.Linear(singleton_input_size + hidden_size, output_size),
        )

    def forward(self, singletons_x, sequential_x):
        # ln_sequential_x = self.seq_layer_norm(sequential_x)

        ff_sequential_x = self.pre_ff(sequential_x)

        seq_dim = 1 if self.batch_first else 0
        seq_len = torch.sqrt(torch.tensor(ff_sequential_x.size(seq_dim))).int().item()

        pe = self.positional_encoding(seq_len, seq_len, ff_sequential_x.get_device()).flatten(0,1)
        # pe_sequential_x = self.positional_encoding(ff_sequential_x)
        pe_sequential_x = ff_sequential_x + pe
        # pe_sequential_x = torch.cat((sequential_x, pe.repeat(sequential_x.size(0), 1, 1)), dim=2)

        sequential_out, _ = self.lstm(pe_sequential_x)
        sequential_out = sequential_out[:, -1, :]

        combined_x = torch.cat((singletons_x, sequential_out), dim=1)
        out = self.ff_stack(combined_x)

        return out

# %%
(singleton_feats, seq_feats) ,_y = training_data[0]

bidirectional = False
bi_str = "_bi" if bidirectional else ""

# model = LSTMModel(seq_feats.size(1), 2, 16, 1, bidirectional).to(device)
model = HybridModel(singleton_feats.size(0), seq_feats.size(1), 2, 64, 8, 0.25, bidirectional, True).to(device)

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
num_model_params = 0
for param in model.parameters():
    num_model_params += param.flatten().shape[0]

print("-This Model Has %d (Approximately %d Thousand) Parameters!" % (num_model_params, num_model_params//1e3))

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
        # pred = model(sequential_X)
        pred = model(singletons_X, sequential_X)

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

            # pred = model(sequential_X)
            pred = model(singletons_X, sequential_X)

            loss = loss_fn(pred, y)
            test_loss += loss.item()

            correct += pred.argmax(1).eq(y.argmax(1)).sum().item()


    test_loss /= num_batches
    correct /= data_len

    # Adjust the learning rate with the scheduler
    scheduler.step(test_loss)

    return test_loss, correct

# %%
def plot_data_to_csv(learning_rates, train_losses, test_losses, train_accuracies, test_accuracies, epoch):
    df = pd.DataFrame({
        "learning_rate": learning_rates[0:epoch],
        "train_loss": train_losses[0:epoch],
        "test_loss": test_losses[0:epoch],
        "train_accuracy": train_accuracies[0:epoch],
        "test_accuracy": test_accuracies[0:epoch],
    })

    df.to_csv(f"lstm_matrix_plot_data/{data_name}_iter{iter_index}{bi_str}_acc_hybrid.csv", index=False)

# %%
learning_rates = np.zeros(epochs, dtype=np.float32)
train_losses, train_accuracies = np.zeros(epochs, dtype=np.float32), np.zeros(epochs, dtype=np.float32)
test_losses, test_accuracies = np.zeros(epochs, dtype=np.float32), np.zeros(epochs, dtype=np.float32)

for t in range(epochs):
    last_lr = scheduler.get_last_lr()
    learning_rates[t] = last_lr[0]

    train_loss, train_accuracy = train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loss, test_accuracy = test_loop(test_dataloader, model, loss_fn)

    train_losses[t] = train_loss
    train_accuracies[t] = train_accuracy
    test_losses[t] = test_loss
    test_accuracies[t] = test_accuracy

    if t % 1 == 0:
        print(f"Epoch {t+1}  |   lr={last_lr}\n-------------------------------")
        print(f"Train Error: \n Accuracy: {(100*train_accuracy):>0.1f}%, Avg loss: {train_loss:>8f} \n")
        print(f"Test Error: \n Accuracy: {(100*test_accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    if t % 1000 == 0:
        plot_data_to_csv(learning_rates, train_losses, test_losses, train_accuracies, test_accuracies, t)

plot_data_to_csv(learning_rates, train_losses, test_losses, train_accuracies, test_accuracies, epochs)
print("Done!")

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
x = np.arange(0, epochs)

losses = np.vstack((train_losses, test_losses))
loss_fig, loss_ax = plot_data(x, losses, ["Training", "Testing"], f"Loss vs. Epoch ({data_name})", "Loss")
plt.savefig(f"lstm_matrix_plots/{data_name}_iter{iter_index}{bi_str}_hybrid_loss")

accuracies = np.vstack((train_accuracies, test_accuracies)) * 100
acc_fig, acc_ax = plot_data(x, accuracies, ["Training", "Testing"], f"Accuracy vs. Epoch ({data_name})", "Accuracy (%)")
plt.savefig(f"lstm_matrix_plots/{data_name}_iter{iter_index}{bi_str}_hybrid_acc")

lr_fig, lr_acc = plot_data(x, learning_rates, title=f"Learning Rate vs. Epoch ({data_name})", ylabel="Learning Rate")
# plt.savefig(f"lstm_matrix_plots/{data_name}_iter{iter_index}{bi_str}_hybrid_lr")


