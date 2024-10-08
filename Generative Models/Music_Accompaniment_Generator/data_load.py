import random

import torch
from miditok import REMI, TokenizerConfig
from utils import midi_to_array
from tqdm import tqdm
from train_parameters import max_len

# Our parameters
TOKENIZER_PARAMS = {
    "pitch_range": (21, 109),
    "beat_res": {(0, 4): 8, (4, 12): 4},
    "num_velocities": 32,
    "special_tokens": ["PAD", "BOS", "EOS"],
    "use_chords": True,
    "use_rests": False,
    "use_tempos": True,
    "use_programs": True,
    "num_tempos": 191,
    "tempo_range": (60, 250),
    "program_changes": True,
    "programs": [-1, 0, 24, 27, 30, 33, 36],
}
config = TokenizerConfig(**TOKENIZER_PARAMS)

# Creates the tokenizer
tokenizer = REMI(config)

word2idx = tokenizer.vocab
idx2word = {idx: word for idx, word in enumerate(word2idx)}
vocab_len = len(word2idx)


def data_load(data_type, split, data_len, x_folder, y_folder):
    print("---Data Load Start!---")
    x = []
    y = []
    data_range = (0, 1)
    if data_type == "train":
        data_range = range(0, split)
    if data_type == "test":
        data_range = range(split, data_len)
    for i in tqdm(data_range, desc="Data Loading...", unit="data"):
        x.append(
            midi_to_array(
                tokenizer=tokenizer,
                midifile=f"{x_folder}{i}.mid",
                max_len=max_len))
        y.append(
            midi_to_array(
                tokenizer=tokenizer,
                midifile=f"{y_folder}{i}.mid",
                max_len=max_len))
    x = torch.tensor(x)
    y = torch.tensor(y)
    print("---Data Load Completed!---")
    return x, y


def get_batch_indices(total_length, batch_size):
    assert (batch_size <=
            total_length), ('Batch size is large than total data length.'
                            'Check your data or change batch size.')
    current_index = 0
    indexes = [i for i in range(total_length)]
    random.shuffle(indexes)
    while True:
        if current_index + batch_size >= total_length:
            yield indexes[current_index:total_length], current_index
            break
        yield indexes[current_index:current_index + batch_size], current_index
        current_index += batch_size
