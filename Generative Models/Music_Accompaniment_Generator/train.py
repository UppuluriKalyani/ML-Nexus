from transformer import Transformer
from data_load import vocab_len
from train_model import train_model
from train_parameters import (max_len, batch_size, lr, d_model, d_ff, n_layers,
                              heads, dropout_rate, n_epochs, PAD_ID, device,
                              print_interval, data_split_rate, len_Dataset)


if __name__ == '__main__':
    instruments = ['Drum', 'Bass', 'Guitar', 'Piano']
    for instrument in instruments:
        model = Transformer(src_vocab_size=vocab_len, dst_vocab_size=vocab_len, pad_idx=PAD_ID, d_model=d_model,
                            d_ff=d_ff, n_layers=n_layers, heads=heads, dropout=dropout_rate, max_seq_len=max_len)
        train_model(
            model=model,
            data_split_rate=data_split_rate,
            data_len=len_Dataset[instrument],
            batch_size=batch_size,
            lr=lr,
            n_epochs=n_epochs,
            PAD_ID=PAD_ID,
            device=device,
            print_interval=print_interval,
            instrument=instrument)
