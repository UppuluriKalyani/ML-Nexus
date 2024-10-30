from utils import merge_midi_tracks, generate_midi_v2
from data_load import tokenizer
from transformer import Transformer
from data_load import vocab_len
import torch

from train_parameters import (max_len, d_model, d_ff, n_layers,
                              heads, dropout_rate, PAD_ID)

if __name__ == '__main__':
    instruments = ['Drum', 'Bass', 'Guitar', 'Piano']
    src_midi = "./HMuseData/Melody2Drum/Melody/69.mid"
    for instrument in instruments:
        print(f"-----Loading {instrument} model-----")
        model = Transformer(src_vocab_size=vocab_len, dst_vocab_size=vocab_len, pad_idx=PAD_ID, d_model=d_model,
                            d_ff=d_ff, n_layers=n_layers, heads=heads, dropout=dropout_rate, max_seq_len=max_len)
        model_path = f"./models/model_{instrument}/model_{instrument}2.pth"
        model.load_state_dict(
            torch.load(
                model_path,
                map_location=torch.device('mps')))
        print(f"-----{instrument} model loaded!-----")
        print(f"-----Generating {instrument} track-----")
        generate_midi_v2(model=model, tokenizer=tokenizer, src_midi=src_midi, max_len=max_len, PAD_ID=PAD_ID,
                         tgt_midi=f"./MIDIs/output_MIDI/{instrument}_track.mid")
        print(f"-----{instrument} track generated!-----")
    merge_midi_tracks(src_midi, "./MIDIs/output_MIDI/Drum_track.mid", "./MIDIs/output_MIDI/Bass_track.mid",
                      "./MIDIs/output_MIDI/Guitar_track.mid", "./MIDIs/output_MIDI/Piano_track.mid",
                      tgt_dir="./MIDIs/output_MIDI/generated_midi.mid")
