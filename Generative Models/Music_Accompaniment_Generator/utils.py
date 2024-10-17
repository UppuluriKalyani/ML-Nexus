import pretty_midi
import mido
import torch
from mido import MidiFile, MidiTrack
from miditok import TokSequence


def midi_to_array(tokenizer, midifile, max_len):
    tokens = tokenizer(midifile)
    ids = [1]
    tokens_len = len(tokens.ids)
    if max_len <= tokens_len:
        ids.extend(tokens.ids[0:(max_len - 5)])
        ids.append(2)
        pad = [0] * (max_len - len(ids[0:(max_len - 3)]))
        ids.extend(pad)
    else:
        ids.extend(tokens.ids)
        ids.append(2)
        pad = [0] * (max_len - len(ids))
        ids.extend(pad)
    return ids[0:max_len]


def ids_to_tokens(array, idx2word):
    tokens = []
    for i in range(0, len(array)):
        tokens.append(idx2word[int(array[i])])
    return tokens


def tensor_to_midi(tensor, tokenizer, tgt_midi):
    ids = [int(element) for element in tensor]
    generated_tokens = TokSequence()
    generated_tokens.ids = ids
    tokenizer.complete_sequence(generated_tokens)
    generated_midi = tokenizer(generated_tokens)
    generated_midi.dump_midi(tgt_midi)


def find_closest_element(target, collection):
    closest_element = None
    min_difference = float('inf')  # 初始设为正无穷大

    for element in collection:
        difference = abs(target - element)

        if difference < min_difference:
            min_difference = difference
            closest_element = element

    return closest_element


def tempo_adjustment(src_midi, tgt_midi, new_bpm):
    midi_file = MidiFile(src_midi)
    new_midi_file = MidiFile(ticks_per_beat=midi_file.ticks_per_beat)
    for i, track in enumerate(midi_file.tracks):
        new_track = MidiTrack()
        new_midi_file.tracks.append(new_track)
        for msg in track:
            if msg.type == 'set_tempo':
                microseconds_per_beat = mido.bpm2tempo(new_bpm)
                new_msg = msg.copy(tempo=microseconds_per_beat)
                new_track.append(new_msg)
            else:
                new_track.append(msg)
    new_midi_file.save(tgt_midi)


def create_timestamp_array(max_time, standard_timestamp):
    timestamp_array = []
    current_time = 0

    while current_time <= max_time:
        timestamp_array.append(round(current_time, 8))
        current_time += standard_timestamp

    return timestamp_array


def note_alignment_32(midi_file):
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    tempo = round(midi_data.get_tempo_changes()[1][0])
    standard_timestamp = 60 / tempo / 8
    max_time = midi_data.get_end_time()
    timestamp_grid = create_timestamp_array(max_time, standard_timestamp)
    for midi_track in midi_data.instruments:
        notes = midi_track.notes
        for note in notes:
            note.start = find_closest_element(note.start, timestamp_grid)
            note.end = find_closest_element(note.end, timestamp_grid)
            if note.end == note.start:
                note.end = note.start + standard_timestamp
    return midi_data


def note_alignment_16(midi_file):
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    tempo = round(midi_data.get_tempo_changes()[1][0])
    standard_timestamp = 60 / tempo / 4
    max_time = midi_data.get_end_time()
    timestamp_grid = create_timestamp_array(max_time, standard_timestamp)
    for midi_track in midi_data.instruments:
        notes = midi_track.notes
        for note in notes:
            note.start = find_closest_element(note.start, timestamp_grid)
            note.end = find_closest_element(note.end, timestamp_grid)
            if note.end == note.start:
                note.end = note.start + standard_timestamp
    return midi_data


def merge_midi_tracks(*midi, tgt_dir):
    bpm = round(pretty_midi.PrettyMIDI(midi[0]).get_tempo_changes()[1][0])
    midi_data = pretty_midi.PrettyMIDI()
    for midi_track in midi:
        midi_track_data = pretty_midi.PrettyMIDI(midi_track)
        if len(midi_track_data.instruments) > 0:
            midi_data.instruments.append(midi_track_data.instruments[0])
    midi_data.write(tgt_dir)
    tempo_adjustment(tgt_dir, tgt_dir, bpm)
    note_alignment_16(tgt_dir).write(tgt_dir)


def midi_strip(midifile, tokenizer, tgt_folder, max_len):
    tokens = tokenizer(midifile)
    strip_tokens = TokSequence()
    strip_tokens.ids = tokens.ids[0:max_len]
    tokenizer.complete_sequence(strip_tokens)
    strip_midi = tokenizer(strip_tokens)
    strip_midi.dump_midi(tgt_folder)


def generate_midi(model, tokenizer, src_midi, max_len, PAD_ID, tgt_midi):
    model.eval()
    x = midi_to_array(tokenizer=tokenizer, midifile=src_midi, max_len=max_len)
    x = torch.LongTensor(x).reshape(1, max_len)
    y_input = torch.ones(1, max_len,
                         dtype=torch.long) * PAD_ID
    y_input[0] = 1
    with torch.no_grad():
        for i in range(1, y_input.shape[1]):
            y_hat = model(x, y_input)
            y_input[0, i] = torch.argmax(y_hat[0, i - 1])
    tensor_to_midi(y_input[0][1:], tokenizer, tgt_midi)


def generate_midi_v2(model, tokenizer, src_midi, max_len, PAD_ID, tgt_midi):
    model.eval()
    x = midi_to_array(tokenizer=tokenizer, midifile=src_midi, max_len=max_len)
    x = torch.LongTensor(x).reshape(1, max_len)
    y_input = torch.ones(1, max_len,
                         dtype=torch.long) * PAD_ID
    y_input[0] = 1
    with torch.no_grad():
        for i in range(1, y_input.shape[1]):
            y_hat = model(x, y_input)
            y_input[0, i] = torch.argmax(y_hat[0, i - 1])
    flag = 3
    for i in range(4, len(y_input[0]) - 1):
        if y_input[0][i] != 3:
            flag = i
            break
    y_ = [
        y_input[0][i] for i in range(
            0, len(
                y_input[0])) if i not in range(
            4, flag)]
    tensor_to_midi(y_[1:], tokenizer, tgt_midi)
