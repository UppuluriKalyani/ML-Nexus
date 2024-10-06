batch_size = 16
lr = 0.0001
d_model = 512
d_ff = 2048
n_layers = 6
heads = 8
dropout_rate = 0.2
n_epochs = 60
PAD_ID = 0
device = "mps"
# device = "cuda:0"
print_interval = 100
max_len = 750
data_split_rate = 0.99
len_Dataset = {'Drum': 18621,
               'Bass': 14316,
               'Guitar': 20037,
               'Piano': 11684}
