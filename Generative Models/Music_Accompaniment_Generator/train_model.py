from data_load import get_batch_indices, data_load
from train_parameters import max_len
from tqdm import tqdm
import torch
from torch import nn
import time
import os


def train_model(model, data_split_rate, data_len, batch_size, lr,
                n_epochs, PAD_ID, device, print_interval, instrument):
    print(f"--------Train Model For {instrument} Start!--------")
    x_folder = f"./HMuseData/Melody2{instrument}/Melody/"
    y_folder = f"./HMuseData/Melody2{instrument}/{instrument}/"
    split = round(data_len * data_split_rate)
    x, y = data_load(data_type="train", split=split, data_len=data_len,
                     x_folder=x_folder, y_folder=y_folder)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    tic = time.time()
    counter = 0
    for epoch in range(n_epochs):
        for index, _ in tqdm(get_batch_indices(
                len(x), batch_size), desc="Processing", unit="batches"):
            x_batch = torch.LongTensor(x[index]).to(device)
            y_batch = torch.LongTensor(y[index]).to(device)
            y_input = y_batch[:, :-1]
            y_label = y_batch[:, 1:]
            y_hat = model(x_batch, y_input)

            y_label_mask = y_label != PAD_ID
            preds = torch.argmax(y_hat, -1)
            correct = preds == y_label
            acc = torch.sum(y_label_mask * correct) / torch.sum(y_label_mask)

            n, seq_len = y_label.shape
            y_hat = torch.reshape(y_hat, (n * seq_len, -1))
            y_label = torch.reshape(y_label, (n * seq_len, ))
            loss = criterion(y_hat, y_label)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            if counter % print_interval == 0:
                toc = time.time()
                interval = toc - tic
                minutes = int(interval // 60)
                seconds = int(interval % 60)
                print(f'{counter:08d} {minutes:02d}:{seconds:02d}'
                      f' loss: {loss.item()} acc: {acc.item()}')
            counter += 1

    model_path = f"models/model_{instrument}/"
    os.makedirs(model_path, exist_ok=True)
    model_name = f"{model_path}model_{instrument}_{max_len}.pth"
    torch.save(model.state_dict(), model_name)

    print(f'Model saved to {model_name}')
    print(f"--------Train Model For {instrument} Completed!--------")
