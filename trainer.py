import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np
from datasets import classes


def train(model, loader, optimizer, loss_fn, scheduler, device='cpu'):
    model.train()

    train_loss = 0
    cnt = 0

    for x, y in tqdm(loader, desc='Train'):
        x, y = x.to(device).float(), y.to(device)

        optimizer.zero_grad()

        output = model(x)

        class_ = output.argmax(dim=1)
        cnt += (class_ == y).type(torch.float).mean()

        loss = loss_fn(output, y)

        train_loss += loss.item()

        loss.backward()

        optimizer.step()

    scheduler.step()

    train_loss /= len(loader)

    return train_loss, cnt.detach().item() / len(loader)


@torch.inference_mode()
def evaluate(model, loader, loss_fn, device='cpu'):
    model.eval()

    total_loss = 0
    cnt_all_class = np.zeros(10)
    cnt_ok_class = np.zeros(10)

    for x, y in tqdm(loader, desc='Evaluation'):
        x, y = x.to(device).float(), y.to(device)
        y_ = y.clone()
        output = model(x)

        loss = loss_fn(output, y)

        class_ = output.argmax(dim=1)

        uniq = (class_ == y_).type(torch.float)

        for i in y_.unique():
            cnt_all_class[i.item()] += len(y_[y_ == i])
            cnt_ok_class[i.item()] += uniq[y_ == i].sum()

        total_loss += loss.item()

    total_loss /= len(loader)

    score = cnt_ok_class.sum() / cnt_all_class.sum()

    return total_loss, score, cnt_ok_class / cnt_all_class


def train_model(model, train_loader, test_loader, epochs=10, device='cpu'):
    class_counts = torch.tensor([208, 313, 155, 298, 181, 87, 148, 123, 80, 80])
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * len(class_counts)

    loss_fn = torch.nn.CrossEntropyLoss(weight=weights.to(device))

    optimizer = optim.Adam([
        {'params': model.layer4.parameters(), 'lr': 1e-4},
        {'params': model.layer3.parameters(), 'lr': 1e-5},
        {'params': model.fc.parameters(), 'lr': 1e-3},
    ])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90)

    for epoch in range(epochs):
        train_loss, acc_train = train(model, train_loader, optimizer, loss_fn, scheduler, device)
        valid_loss, acc_test, recall_all = evaluate(model, test_loader, loss_fn, device)

        print(f'lr: {scheduler.get_lr()}')
        print(f'train_loss: {train_loss}')
        print(f'valid_loss: {valid_loss}')
        print(f'acc_train: {acc_train}')
        print(f'acc_test: {acc_test}')

        for i in range(10):
            print(classes[i], recall_all[i])
