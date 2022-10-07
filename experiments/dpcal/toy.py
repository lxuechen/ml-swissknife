import fire
import matplotlib.pyplot as plt
import opacus
import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset

from ml_swissknife import utils


def make_data(ntrain=100000, ntest=10000):
    n = ntrain + ntest

    y = coin_flips = (torch.rand(size=(n,)) > 0.5).long()

    x_pos = torch.randn(size=(n, 2)) + torch.tensor([1.5, 0.])
    x_neg = torch.randn(size=(n, 2)) + torch.tensor([0., 1.5])
    x = x_pos * coin_flips[:, None] + x_neg * (1. - coin_flips)[:, None]

    xtrain, xtest = x[:ntrain], x[ntrain:]
    ytrain, ytest = y[:ntrain], y[ntrain:]
    return (xtrain, ytrain), (xtest, ytest)


@torch.inference_mode()
def evaluate(loader, model):
    avg_loss = []
    for x, y in loader:
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction='none')
        avg_loss.append(loss)
    return torch.cat(avg_loss).mean(dim=0)


@torch.inference_mode()
def get_confidences(loader, model):
    confidences = []
    for x, y in loader:
        probs = F.softmax(model(x))[:, 0]
        confidences.append(probs)
    return torch.cat(confidences)


@torch.inference_mode()
def get_logits_and_labels(loader, model):
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x))
        labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def main(ntrain=100000, ntest=20000, batch_size=5000, epochs=30, lr=1e-1, target_epsilon=8, max_grad_norm=0.1, seed=42):
    utils.manual_seed(seed)

    (xtrain, ytrain), (xtest, ytest) = make_data(ntrain=ntrain, ntest=ntest)
    train_dataset = TensorDataset(xtrain, ytrain)
    test_dataset = TensorDataset(xtest, ytest)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    model = nn.Linear(2, 2)
    optimizer = optim.SGD(params=model.parameters(), lr=lr, momentum=0.)

    if target_epsilon > 0.:
        model, optimizer, train_dataloader = opacus.PrivacyEngine().make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_dataloader,
            target_epsilon=target_epsilon,
            target_delta=ntrain ** -1.1,
            epochs=epochs,
            max_grad_norm=max_grad_norm,
        )

    for epoch in range(epochs):
        for (x, y) in train_dataloader:
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y, reduction="mean")
            loss.backward()
            optimizer.step()

        tr_loss = evaluate(train_dataloader, model)
        te_loss = evaluate(test_dataloader, model)
        print(f'epoch: {epoch}, tr loss: {tr_loss.item():.4f}, te loss: {te_loss.item():.4f}')

    # Plot train and test model confidence.
    tr_confidences = get_confidences(train_dataloader, model)
    te_confidences = get_confidences(test_dataloader, model)

    plt.hist(x=tr_confidences.numpy(), bins=100, label="train", alpha=0.4, density=True)
    plt.hist(x=te_confidences.numpy(), bins=100, label="test", alpha=0.4, density=True)
    plt.show()

    # evaluate ece
    tr_logits, tr_labels = get_logits_and_labels(train_dataloader, model)
    te_logits, te_labels = get_logits_and_labels(test_dataloader, model)

    ECE = utils.ECELoss()
    tr_ece = ECE(tr_logits, tr_labels)
    te_ece = ECE(te_logits, te_labels)
    print(tr_ece, te_ece)


if __name__ == "__main__":
    fire.Fire(main)
