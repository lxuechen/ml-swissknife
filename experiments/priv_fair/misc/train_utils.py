import torch
import torch.nn.functional as F


def get_device():
    use_cuda = torch.cuda.is_available()
    assert use_cuda
    device = torch.device("cuda" if use_cuda else "cpu")
    return device


def train(model, train_loader, optimizer, n_acc_steps=1):
    device = next(model.parameters()).device
    model.train()
    num_examples = 0
    correct = 0
    train_loss = 0

    rem = len(train_loader) % n_acc_steps
    num_batches = len(train_loader)
    num_batches -= rem

    bs = train_loader.batch_size if train_loader.batch_size is not None else train_loader.batch_sampler.batch_size
    print(f"training on {num_batches} batches of size {bs}")

    for batch_idx, (data, target) in enumerate(train_loader):

        if batch_idx > num_batches - 1:
            break

        data, target = data.to(device), target.to(device)

        output = model(data)

        loss = F.cross_entropy(output, target)
        loss.backward()

        if ((batch_idx + 1) % n_acc_steps == 0) or ((batch_idx + 1) == len(train_loader)):
            optimizer.step()
            optimizer.zero_grad()
        else:
            with torch.no_grad():
                # accumulate per-example gradients but don't take a step yet
                if hasattr(optimizer, 'virtual_step'):
                    optimizer.virtual_step()

        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        train_loss += F.cross_entropy(output, target, reduction='sum').item()
        num_examples += len(data)

    train_loss /= num_examples
    train_acc = 100. * correct / num_examples

    print(f'Train set: Average loss: {train_loss:.4f}, '
          f'Accuracy: {correct}/{num_examples} ({train_acc:.2f}%)')

    return train_loss, train_acc


@torch.no_grad()
def test(model, test_loader, msg=''):
    device = next(model.parameters()).device
    model.eval()
    num_examples = 0
    test_loss = 0
    correct = 0

    for i, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += F.cross_entropy(output, target, reduction='sum').item()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        num_examples += len(data)

    test_loss /= num_examples
    test_acc = 100. * correct / num_examples

    msg += f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{num_examples} ({test_acc:.2f}%)"
    print(msg)

    return test_loss, test_acc


@torch.no_grad()
def test_by_groups(model, test_loader):
    import collections
    import numpy as np
    device = next(model.parameters()).device
    model.eval()
    zeons, xents = collections.OrderedDict(), collections.OrderedDict()

    for x, y in test_loader:
        x, y = tuple(t.to(device) for t in (x, y))
        y_hat = model(x)
        zeon = torch.eq(y_hat.argmax(dim=1), y).cpu().float().tolist()
        xent = F.cross_entropy(y_hat, y, reduction="none").cpu().tolist()
        y = y.cpu().tolist()
        for y_i, zeon_i, xent_i in zip(y, zeon, xent):
            if y_i not in zeons:
                zeons[y_i] = [zeon_i]
                xents[y_i] = [xent_i]
            else:
                zeons[y_i].append(zeon_i)
                xents[y_i].append(xent_i)

    for y_i in zeons.keys():
        zeons[y_i] = np.mean(zeons[y_i])
        xents[y_i] = np.mean(xents[y_i])

    # Re-sort.
    zeons = {key: zeons[key] for key in sorted(zeons.keys())}
    xents = {key: xents[key] for key in sorted(xents.keys())}

    return xents, zeons
