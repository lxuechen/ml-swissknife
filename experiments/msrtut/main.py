import fire
import opacus

from torch import nn, optim
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_loss(model, x, y):
    # model(x): bsz x o
    # y: bsz x o
    # (model(x) - y) ** 2.): bsz x o
    return ((model(x) - y) ** 2.).sum(dim=1).mean(dim=0)


def main():
    # hvp
    d, p, o = 10, 100, 20
    bsz = 16
    m = nn.Sequential(nn.Linear(d, p, bias=False), nn.ReLU(), nn.Linear(p, o, bias=False)).to(device)  # (dp + po)
    x, y = torch.randn(bsz, d, device=device), torch.randn(bsz, o, device=device)

    loss = compute_loss(m, x, y)
    grad = torch.autograd.grad(loss, list(m.parameters()), create_graph=True)  # (dp + po)

    dp_plus_po = d * p + p * o
    v = torch.randn(dp_plus_po, device=device)
    v_unflat = []
    index = 0
    for gradi in grad:
        v_unflat.append(
            v[index:index+gradi.numel()].reshape(gradi.size())
        )
        index += gradi.numel()

    dot_product = sum([(gradi * vi).sum() for gradi, vi in zip(grad, v_unflat)])
    print(type(m.parameters()))
    hvp = torch.autograd.grad(
        dot_product, inputs=list(m.parameters()), create_graph=True
    )
    print(hvp[0].requires_grad)
    print(hvp[0].size(), hvp[1].size())


def main2():
    d, p, o = 10, 100, 20
    bsz = 16
    m = nn.Sequential(nn.Linear(d, p, bias=False), nn.ReLU(), nn.Linear(p, o, bias=False)).to(device)  # (dp + po)
    params = list(m.parameters())
    x, y = torch.randn(bsz, d, device=device), torch.randn(bsz, o, device=device)

    opacus.GradSampleModule(m=m)

    loss = compute_loss(m, x, y)
    loss.backward()

    print(params[0].grad_sample.size())
    breakpoint()


if __name__ == '__main__':
    fire.Fire(main2)
