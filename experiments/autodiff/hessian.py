import fire
import torch
from torch import nn
from torch.nn import functional as F


def main():
    bsz = 16
    d, c = 100, 10
    x = torch.randn(bsz, d)
    t = torch.randint(low=0, high=c, size=(bsz,))
    net = nn.Linear(d, c)
    params = tuple(net.parameters())
    y = net(x)
    loss = F.cross_entropy(y, t)

    grads = torch.autograd.grad(loss, params, retain_graph=True, create_graph=True)
    flat_grads = torch.cat([grad.flatten() for grad in grads])
    hessian = []

    for i in range(len(flat_grads)):
        one_hot = torch.zeros_like(flat_grads)
        one_hot[i] = 1.

        gvp = torch.sum(flat_grads * one_hot)
        slice = torch.autograd.grad(gvp, params, retain_graph=True, )
        flat_slice = torch.cat([t.flatten() for t in slice])
        hessian.append(flat_slice)
    hessian = torch.stack(hessian)
    print(hessian.size())


if __name__ == "__main__":
    fire.Fire(main)
