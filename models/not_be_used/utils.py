import torch
import matplotlib.pyplot as plt


def PositionalEncoding(length, dim):
    """    
    PE(pos,2i/2i+1) = sin/cos(pos/10000^{2i/d_{model}})
    """
    assert dim % 2 == 0, "Cannot use sin/cos positional encoding with odd dim (got dim={:d})".format(dim)

    pe = torch.zeros(length, dim)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = 10000 ** (torch.arange(0, dim, 2) / dim)
    div_term = div_term.unsqueeze(0)

    pe[:, 0::2] = torch.sin(position / div_term)
    pe[:, 1::2] = torch.cos(position / div_term)

    return pe


def PositionalEncoding_line(length, dim):
    position = torch.linspace(-1, 1, length).unsqueeze(1)
    position = position.repeat(1, dim)
    return position


def GenerateNoise(length, dim):
    t = torch.empty((length, dim)).uniform_(-1, 1)
    return t


def ShowTensor(tensor):
    p = tensor.numpy()
    plt.imshow(p, cmap='turbo')
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    pe = PositionalEncoding(50, 128)
    #print(pe.shape)
    t = pe + 0.5 * GenerateNoise(50, 128)
    ShowTensor(pe)
