import torch.nn as nn
from nn_utils import MaskedConv1d, Permute


class Processor(nn.Module):
    def __init__(self, vocab_size, abstraction_dim) -> None:
        super(Processor, self).__init__()
        embed_dim = 32
        self._net = nn.Sequential(
            nn.Embedding(vocab_size, embed_dim),
            Permute(0, 2, 1),
            MaskedConv1d(embed_dim, abstraction_dim, mask=[1, 1, 1, 0, 1, 1, 1]),
        )

    def forward(self, batch):
        return self._net(batch)


class AbstractToGoal(nn.Module):
    def __init__(self, vocab_size, abstraction_dim) -> None:
        super(AbstractToGoal, self).__init__()
        self._net = nn.Sequential(
            nn.BatchNorm1d(abstraction_dim),
            nn.Conv1d(abstraction_dim, 256, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, vocab_size, kernel_size=1)
        )

    def forward(self, abstraction):
        return self._net(abstraction)


class AbstractToAbstract(nn.Module):
    """ predicts sign of the target """
    def __init__(self, abstraction_dim) -> None:
        super(AbstractToAbstract, self).__init__()
        self._net = nn.Sequential(
            nn.BatchNorm1d(abstraction_dim),
            nn.Conv1d(abstraction_dim, 256, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, abstraction_dim, kernel_size=1)
        )

    def forward(self, batch):
        return self._net(batch)
