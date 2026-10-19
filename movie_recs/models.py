"""Two independent towers; both models use an inner product for retrieval."""

import torch
from torch import nn


class MatrixFactorization(nn.Module):
    def __init__(self, users: int, items: int, dim: int, genres: torch.Tensor):
        super().__init__()
        self.user_embedding = nn.Embedding(users, dim)
        self.item_embedding = nn.Embedding(items, dim)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def encode_users(self, users: torch.Tensor) -> torch.Tensor:
        return self.user_embedding(users)

    def encode_items(self, items: torch.Tensor) -> torch.Tensor:
        return self.item_embedding(items)


class NeuralTwoTower(MatrixFactorization):
    def __init__(self, users: int, items: int, dim: int, genres: torch.Tensor):
        super().__init__(users, items, dim, genres)
        self.register_buffer("genres", genres)
        self.user_tower = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.item_tower = nn.Sequential(
            nn.Linear(dim + genres.shape[1], dim), nn.ReLU(), nn.Linear(dim, dim)
        )

    def encode_users(self, users: torch.Tensor) -> torch.Tensor:
        return self.user_tower(self.user_embedding(users))

    def encode_items(self, items: torch.Tensor) -> torch.Tensor:
        return self.item_tower(torch.cat([self.item_embedding(items), self.genres[items]], dim=-1))


MODELS = {"mf": MatrixFactorization, "neural": NeuralTwoTower}
