"""A DiT that denoises an image and a label together (UniDiffuser-style; Bao et al., 2023).

The label is a vector with one entry per class. It goes in as one extra token next to the 256
image patches, so attention connects the label and every patch in both directions. The image
and the label each have their own time, and the sum of the two time embeddings conditions every
block through adaLN-Zero, as in the class-conditional DiT. The network predicts the flow-matching
velocity of both parts.
"""

import torch
from torch import nn

from diffcls.models.dit import (
    ADALN_FINAL_CHUNKS, LAYER_NORM_EPS, DiTBlock, DiTConfig, TimestepEmbedding, modulate,
)


class JointDiT(nn.Module):
    """forward(x, y, t_x, t_y) -> (image velocity, label velocity). DiTConfig.num_classes sets the label size."""

    def __init__(self, config: DiTConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = nn.Conv2d(config.channels, config.dim, config.patch_size, stride=config.patch_size)
        self.label_in = nn.Linear(config.num_classes, config.dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.grid_size ** 2 + 1, config.dim))  # + the label token
        self.image_time_embed = TimestepEmbedding(config.time_freq_dim, config.dim)
        self.label_time_embed = TimestepEmbedding(config.time_freq_dim, config.dim)
        self.blocks = nn.ModuleList(DiTBlock(config) for _ in range(config.depth))
        self.final_norm = nn.LayerNorm(config.dim, elementwise_affine=False, eps=LAYER_NORM_EPS)
        self.final_adaln = nn.Sequential(nn.SiLU(), nn.Linear(config.dim, ADALN_FINAL_CHUNKS * config.dim))
        self.image_out = nn.Linear(config.dim, config.patch_size ** 2 * config.channels)
        self.label_out = nn.Linear(config.dim, config.num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for block in self.blocks:  # adaLN-Zero: every block starts as the identity
            nn.init.zeros_(block.adaln[-1].weight)
            nn.init.zeros_(block.adaln[-1].bias)
        for layer in (self.final_adaln[-1], self.image_out, self.label_out):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor, t_x: torch.Tensor,
                t_y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, 3, H, W) noisy image, y: (B, classes) noisy label; t_x, t_y: (B,) DiT time inputs."""
        patches = self.patch_embed(x).flatten(2).transpose(1, 2)
        tokens = torch.cat([patches, self.label_in(y.to(patches.dtype)).unsqueeze(1)], dim=1) + self.pos_embed
        cond = self.image_time_embed(t_x) + self.label_time_embed(t_y)
        for block in self.blocks:
            tokens = block(tokens, cond)
        shift, scale = self.final_adaln(cond).chunk(ADALN_FINAL_CHUNKS, dim=1)
        tokens = modulate(self.final_norm(tokens), shift, scale)
        return self._unpatchify(self.image_out(tokens[:, :-1])), self.label_out(tokens[:, -1])

    def _unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        c, p, g = self.config.channels, self.config.patch_size, self.config.grid_size
        patches = tokens.view(tokens.shape[0], g, g, p, p, c)
        return patches.permute(0, 5, 1, 3, 2, 4).reshape(tokens.shape[0], c, g * p, g * p)
