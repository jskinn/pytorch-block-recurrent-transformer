# Copyright 2023 Google, John Skinner
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import torch


def tiled_dropout(
    x: torch.Tensor,
    shape: tuple[int, ...],
    dropout_rate: float,
    deterministic: bool,
    generator: torch.random.Generator = None,
) -> torch.Tensor:
    """Tiles a dropout mask over a larger array.
    This will generate a smaller dropout mask of the given shape, and tile it
    over a larger array, which reduces the computational cost and memory
    associated with generating a large dropout mask.
    Args:
      x: The input array.
      shape: The shape of the dropout mask to tile.
      dropout_rate: The rate at which to drop.
      deterministic: If True, don't do dropout.
      generator: Pytorch Generator used for random number generation
    Returns:
      An array of the same shape as x, with some values dropped out.
    """
    if deterministic or dropout_rate <= 0.0:
        return x

    if x.ndim != len(shape):
        raise ValueError(
            "Shapes must have same number of dimensions %r, %r." % (x.shape, shape)
        )
    for xd, sd in zip(x.shape, shape):
        if (xd % sd) != 0:
            raise ValueError("Incompatible shapes %r, %r" % (x.shape, shape))

    # Work out how the tiles repeat
    repeats = [(1 if sd == 1 else xd // sd) for (xd, sd) in zip(x.shape, shape)]
    logging.getLogger(__name__).info("tiled dropout %r, tile: %r", x.shape, shape)

    keep_prob = 1.0 - dropout_rate
    keep = torch.bernoulli(torch.full(shape, keep_prob), generator=generator)
    keep = torch.tile(keep, repeats)
    keep = torch.broadcast_to(keep, x.shape)
    x_scaled = x / keep_prob
    return torch.where(keep, x_scaled, torch.zeros_like(x, dtype=x.dtype))
