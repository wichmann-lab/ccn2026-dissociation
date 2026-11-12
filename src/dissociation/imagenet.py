"""The ImageNet validation set read from WebDataset shards.

The ImageNet runs are evaluated on the ILSVRC-2012 validation images as
distributed by the ``timm/imagenet-1k-wds`` dataset on the Hugging Face
Hub: 64 uncompressed tar shards, ``imagenet1k-validation-00.tar`` to
``imagenet1k-validation-63.tar``, holding for every image a ``<key>.jpg``
member with the JPEG bytes and a ``<key>.cls`` member with the class index
(0-999, in the sorted WordNet-id order torchvision uses) as text. The
shards are read directly with :mod:`tarfile`, so the evaluation needs no
``webdataset`` package, and samples are served in shard order, which the
published files fix.

Classes:
-------

* :class:`ShardSample`: An image's location within a shard and its class
  label.
* :class:`ImageNetShards`: Map-style dataset over the indexed shards.

Functions:
---------

* :func:`find_shards`: The validation shards below a directory.
* :func:`index_shards`: Locate every image / label pair in the shards.
"""

__all__ = [
    "IMAGE_KEY",
    "LABEL_KEY",
    "VALIDATION_SHARDS",
    "ImageNetShards",
    "ShardSample",
    "find_shards",
    "index_shards",
]

import io
import tarfile
from collections.abc import Callable
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import IO
from typing import Any

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

VALIDATION_SHARDS = "imagenet1k-validation-*.tar"
IMAGE_KEY = "jpg"
LABEL_KEY = "cls"


@dataclass(frozen=True)
class ShardSample:
    """An image's location within a shard and its class label.

    Attributes:
        key: The WebDataset key (member name up to the first dot).
        shard: Index of the shard in the shard list.
        offset: Byte offset of the JPEG data inside the shard.
        size: Size of the JPEG data in bytes.
        label: The class index.
    """

    key: str
    shard: int
    offset: int
    size: int
    label: int


def find_shards(root: Path, pattern: str = VALIDATION_SHARDS) -> list[Path]:
    """Return the shards below ``root`` matching ``pattern``, sorted by name.

    Args:
        root: Directory holding the shards.
        pattern: Glob pattern of the shard files.

    Returns:
        The matching shard paths in name order.

    Raises:
        FileNotFoundError: If no shard matches.
    """
    shards = sorted(Path(root).glob(pattern))
    if not shards:
        msg = (
            f"no {pattern!r} shards found under {root}; download the "
            "validation shards of the 'timm/imagenet-1k-wds' dataset from the "
            "Hugging Face Hub first (see results/README.md)."
        )
        raise FileNotFoundError(msg)
    return shards


def _split_member_name(name: str) -> tuple[str, str]:
    """Split a tar member name into its WebDataset key and extension."""
    base = name.rsplit("/", 1)[-1]
    key, _, extension = base.partition(".")
    return key, extension


def index_shards(
    shards: Iterable[Path],
    *,
    image_key: str = IMAGE_KEY,
    label_key: str = LABEL_KEY,
) -> list[ShardSample]:
    """Locate every image / label pair in the shards.

    Scans the tar headers of each shard (no image is decoded) and reads
    the small label members. Samples keep the order in which they appear
    in the shards, shard by shard.

    Args:
        shards: The shard files, in the order their samples are numbered.
        image_key: Member extension holding the JPEG bytes.
        label_key: Member extension holding the class index as text.

    Returns:
        One :class:`ShardSample` per image.

    Raises:
        ValueError: If a sample lacks its image or label member.
    """
    samples: list[ShardSample] = []
    for shard_index, shard in enumerate(shards):
        images: dict[str, tarfile.TarInfo] = {}
        labels: dict[str, int] = {}
        order: list[str] = []
        with tarfile.open(shard, mode="r:") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                key, extension = _split_member_name(member.name)
                if key not in images and key not in labels:
                    order.append(key)
                if extension == image_key:
                    images[key] = member
                elif extension == label_key:
                    handle = tar.extractfile(member)
                    if handle is None:  # pragma: no cover - isfile() holds
                        continue
                    with handle:
                        labels[key] = int(handle.read().decode("utf-8"))
        for key in order:
            if key not in images or key not in labels:
                msg = (
                    f"{shard}: sample {key!r} lacks its .{image_key} or "
                    f".{label_key} member."
                )
                raise ValueError(msg)
            member = images[key]
            samples.append(
                ShardSample(
                    key,
                    shard_index,
                    member.offset_data,
                    member.size,
                    labels[key],
                )
            )
    return samples


class ImageNetShards(Dataset[tuple[Any, int]]):
    """Map-style dataset over indexed WebDataset shards.

    Serves ``(image, label)`` pairs like torchvision's ``ImageNet``: the
    image decoded with PIL as RGB and passed through ``transform`` when one
    is given, the label as the integer class index. Shard files are opened
    lazily and kept open per process, so the dataset can be pickled for
    ``DataLoader`` workers.
    """

    def __init__(
        self,
        shards: Iterable[Path],
        samples: Iterable[ShardSample],
        transform: Callable[[Image.Image], Any] | None = None,
    ) -> None:
        """Wrap indexed shards.

        Args:
            shards: The shard files that ``samples`` refer to, in order.
            samples: The index from :func:`index_shards`.
            transform: Applied to each decoded image.
        """
        self.shards = [Path(shard) for shard in shards]
        self.samples = list(samples)
        self.transform = transform
        self._handles: dict[int, IO[bytes]] = {}

    @classmethod
    def from_directory(
        cls,
        root: Path,
        transform: Callable[[Image.Image], Any] | None = None,
        pattern: str = VALIDATION_SHARDS,
    ) -> "ImageNetShards":
        """Find and index the shards below ``root``.

        Args:
            root: Directory holding the shards.
            transform: Applied to each decoded image.
            pattern: Glob pattern of the shard files.

        Returns:
            The dataset over every sample found.
        """
        shards = find_shards(root, pattern)
        return cls(shards, index_shards(shards), transform=transform)

    @property
    def labels(self) -> np.ndarray:
        """The class index of every sample, in dataset order."""
        return np.array([sample.label for sample in self.samples], dtype=int)

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        """Return the (transformed) image and label at ``index``."""
        sample = self.samples[index]
        handle = self._handles.get(sample.shard)
        if handle is None:
            # Kept open for the lifetime of the process (see ``close``).
            handle = self.shards[sample.shard].open("rb")
            self._handles[sample.shard] = handle
        handle.seek(sample.offset)
        with Image.open(io.BytesIO(handle.read(sample.size))) as image:
            rgb = image.convert("RGB")
        if self.transform is not None:
            return self.transform(rgb), sample.label
        return rgb, sample.label

    def close(self) -> None:
        """Close the shard files opened so far."""
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __getstate__(self) -> dict[str, Any]:
        """Pickle without open file handles (``DataLoader`` workers)."""
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state
