"""Tests for :mod:`dissociation.imagenet` on synthetic WebDataset shards."""

import io
import pickle
import tarfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from dissociation import imagenet

# (key, label, fill colour) per shard, in member order. Keys inside the
# first shard are deliberately not sorted.
SHARDS = {
    "imagenet1k-validation-00.tar": [
        ("n01440764_10026", 7, (255, 0, 0)),
        ("ILSVRC2012_val_00000002", 0, (0, 255, 0)),
    ],
    "imagenet1k-validation-01.tar": [("n04579432_2", 999, (0, 0, 255))],
}


def _jpeg(colour: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), colour).save(buffer, format="JPEG")
    return buffer.getvalue()


def _add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


@pytest.fixture
def shard_dir(tmp_path: Path) -> Path:
    """Write the synthetic shards and an unrelated tar into ``tmp_path``."""
    for name, samples in SHARDS.items():
        with tarfile.open(tmp_path / name, mode="w") as tar:
            for key, label, colour in samples:
                _add(tar, f"{key}.jpg", _jpeg(colour))
                _add(tar, f"{key}.cls", f"{label}\n".encode())
    with tarfile.open(tmp_path / "imagenet1k-train-0000.tar", mode="w") as tar:
        _add(tar, "x.jpg", _jpeg((1, 2, 3)))
        _add(tar, "x.cls", b"1")
    return tmp_path


class TestIndex:
    """Shard discovery and indexing."""

    def test_find_shards_sorted(self, shard_dir: Path) -> None:
        """Only validation shards are found, in name order."""
        shards = imagenet.find_shards(shard_dir)
        assert [s.name for s in shards] == sorted(SHARDS)

    def test_find_shards_empty(self, tmp_path: Path) -> None:
        """An empty directory is an error that names the dataset."""
        with pytest.raises(FileNotFoundError, match="imagenet-1k-wds"):
            imagenet.find_shards(tmp_path)

    def test_index_order_and_labels(self, shard_dir: Path) -> None:
        """Samples keep shard order and carry the parsed labels."""
        samples = imagenet.index_shards(imagenet.find_shards(shard_dir))
        expected = [
            (key, label) for group in SHARDS.values() for key, label, _ in group
        ]
        assert [(s.key, s.label) for s in samples] == expected
        assert [s.shard for s in samples] == [0, 0, 1]

    def test_index_offsets_locate_jpeg(self, shard_dir: Path) -> None:
        """Offsets and sizes address the raw JPEG bytes."""
        shards = imagenet.find_shards(shard_dir)
        sample = imagenet.index_shards(shards)[1]
        with shards[sample.shard].open("rb") as handle:
            handle.seek(sample.offset)
            data = handle.read(sample.size)
        assert data == _jpeg((0, 255, 0))

    def test_missing_label_raises(self, tmp_path: Path) -> None:
        """An image without a label member is an error."""
        with tarfile.open(
            tmp_path / "imagenet1k-validation-00.tar", mode="w"
        ) as tar:
            _add(tar, "a.jpg", _jpeg((0, 0, 0)))
        with pytest.raises(ValueError, match="lacks"):
            imagenet.index_shards(imagenet.find_shards(tmp_path))


class TestDataset:
    """``ImageNetShards`` behaviour."""

    def test_getitem_decodes_rgb(self, shard_dir: Path) -> None:
        """Without a transform, items are RGB PIL images and int labels."""
        dataset = imagenet.ImageNetShards.from_directory(shard_dir)
        image, label = dataset[2]
        assert isinstance(image, Image.Image)
        assert image.mode == "RGB"
        assert image.size == (8, 8)
        assert label == 999
        assert np.array(image)[0, 0, 2] > 200
        dataset.close()

    def test_transform_and_loader(self, shard_dir: Path) -> None:
        """A transform is applied and a ``DataLoader`` sees every sample."""
        dataset = imagenet.ImageNetShards.from_directory(
            shard_dir, transform=transforms.ToTensor()
        )
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        batches = list(loader)
        assert [tuple(images.shape) for images, _ in batches] == [
            (2, 3, 8, 8),
            (1, 3, 8, 8),
        ]
        labels = np.concatenate([lab.numpy() for _, lab in batches])
        assert labels.tolist() == dataset.labels.tolist() == [7, 0, 999]

    def test_pickles_without_handles(self, shard_dir: Path) -> None:
        """Pickling the dataset excludes open file handles."""
        dataset = imagenet.ImageNetShards.from_directory(shard_dir)
        dataset[0]
        copy = pickle.loads(pickle.dumps(dataset))
        assert copy._handles == {}  # noqa: SLF001
        assert copy[0][1] == 7
        dataset.close()
        copy.close()
