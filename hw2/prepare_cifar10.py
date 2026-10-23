"""Prepare the original CIFAR-10 files using verified, resumable mirrors.

Run independently with ``python prepare_cifar10.py --data data``.
No PyTorch import or GPU allocation is needed for downloading.
"""
import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import time
from urllib.request import Request, urlopen


# SHA256: keras-team/keras, keras/src/datasets/cifar10.py.
# Per-file MD5s: pytorch/vision, torchvision/datasets/cifar.py.
ARCHIVE_SHA256 = "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce"
ARCHIVE_SIZE = 170498071
BASE_FOLDER = "cifar-10-batches-py"
FILE_MD5 = {
    "data_batch_1": "c99cafc152244af753f735de768cd75f",
    "data_batch_2": "d4bba439e000b95fd0a9bffe97cbabec",
    "data_batch_3": "54ebc095f3ab1f0389bbae665268c751",
    "data_batch_4": "634d18415352ddfa80567beed471001a",
    "data_batch_5": "482c414d41f54cd18b22e5b47cb7c3cb",
    "test_batch": "40351d587109b95175f43aff81a1287e",
    "batches.meta": "5ff9c542aee3614f3951f8cda6e48888",
}
MIRRORS = (
    "https://huggingface.co/datasets/MIT-OL-AI-D/cifar-10-python/resolve/"
    "a48007227f9e2cd0af96175f4afb5ac4965e261b/cifar-10-python.tar.gz?download=true",
    "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz",
    "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
)


def _digest(path, algorithm):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ready(root):
    return all(
        (root / BASE_FOLDER / name).is_file()
        and _digest(root / BASE_FOLDER / name, "md5") == checksum
        for name, checksum in FILE_MD5.items()
    )


def _valid_archive(path):
    return (path.is_file() and path.stat().st_size == ARCHIVE_SIZE
            and _digest(path, "sha256") == ARCHIVE_SHA256)


def _download(url, partial):
    offset = partial.stat().st_size if partial.exists() else 0
    if offset == ARCHIVE_SIZE:
        return
    if offset > ARCHIVE_SIZE:
        partial.unlink()
        offset = 0
    headers = {"User-Agent": "hw2-cifar10/1.0", "Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    started, last_update = time.monotonic(), 0.0
    with urlopen(Request(url, headers=headers), timeout=20) as response:
        if response.status == 206:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {offset}-") or not content_range.endswith(f"/{ARCHIVE_SIZE}"):
                raise OSError(f"Unexpected HTTP range: {content_range}")
        elif response.status == 200:
            # Servers that ignore Range must replace, never append to, the partial file.
            offset = 0
        else:
            raise OSError(f"Unexpected HTTP status: {response.status}")
        received = offset
        with partial.open("ab" if offset else "wb") as output:
            while True:
                block = response.read(256 * 1024)
                if not block:
                    break
                output.write(block)
                received += len(block)
                if received > ARCHIVE_SIZE:
                    raise OSError("Download exceeds expected archive size")
                now = time.monotonic()
                if now - last_update >= 1:
                    speed = (received - offset) / max(now - started, 0.001) / 1e6
                    print(f"\r[CIFAR-10] {received / ARCHIVE_SIZE:5.1%}  "
                          f"{received / 1e6:.1f}/{ARCHIVE_SIZE / 1e6:.1f} MB  "
                          f"{speed:.2f} MB/s", end="", flush=True)
                    last_update = now
        print(flush=True)
    if received != ARCHIVE_SIZE:
        raise OSError(f"Incomplete download: {received}/{ARCHIVE_SIZE} bytes; partial file retained")


def ensure_cifar10(root):
    """Return after all seven original data files pass their upstream checksums."""
    root = Path(root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    if _ready(root):
        return
    # Shared by training processes on Linux/macOS/Colab; released on interruption.
    with (root / ".cifar10-download.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if _ready(root):
            return
        archive = root / "cifar-10-python.fast.tar.gz"
        original = root / "cifar-10-python.tar.gz"
        if _valid_archive(original):
            archive = original
        elif not _valid_archive(archive):
            partial = root / ".cifar-10-python.fast.tar.gz.part"
            errors = []
            for url in MIRRORS:
                for attempt in range(2):
                    print(f"[CIFAR-10] Download source: {url}", flush=True)
                    try:
                        _download(url, partial)
                        if not _valid_archive(partial):
                            partial.unlink(missing_ok=True)
                            raise OSError("SHA256 mismatch; refusing to extract downloaded content")
                        os.replace(partial, archive)
                        break
                    except (OSError, ValueError) as exc:
                        errors.append(str(exc))
                        print(f"\n[CIFAR-10] Attempt {attempt + 1}/2: {exc}", flush=True)
                else:
                    continue
                break
            else:
                raise RuntimeError("CIFAR-10 download failed: " + "; ".join(errors))
        print("[CIFAR-10] Archive verified; extracting original dataset...", flush=True)
        with tempfile.TemporaryDirectory(prefix=".cifar10-extract-", dir=root) as staging:
            staging = Path(staging)
            target = staging / BASE_FOLDER
            target.mkdir()
            # Extract only known regular files, without accepting paths or symlinks from a tar.
            with tarfile.open(archive, "r:gz") as bundle:
                for name in FILE_MD5:
                    member = bundle.getmember(f"{BASE_FOLDER}/{name}")
                    if not member.isfile():
                        raise OSError(f"Unexpected dataset member: {name}")
                    with bundle.extractfile(member) as source, (target / name).open("wb") as output:
                        shutil.copyfileobj(source, output)
            if not _ready(staging):
                raise OSError("Extracted dataset failed upstream checksums")
            destination = root / BASE_FOLDER
            destination.mkdir(exist_ok=True)
            for name in FILE_MD5:
                os.replace(target / name, destination / name)
        print("[CIFAR-10] Ready: 50,000 training / 10,000 test images.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data", help="CIFAR-10 cache directory")
    ensure_cifar10(parser.parse_args().data)
