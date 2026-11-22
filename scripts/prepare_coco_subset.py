"""Build a fixed COCO 2017 validation subset for accuracy evaluation.

The subset must be **fixed and reproducible**: every configuration is scored on
exactly the same images, otherwise accuracy differences between configurations
would partly reflect which images they happened to see.

Selection rule: take the COCO ``val2017`` images sorted by image id and keep the
first ``--num-images``. Sorting makes the choice deterministic and independent of
file-system order or download order.

Only the images of the subset are downloaded (COCO serves individual images), and
the ground-truth annotations are filtered down to those images, so the resulting
``annotations.json`` is small and self-contained.

Usage
-----
    python scripts/prepare_coco_subset.py --num-images 100
    python scripts/prepare_coco_subset.py --num-images 100 --out data/coco_subset
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]

ANNOTATIONS_ZIP = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMAGE_URL = "http://images.cocodataset.org/val2017/{file_name}"


def download(url: str, target: Path, label: str, retries: int = 8) -> None:
    """Download with resume support and retries on truncated transfers.

    The COCO server terminates long transfers fairly often; restarting a 250 MB
    download from zero each time wastes minutes. A ``Range`` request resumes from
    whatever the ``.part`` file already holds.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print(f"  {label}: already present ({target.stat().st_size / 1e6:.1f} MB)")
        return

    tmp = target.with_suffix(target.suffix + ".part")

    for attempt in range(1, retries + 1):
        have = tmp.stat().st_size if tmp.exists() else 0
        request = urllib.request.Request(url)
        if have:
            request.add_header("Range", f"bytes={have}-")

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                total = response.headers.get("Content-Length")
                total = int(total) + have if total else 0
                mode = "ab" if have and response.status == 206 else "wb"
                if mode == "wb":
                    have = 0
                with tmp.open(mode) as handle:
                    while True:
                        chunk = response.read(1 << 20)
                        if not chunk:
                            break
                        handle.write(chunk)
                        have += len(chunk)
                        if total:
                            print(f"\r  {label}: {have / 1e6:.1f}/{total / 1e6:.1f} MB",
                                  end="", flush=True)

            size = tmp.stat().st_size
            if total and size < total:
                raise IOError(f"truncated at {size} of {total} bytes")
            tmp.replace(target)
            print(f"\r  {label}: done ({size / 1e6:.1f} MB)" + " " * 20)
            return
        except Exception as error:  # noqa: BLE001
            print(f"\r  {label}: attempt {attempt}/{retries} interrupted "
                  f"({type(error).__name__}); resuming from "
                  f"{tmp.stat().st_size / 1e6 if tmp.exists() else 0:.1f} MB")
            if attempt == retries:
                raise

    raise RuntimeError(f"could not download {url}")


def ensure_annotations(cache_dir: Path) -> Path:
    """Return the path to ``instances_val2017.json``, downloading it if needed."""
    json_path = cache_dir / "instances_val2017.json"
    if json_path.exists():
        return json_path

    archive = cache_dir / "annotations_trainval2017.zip"
    download(ANNOTATIONS_ZIP, archive, "annotations zip")

    print("  extracting instances_val2017.json ...")
    with zipfile.ZipFile(archive) as zf:
        member = "annotations/instances_val2017.json"
        if member not in zf.namelist():
            raise RuntimeError(f"{member} not found in the annotations archive")
        with zf.open(member) as source, json_path.open("wb") as target:
            target.write(source.read())
    print(f"  annotations ready ({json_path.stat().st_size / 1e6:.1f} MB)")
    return json_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--num-images", type=int, default=100)
    parser.add_argument("--out", default=str(REPO_ROOT / "data" / "coco_subset"))
    parser.add_argument("--cache", default=str(REPO_ROOT / "data" / "_coco_cache"))
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("Preparing the fixed COCO validation subset")
    print("=" * 74)

    annotations_path = ensure_annotations(cache_dir)
    with annotations_path.open(encoding="utf-8") as handle:
        coco = json.load(handle)

    images: List[Dict] = sorted(coco["images"], key=lambda item: item["id"])
    selected = images[: args.num_images]
    selected_ids = {image["id"] for image in selected}
    print(f"  selected {len(selected)} images "
          f"(ids {selected[0]['id']} .. {selected[-1]['id']})")

    for index, image in enumerate(selected, 1):
        target = images_dir / image["file_name"]
        if target.exists():
            continue
        if index == 1 or index % 20 == 0 or index == len(selected):
            print(f"  images: {index}/{len(selected)}")
        try:
            urllib.request.urlretrieve(IMAGE_URL.format(file_name=image["file_name"]), target)
        except Exception as error:  # noqa: BLE001
            print(f"  WARNING: failed to download {image['file_name']}: {error}",
                  file=sys.stderr)

    present = {path.name for path in images_dir.glob("*.jpg")}
    kept_images = [image for image in selected if image["file_name"] in present]
    kept_ids = {image["id"] for image in kept_images}
    print(f"  downloaded {len(kept_images)}/{len(selected)} images")

    annotations = [a for a in coco["annotations"] if a["image_id"] in kept_ids]
    categories = coco["categories"]

    subset = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": kept_images,
        "annotations": annotations,
        "categories": categories,
    }
    annotations_out = out_dir / "annotations.json"
    with annotations_out.open("w", encoding="utf-8") as handle:
        json.dump(subset, handle)

    manifest = {
        "dataset": "COCO 2017 validation",
        "selection": "val2017 images sorted by id, first N",
        "num_images": len(kept_images),
        "num_annotations": len(annotations),
        "image_ids": sorted(kept_ids),
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print()
    print(f"subset written to {out_dir}")
    print(f"  images      : {len(kept_images)}")
    print(f"  annotations : {len(annotations)}")
    print(f"  manifest    : {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
