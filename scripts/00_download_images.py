from __future__ import annotations

import argparse
import math
from io import BytesIO

import requests
from PIL import Image, ImageOps

from pipeline_utils import coco, image_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the 1,500 official TACO images.")
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()
    data = coco()
    session = requests.Session()
    failures: list[str] = []
    total = len(data["images"])
    for index, image in enumerate(data["images"], 1):
        target = image_path(image)
        if target.exists():
            try:
                with Image.open(target) as current:
                    current.verify()
                with Image.open(target) as current:
                    actual_ratio = current.width / current.height
                expected_ratio = image["width"] / image["height"]
                if abs(math.log(actual_ratio / expected_ratio)) < 0.01:
                    continue
            except Exception:
                pass
        target.parent.mkdir(parents=True, exist_ok=True)
        error = None
        for url_key in ("flickr_url", "flickr_640_url"):
            if not image.get(url_key):
                continue
            try:
                response = session.get(image[url_key], timeout=args.timeout)
                response.raise_for_status()
                loaded = Image.open(BytesIO(response.content))
                loaded.load()
                loaded = ImageOps.exif_transpose(loaded)
                if loaded.mode not in ("RGB", "L"):
                    loaded = loaded.convert("RGB")
                actual_ratio = loaded.width / loaded.height
                expected_ratio = image["width"] / image["height"]
                if abs(math.log(actual_ratio / expected_ratio)) >= 0.01:
                    raise ValueError(f"aspect ratio {loaded.size} does not match annotation {(image['width'], image['height'])}")
                loaded.save(target, quality=95)
                error = None
                break
            except Exception as exc:  # report all failures together
                error = exc
        if error is not None:
            failures.append(f"{image['file_name']}: {error}")
        if index % 25 == 0 or index == total:
            print(f"Official TACO images: {index}/{total} (failures: {len(failures)})", flush=True)
    if failures:
        raise RuntimeError("Could not download:\n" + "\n".join(failures))
    print(f"Ready: {len(data['images'])} official images with annotation-aligned orientation")


if __name__ == "__main__":
    main()
