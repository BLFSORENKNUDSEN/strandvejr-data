from __future__ import annotations

import io
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import requests
from PIL import Image

API = "https://opendataapi.dmi.dk/v1/radardata/collections/composite/items"
OUT = Path(os.getenv("RADAR_OUT", "radar/site/data"))
FRAMES = OUT / "frames"
FRAME_COUNT = int(os.getenv("RADAR_FRAME_COUNT", "13"))
TIMEOUT = 45

# Radar reflectivity palette in 5 dBZ steps.
BREAKS = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75], dtype=float)
COLORS = np.array([
    [64, 240, 240, 255],
    [65, 184, 248, 255],
    [64, 64, 248, 255],
    [64, 255, 64, 255],
    [64, 213, 64, 255],
    [64, 172, 64, 255],
    [255, 255, 64, 255],
    [237, 207, 64, 255],
    [255, 172, 64, 255],
    [255, 64, 64, 255],
    [224, 64, 64, 255],
    [207, 64, 64, 255],
    [255, 64, 255, 255],
    [178, 128, 214, 255],
    [64, 64, 64, 255],
], dtype=np.uint8)


def decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").rstrip("\x00")
    return value.item() if hasattr(value, "item") else value


def attrs(group):
    return {str(k): decode(v) for k, v in group.attrs.items()}


def group_meta(group: h5py.Group) -> dict:
    meta = {}
    if "what" in group and isinstance(group["what"], h5py.Group):
        meta.update(attrs(group["what"]))
    meta.update(attrs(group))
    return meta


def locate_reflectivity(h5: h5py.File):
    candidates = []

    def visit(name, obj):
        if not isinstance(obj, h5py.Group):
            return
        if "data" not in obj or not isinstance(obj["data"], h5py.Dataset):
            return

        # DMI composite files do not always place all ODIM metadata in the
        # same group. Collect metadata from the data group and its ancestors.
        # More specific metadata wins over values inherited from a parent.
        lineage = []
        current = obj
        while isinstance(current, h5py.Group):
            lineage.append(current)
            if current.name == "/":
                break
            current = current.parent

        meta = {}
        for group in reversed(lineage):
            meta.update(group_meta(group))

        quantity = str(meta.get("quantity", "")).upper()
        score = 0 if quantity == "DBZH" else 1 if "DBZ" in quantity else 2
        candidates.append((score, name, obj["data"], meta))

    h5.visititems(visit)
    if not candidates:
        raise RuntimeError("Ingen radar raster blev fundet i HDF5 filen")
    candidates.sort(key=lambda x: x[0])
    return candidates[0][2], candidates[0][3]


def dbz_from_h5(raw_bytes: bytes) -> np.ndarray:
    with h5py.File(io.BytesIO(raw_bytes), "r") as h5:
        dataset, meta = locate_reflectivity(h5)
        raw = dataset[...]

        # ODIM encodes physical values as raw * gain + offset.
        # DMI's composite raster is uint8. Older/current DMI composite files
        # may omit these values at data1 level, so use the established DMI
        # DBZH count conversion as a defensive fallback rather than treating
        # the byte value itself as dBZ.
        gain = float(meta.get("gain", 0.5))
        offset = float(meta.get("offset", -32.0))
        nodata = meta.get("nodata", 255)
        undetect = meta.get("undetect", 0)

        z = raw.astype(np.float32) * gain + offset
        invalid = np.zeros(raw.shape, dtype=bool)
        if nodata is not None:
            invalid |= raw == nodata
        if undetect is not None:
            invalid |= raw == undetect
        z[invalid] = np.nan
        return z


def colorize(z: np.ndarray) -> Image.Image:
    rgba = np.zeros((*z.shape, 4), dtype=np.uint8)
    valid = np.isfinite(z) & (z >= BREAKS[0])
    idx = np.searchsorted(BREAKS, z, side="right") - 1
    idx = np.clip(idx, 0, len(COLORS) - 1)
    rgba[valid] = COLORS[idx[valid]]
    return Image.fromarray(rgba, mode="RGBA")


def fetch_features():
    response = requests.get(
        API,
        params={"limit": 80, "scanType": "fullRange", "sortorder": "datetime,DESC"},
        timeout=TIMEOUT,
        headers={"User-Agent": "strandvejr-radar/1.0"},
    )
    response.raise_for_status()
    features = response.json().get("features", [])
    features.sort(key=lambda f: f.get("properties", {}).get("datetime", ""), reverse=True)
    return features[:FRAME_COUNT]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if FRAMES.exists():
        shutil.rmtree(FRAMES)
    FRAMES.mkdir(parents=True)

    frames = []
    for feature in reversed(fetch_features()):
        dt = feature["properties"]["datetime"]
        href = feature["asset"]["data"]["href"]
        bbox = feature["bbox"]
        file_id = feature["id"]

        r = requests.get(href, timeout=TIMEOUT, headers={"User-Agent": "strandvejr-radar/1.0"})
        r.raise_for_status()
        z = dbz_from_h5(r.content)
        image = colorize(z)

        stamp = datetime.fromisoformat(dt.replace("Z", "+00:00")).astimezone(timezone.utc)
        filename = stamp.strftime("%Y%m%dT%H%M%SZ.png")
        image.save(FRAMES / filename, optimize=True)

        frames.append({
            "time": dt,
            "file": f"data/frames/{filename}",
            "bbox": bbox,
            "source": file_id,
        })

    if not frames:
        raise RuntimeError("DMI API returnerede ingen full range radar scans")

    manifest = {
        "generated": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "frameCount": len(frames),
        "frames": frames,
        "legend": [{"dbz": int(v), "rgba": c.tolist()} for v, c in zip(BREAKS, COLORS)],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Skrev {len(frames)} radarframes til {OUT}")


if __name__ == "__main__":
    main()
