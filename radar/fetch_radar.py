from __future__ import annotations

import io
import json
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import h5py
import numpy as np
import requests
from PIL import Image
from pyproj import CRS, Transformer

API = "https://opendataapi.dmi.dk/v1/radardata/collections/composite/items"
OUT = Path(os.getenv("RADAR_OUT", "radar/site/data"))
FRAMES = OUT / "frames"
FRAME_COUNT = int(os.getenv("RADAR_FRAME_COUNT", "13"))
NOWCAST_MINUTES = int(os.getenv("RADAR_NOWCAST_MINUTES", "60"))
NOWCAST_STEP = int(os.getenv("RADAR_NOWCAST_STEP", "10"))
TIMEOUT = 45

BREAKS = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75], dtype=float)
COLORS = np.array([
    [64, 240, 240, 255], [65, 184, 248, 255], [64, 64, 248, 255],
    [64, 255, 64, 255], [64, 213, 64, 255], [64, 172, 64, 255],
    [255, 255, 64, 255], [237, 207, 64, 255], [255, 172, 64, 255],
    [255, 64, 64, 255], [224, 64, 64, 255], [207, 64, 64, 255],
    [255, 64, 255, 255], [178, 128, 214, 255], [64, 64, 64, 255],
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
        if not isinstance(obj, h5py.Group) or "data" not in obj or not isinstance(obj["data"], h5py.Dataset):
            return
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


def radar_georef(h5: h5py.File, shape: tuple[int, int]) -> dict | None:
    """Read ODIM cartesian georeferencing from the HDF5 /where group."""
    if "where" not in h5 or not isinstance(h5["where"], h5py.Group):
        return None
    where = attrs(h5["where"])
    required = ("projdef", "xscale", "yscale", "LL_lon", "LL_lat")
    if not all(k in where for k in required):
        return None
    return {
        "projdef": str(where["projdef"]),
        "xscale": float(where["xscale"]),
        "yscale": float(where["yscale"]),
        "ll_lon": float(where["LL_lon"]),
        "ll_lat": float(where["LL_lat"]),
        "xsize": int(where.get("xsize", shape[1])),
        "ysize": int(where.get("ysize", shape[0])),
    }


def dbz_from_h5(raw_bytes: bytes) -> tuple[np.ndarray, dict | None]:
    with h5py.File(io.BytesIO(raw_bytes), "r") as h5:
        dataset, meta = locate_reflectivity(h5)
        raw = dataset[...]
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
        return z, radar_georef(h5, z.shape)


def reproject_to_lonlat(z: np.ndarray, georef: dict | None, bbox: list[float]) -> np.ndarray:
    """Warp the ODIM projected raster to a regular lon/lat raster for Leaflet imageOverlay."""
    if not georef:
        return z

    try:
        src_crs = CRS.from_user_input(georef["projdef"])
        to_src = Transformer.from_crs("EPSG:4326", src_crs, always_xy=True)
        llx, lly = to_src.transform(georef["ll_lon"], georef["ll_lat"])
    except Exception as exc:
        print(f"Advarsel: kunne ikke læse radarprojektion: {exc}")
        return z

    h, w = z.shape
    west, south, east, north = [float(v) for v in bbox]
    out = np.full((h, w), np.nan, dtype=np.float32)
    source_values = np.nan_to_num(z, nan=0.0).astype(np.float32)
    source_valid = np.isfinite(z).astype(np.float32)

    lons = west + (np.arange(w, dtype=np.float64) + 0.5) * (east - west) / w
    chunk = 128
    xscale = georef["xscale"]
    yscale = georef["yscale"]
    ysize = georef["ysize"]

    for row0 in range(0, h, chunk):
        row1 = min(h, row0 + chunk)
        rows = np.arange(row0, row1, dtype=np.float64)
        lats = north - (rows + 0.5) * (north - south) / h
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        xs, ys = to_src.transform(lon_grid, lat_grid)

        map_x = ((xs - llx) / xscale - 0.5).astype(np.float32)
        map_y = (ysize - 0.5 - (ys - lly) / yscale).astype(np.float32)

        moved = cv2.remap(
            source_values,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        valid = cv2.remap(
            source_valid,
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        moved[valid < 0.5] = np.nan
        out[row0:row1] = moved

    return out


def colorize(z: np.ndarray) -> Image.Image:
    rgba = np.zeros((*z.shape, 4), dtype=np.uint8)
    valid = np.isfinite(z) & (z >= BREAKS[0])
    idx = np.searchsorted(BREAKS, z, side="right") - 1
    idx = np.clip(idx, 0, len(COLORS) - 1)
    rgba[valid] = COLORS[idx[valid]]
    return Image.fromarray(rgba, mode="RGBA")


def motion_image(z: np.ndarray) -> np.ndarray:
    a = np.nan_to_num(z, nan=0.0)
    a = np.clip(a - 5.0, 0.0, 45.0)
    return cv2.GaussianBlur(a.astype(np.float32), (0, 0), 2.0)


def estimate_motion(history: list[tuple[datetime, np.ndarray]]) -> tuple[float, float, float]:
    estimates = []
    for (t0, z0), (t1, z1) in zip(history[-5:-1], history[-4:]):
        minutes = (t1 - t0).total_seconds() / 60.0
        if minutes <= 0 or minutes > 20:
            continue
        a = motion_image(z0)
        b = motion_image(z1)
        if np.count_nonzero(a > 0.5) < 300 or np.count_nonzero(b > 0.5) < 300:
            continue
        (dx, dy), response = cv2.phaseCorrelate(a, b)
        if not np.isfinite(dx + dy) or response < 0.02:
            continue
        scale = NOWCAST_STEP / minutes
        dx *= scale
        dy *= scale
        if np.hypot(dx, dy) > 50:
            continue
        estimates.append((dx, dy, response))

    if not estimates:
        return 0.0, 0.0, 0.0
    return (
        float(np.median([e[0] for e in estimates])),
        float(np.median([e[1] for e in estimates])),
        float(np.median([e[2] for e in estimates])),
    )


def advect(z: np.ndarray, dx: float, dy: float, factor: float) -> np.ndarray:
    h, w = z.shape
    values = np.nan_to_num(z, nan=0.0).astype(np.float32)
    valid = np.isfinite(z).astype(np.float32)
    matrix = np.float32([[1, 0, dx * factor], [0, 1, dy * factor]])
    moved = cv2.warpAffine(values, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    moved_valid = cv2.warpAffine(valid, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    moved[moved_valid < 0.5] = np.nan
    return moved


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
    history = []
    bbox = None
    latest_georef = None

    for feature in reversed(fetch_features()):
        dt_text = feature["properties"]["datetime"]
        dt = datetime.fromisoformat(dt_text.replace("Z", "+00:00")).astimezone(timezone.utc)
        href = feature["asset"]["data"]["href"]
        bbox = feature["bbox"]
        file_id = feature["id"]
        r = requests.get(href, timeout=TIMEOUT, headers={"User-Agent": "strandvejr-radar/1.0"})
        r.raise_for_status()
        z, georef = dbz_from_h5(r.content)
        latest_georef = georef or latest_georef
        history.append((dt, z))

        filename = dt.strftime("%Y%m%dT%H%M%SZ.png")
        display_z = reproject_to_lonlat(z, georef, bbox)
        colorize(display_z).save(FRAMES / filename, optimize=True)
        frames.append({
            "time": dt.isoformat().replace("+00:00", "Z"),
            "file": f"data/frames/{filename}",
            "bbox": bbox,
            "source": file_id,
            "kind": "observation",
            "leadMinutes": 0,
        })

    if not frames:
        raise RuntimeError("DMI API returnerede ingen full range radar scans")

    dx, dy, confidence = estimate_motion(history)
    latest_time, latest_z = history[-1]
    forecast_count = max(0, NOWCAST_MINUTES // NOWCAST_STEP)
    for step in range(1, forecast_count + 1):
        lead = step * NOWCAST_STEP
        forecast_time = latest_time + timedelta(minutes=lead)
        forecast_z = advect(latest_z, dx, dy, step)
        filename = forecast_time.strftime("%Y%m%dT%H%M%SZ_nowcast.png")
        display_z = reproject_to_lonlat(forecast_z, latest_georef, bbox)
        colorize(display_z).save(FRAMES / filename, optimize=True)
        frames.append({
            "time": forecast_time.isoformat().replace("+00:00", "Z"),
            "file": f"data/frames/{filename}",
            "bbox": bbox,
            "source": "strandvejr-nowcast",
            "kind": "forecast",
            "leadMinutes": lead,
        })

    manifest = {
        "generated": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "frameCount": len(frames),
        "observationCount": len(history),
        "forecastCount": forecast_count,
        "georeferencing": "ODIM projection reprojected to regular lonlat grid",
        "nowcast": {
            "method": "radar-advection-phase-correlation",
            "horizonMinutes": NOWCAST_MINUTES,
            "stepMinutes": NOWCAST_STEP,
            "motionPixelsPerStep": {"x": round(dx, 3), "y": round(dy, 3)},
            "confidence": round(confidence, 3),
            "note": "Ekstrapolation af seneste radarobservation. Byger kan vokse eller aftage hurtigere end modellen viser.",
        },
        "frames": frames,
        "legend": [{"dbz": int(v), "rgba": c.tolist()} for v, c in zip(BREAKS, COLORS)],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Skrev {len(history)} observationer og {forecast_count} prognoseframes med geografisk reprojektion")


if __name__ == "__main__":
    main()
