#!/usr/bin/env python3
"""Fetch ECMWF Open Data wind for Strandvejr.

Produces one compact JSON file with 10 m wind and pressure level wind
for 850, 500 and 250 hPa.
"""
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone

from ecmwf.opendata import Client
from eccodes import codes_grib_new_from_file, codes_get, codes_get_array, codes_release

WEST = -12.0
EAST = 32.0
SOUTH = 48.0
NORTH = 72.5
DOWNSAMPLE = 2
STEP = 0
OUTPUT = os.environ.get("ECMWF_WIND_OUTPUT", "ecmwf_wind.json")
PRESSURE_LEVELS = (850, 500, 250)


def normalise_lon(lon):
    lon = float(lon)
    if lon > 180.0:
        lon -= 360.0
    return lon


def iso_utc(dt):
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def existing_run():
    try:
        with open(OUTPUT, "r", encoding="utf-8") as fh:
            return json.load(fh).get("run")
    except Exception:
        return None


def collect_grib(path):
    fields = {}
    with open(path, "rb") as fh:
        while True:
            gid = codes_grib_new_from_file(fh)
            if gid is None:
                break
            try:
                short = str(codes_get(gid, "shortName"))
                level = int(codes_get(gid, "level")) if short in ("u", "v") else 10
                key = (short, level)
                if short not in ("10u", "10v", "u", "v"):
                    continue
                if short in ("u", "v") and level not in PRESSURE_LEVELS:
                    continue
                lats = codes_get_array(gid, "latitudes")
                lons = codes_get_array(gid, "longitudes")
                vals = codes_get_array(gid, "values")
                pts = {}
                for lat, lon, value in zip(lats, lons, vals):
                    lat = round(float(lat), 6)
                    lon = round(normalise_lon(lon), 6)
                    if SOUTH <= lat <= NORTH and WEST <= lon <= EAST:
                        pts[(lat, lon)] = float(value)
                fields[key] = pts
            finally:
                codes_release(gid)
    return fields


def build_grid(u_points, v_points):
    common = set(u_points.keys()) & set(v_points.keys())
    if not common:
        raise RuntimeError("Ingen faelles u/v gitterpunkter")
    all_lats = sorted({p[0] for p in common}, reverse=True)
    all_lons = sorted({p[1] for p in common})
    lats = all_lats[::DOWNSAMPLE]
    lons = all_lons[::DOWNSAMPLE]
    if len(lats) < 2 or len(lons) < 2:
        raise RuntimeError("For faa gitterpunkter")
    u = []
    v = []
    for lat in lats:
        for lon in lons:
            key = (lat, lon)
            u.append(round(u_points.get(key, 0.0), 3))
            v.append(round(v_points.get(key, 0.0), 3))
    return {
        "grid": {
            "west": lons[0], "east": lons[-1],
            "north": lats[0], "south": lats[-1],
            "dx": round(abs(lons[1] - lons[0]), 6),
            "dy": round(abs(lats[0] - lats[1]), 6),
            "nx": len(lons), "ny": len(lats)
        },
        "u": u, "v": v, "units": "m/s"
    }


def main():
    out_dir = os.path.dirname(OUTPUT)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    client = Client(source="ecmwf", model="ifs", resol="0p25")
    latest = client.latest(type="fc", step=STEP, param=["10u", "10v"])
    latest_iso = iso_utc(latest)
    if existing_run() == latest_iso:
        print("ECMWF vind er allerede opdateret:", latest_iso)
        return 0

    fd, surface_path = tempfile.mkstemp(prefix="ecmwf_surface_", suffix=".grib2")
    os.close(fd)
    fd, pressure_path = tempfile.mkstemp(prefix="ecmwf_pressure_", suffix=".grib2")
    os.close(fd)

    try:
        result = client.retrieve(type="fc", step=STEP, param=["10u", "10v"], target=surface_path)
        client.retrieve(
            type="fc",
            step=STEP,
            param=["u", "v"],
            levelist=list(PRESSURE_LEVELS),
            target=pressure_path,
        )

        fields = collect_grib(surface_path)
        fields.update(collect_grib(pressure_path))

        levels = {
            "10m": build_grid(fields[("10u", 10)], fields[("10v", 10)])
        }
        for level in PRESSURE_LEVELS:
            levels[str(level)] = build_grid(fields[("u", level)], fields[("v", level)])

        run = result.datetime
        valid = run + timedelta(hours=STEP)
        out = {
            "source": "ECMWF Open Data IFS",
            "model": "IFS",
            "resolution_source": "0.25 degree",
            "run": iso_utc(run),
            "valid": iso_utc(valid),
            "step_hours": STEP,
            "generated": iso_utc(datetime.now(timezone.utc)),
            "attribution": "ECMWF Open Data, CC BY 4.0",
            "levels": levels,
        }

        tmp = OUTPUT + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp, OUTPUT)
        print("Skrev", OUTPUT, "run", out["run"], "levels", ",".join(levels.keys()))
        return 0
    finally:
        for path in (surface_path, pressure_path):
            try:
                os.unlink(path)
            except OSError:
                pass


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("FEJL:", exc, file=sys.stderr)
        sys.exit(1)
