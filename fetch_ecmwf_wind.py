#!/usr/bin/env python3
"""
Henter seneste ECMWF IFS 10 m vind fra ECMWF Open Data og skriver en
kompakt JSON fil til Strandvejr.

Kraever:
    pip3 install ecmwf-opendata eccodes

Cron eksempel, hvert 30. minut:
    */30 * * * * /usr/bin/python3 /var/www/strandvejr.dk/scripts/fetch_ecmwf_wind.py >> /var/log/ecmwf_wind.log 2>&1
"""

import json
import math
import os
import sys
import tempfile
from datetime import timedelta, timezone

from ecmwf.opendata import Client
from eccodes import codes_grib_new_from_file, codes_get, codes_get_array, codes_release

# Omraade: Nordeuropa og Skandinavien
WEST = -12.0
EAST = 32.0
SOUTH = 48.0
NORTH = 72.5

# ECMWF Open Data er 0.25 grader. 2 giver 0.50 graders output og et let overlay.
DOWNSAMPLE = 2
STEP = 0

# Tilpas denne sti hvis scriptet ligger et andet sted.
OUTPUT = os.environ.get("ECMWF_WIND_OUTPUT", "ecmwf_wind.json")


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
            obj = json.load(fh)
        return obj.get("run")
    except Exception:
        return None


def read_grib(path):
    fields = {}
    meta = {}

    with open(path, "rb") as fh:
        while True:
            gid = codes_grib_new_from_file(fh)
            if gid is None:
                break
            try:
                short_name = str(codes_get(gid, "shortName"))
                if short_name not in ("10u", "10v"):
                    continue

                lats = codes_get_array(gid, "latitudes")
                lons = codes_get_array(gid, "longitudes")
                vals = codes_get_array(gid, "values")

                points = {}
                for lat, lon, value in zip(lats, lons, vals):
                    lat = round(float(lat), 6)
                    lon = round(normalise_lon(lon), 6)
                    if SOUTH <= lat <= NORTH and WEST <= lon <= EAST:
                        points[(lat, lon)] = float(value)

                fields[short_name] = points
                meta[short_name] = {
                    "units": str(codes_get(gid, "units")),
                    "step": int(codes_get(gid, "step")),
                }
            finally:
                codes_release(gid)

    if "10u" not in fields or "10v" not in fields:
        raise RuntimeError("GRIB filen indeholder ikke baade 10u og 10v")

    common = set(fields["10u"].keys()) & set(fields["10v"].keys())
    if not common:
        raise RuntimeError("Ingen faelles 10u/10v gitterpunkter i det valgte omraade")

    all_lats = sorted({p[0] for p in common}, reverse=True)
    all_lons = sorted({p[1] for p in common})

    lats = all_lats[::DOWNSAMPLE]
    lons = all_lons[::DOWNSAMPLE]

    if len(lats) < 2 or len(lons) < 2:
        raise RuntimeError("For faa gitterpunkter efter filtrering")

    u = []
    v = []
    for lat in lats:
        for lon in lons:
            key = (lat, lon)
            u.append(round(fields["10u"].get(key, 0.0), 3))
            v.append(round(fields["10v"].get(key, 0.0), 3))

    dx = round(abs(lons[1] - lons[0]), 6)
    dy = round(abs(lats[0] - lats[1]), 6)

    return {
        "grid": {
            "west": lons[0],
            "east": lons[-1],
            "north": lats[0],
            "south": lats[-1],
            "dx": dx,
            "dy": dy,
            "nx": len(lons),
            "ny": len(lats),
        },
        "u": u,
        "v": v,
        "units": "m/s",
    }


def main():
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

    client = Client(source="ecmwf", model="ifs", resol="0p25")

    latest = client.latest(
        type="fc",
        step=STEP,
        param=["10u", "10v"],
    )

    latest_iso = iso_utc(latest)
    if existing_run() == latest_iso:
        print("ECMWF vind er allerede opdateret:", latest_iso)
        return 0

    fd, grib_path = tempfile.mkstemp(prefix="ecmwf_wind_", suffix=".grib2")
    os.close(fd)

    try:
        result = client.retrieve(
            type="fc",
            step=STEP,
            param=["10u", "10v"],
            target=grib_path,
        )

        grid = read_grib(grib_path)
        run = result.datetime
        valid = run + timedelta(hours=STEP)

        out = {
            "source": "ECMWF Open Data IFS",
            "model": "IFS",
            "resolution_source": "0.25 degree",
            "run": iso_utc(run),
            "valid": iso_utc(valid),
            "step_hours": STEP,
            "generated": iso_utc(__import__("datetime").datetime.now(timezone.utc)),
            "attribution": "ECMWF Open Data, CC BY 4.0",
            **grid,
        }

        tmp = OUTPUT + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp, OUTPUT)

        print(
            "Skrev", OUTPUT,
            "run", out["run"],
            "grid", out["grid"]["nx"], "x", out["grid"]["ny"],
        )
        return 0
    finally:
        try:
            os.unlink(grib_path)
        except OSError:
            pass


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("FEJL:", exc, file=sys.stderr)
        sys.exit(1)
