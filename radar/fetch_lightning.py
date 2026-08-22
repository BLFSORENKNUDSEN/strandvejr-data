from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests

API = "https://opendataapi.dmi.dk/v2/lightningdata/collections/observation/items"
OUT = Path(os.getenv("RADAR_OUT", "radar/site/data"))
TIMEOUT = 30

# Dækker Danmark og de nærmeste havområder, så lyn på vej mod landet også ses.
BBOX = os.getenv("LIGHTNING_BBOX", "6.5,53.5,16.5,58.5")
LIMIT = int(os.getenv("LIGHTNING_LIMIT", "20000"))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    response = requests.get(
        API,
        params={
            "period": "latest-hour",
            "bbox": BBOX,
            "limit": LIMIT,
            "sortorder": "observed,DESC",
        },
        timeout=TIMEOUT,
        headers={"User-Agent": "strandvejr-lightning/1.0"},
    )
    response.raise_for_status()
    payload = response.json()

    strikes = []
    for feature in payload.get("features", []):
        geometry = feature.get("geometry") or {}
        coords = geometry.get("coordinates") or []
        props = feature.get("properties") or {}
        if len(coords) < 2:
            continue
        observed = props.get("observed")
        if not observed:
            continue
        strikes.append({
            "id": feature.get("id"),
            "lon": float(coords[0]),
            "lat": float(coords[1]),
            "observed": observed,
            "amp": props.get("amp"),
            "strokes": props.get("strokes"),
            "type": props.get("type"),
        })

    result = {
        "generated": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": "DMI Lightning Data API",
        "periodMinutes": 60,
        "bbox": [float(v) for v in BBOX.split(",")],
        "count": len(strikes),
        "strikes": strikes,
    }

    (OUT / "lightning.json").write_text(
        json.dumps(result, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"Skrev {len(strikes)} lynnedslag til {OUT / 'lightning.json'}")


if __name__ == "__main__":
    main()
