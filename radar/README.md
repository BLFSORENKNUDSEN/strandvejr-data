# Strandvejr radar

Løsningen henter DMI Frie Data fra Radar Data API og bygger et animeret radarkort til Strandvejr.

## Sådan virker den

1. GitHub Actions kører hvert 10. minut.
2. Scriptet henter de seneste 13 `fullRange` composite scanninger fra DMI.
3. HDF5 radarreflektivitet læses som `DBZH` og farvelægges i transparente PNG billeder.
4. Der oprettes en `manifest.json` med tid, geografisk afgrænsning og radarframes.
5. GitHub Pages publicerer kortet.
6. Websiden afspiller billederne som en animation med tidslinje og pauseknap.

DMI oplyser, at composite data har 500 meter pixelstørrelse. Full range scanninger kommer hvert 10. minut og har større dækningsområde end Doppler scanningerne.

## Aktivering

Efter merge vælges GitHub Actions som kilde under repository Settings, Pages. Workflowet `Strandvejr radar` kan derefter startes manuelt første gang. Herefter opdateres radarvisningen automatisk.

## Filer

`fetch_radar.py` henter og renderer radardata.

`site/index.html` viser den animerede radar.

`requirements.txt` indeholder Python afhængigheder.

`.github/workflows/radar.yml` står for automatisk opdatering og publicering.

## Data

Kilde: DMI Frie Data, Radar Data API.
