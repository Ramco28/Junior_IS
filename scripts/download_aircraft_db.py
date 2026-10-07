"""Feature 18: download the OpenSky aircraft database once (about 110 MB).

    python scripts/download_aircraft_db.py

It saves the two CSV files in data/aircraft/ (gitignored) and writes a slim
copy, aircraft_db.parquet, with only the columns I need. The slim copy loads
in a fraction of a second, the full CSV takes several seconds.
"""
import pandas as pd
import requests

import _path  # noqa: F401  (lets me import adsb from here)
from adsb.config import (AIRCRAFT_DB_CSV, AIRCRAFT_DB_FILE, AIRCRAFT_DIR,
                         AIRCRAFT_TYPES_CSV, METADATA_URL)


def download(name: str):
    """Download one file from OpenSky's metadata folder, unless I already have it."""
    path = AIRCRAFT_DIR / name
    if path.exists():
        print(f"Already downloaded: {path.name}")
        return path
    print(f"Downloading {name}...")
    # stream=True reads the file piece by piece, so 110 MB never sits in memory at once
    with requests.get(f"{METADATA_URL}/{name}", stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1_000_000):
                f.write(chunk)
    return path


def main():
    AIRCRAFT_DIR.mkdir(parents=True, exist_ok=True)
    db_csv = download(AIRCRAFT_DB_CSV)
    types_csv = download(AIRCRAFT_TYPES_CSV)

    # the database quotes text with single quotes; I read everything as text
    db = pd.read_csv(db_csv, quotechar="'", dtype=str, usecols=[
        "icao24", "registration", "typecode", "manufacturerName", "model",
        "icaoAircraftClass", "categoryDescription"])
    db = db.rename(columns={"manufacturerName": "manufacturer",
                            "icaoAircraftClass": "icao_class",
                            "categoryDescription": "category_description"})
    db["icao24"] = db["icao24"].str.lower()
    db = db.dropna(subset=["icao24"]).drop_duplicates(subset="icao24")

    # ICAO's list of aircraft types: for each type code (like C172 or B738) it
    # gives the class code (like L1P) and the wake turbulence category (L, M, H, J)
    types = pd.read_csv(types_csv, dtype=str)[["Designator", "Description", "WTC"]]
    types = types.drop_duplicates(subset="Designator").set_index("Designator")

    # when the database has a type code but no class code, I take the class from ICAO's list
    db["icao_class"] = db["icao_class"].fillna(db["typecode"].map(types["Description"]))
    db["wtc"] = db["typecode"].map(types["WTC"])

    db.to_parquet(AIRCRAFT_DB_FILE, index=False)
    print(f"Aircraft in the database: {len(db):,}")
    print(f"With a class code:        {db['icao_class'].notna().sum():,}")
    print(f"Saved {AIRCRAFT_DB_FILE}")


if __name__ == "__main__":
    main()
