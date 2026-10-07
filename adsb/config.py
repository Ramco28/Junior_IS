"""Settings I use in more than one file: folders, URLs, Trino login and the Ohio box."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
HISTORICAL_DIR = DATA_DIR / "historical"
FEATURES_DIR = DATA_DIR / "features"
CREDENTIALS_FILE = ROOT / "credentials.json"

# Live data: the OpenSky REST API, and the server that gives me login tokens
API_BASE = "https://opensky-network.org/api"
TOKEN_URL = ("https://auth.opensky-network.org/auth/realms/opensky-network"
             "/protocol/openid-connect/token")

# Historical data: OpenSky's Trino database (same settings as my Trino CLI)
TRINO_HOST = "trino.opensky-network.org"
TRINO_PORT = 443
TRINO_CATALOG = "minio"
TRINO_SCHEMA = "osky"
TRINO_USER = "ramco_28"

# Aircraft metadata: OpenSky's public aircraft database (who is behind an icao24)
# and the ICAO list of aircraft types. I download both once into data/aircraft/.
AIRCRAFT_DIR = DATA_DIR / "aircraft"
AIRCRAFT_DB_FILE = AIRCRAFT_DIR / "aircraft_db.parquet"  # my slim copy, fast to load
METADATA_URL = "https://s3.opensky-network.org/data-samples/metadata"
AIRCRAFT_DB_CSV = "aircraft-database-complete-2025-08.csv"  # newest file OpenSky publishes
AIRCRAFT_TYPES_CSV = "doc8643AircraftTypes.csv"

# /states/all sends each aircraft as a plain list with no names, so I keep the
# field names here in the same order as the OpenSky docs to label the columns
STATE_FIELDS = [
    "icao24", "callsign", "origin_country", "time_position", "last_contact",
    "longitude", "latitude", "baro_altitude", "on_ground", "velocity",
    "true_track", "vertical_rate", "sensors", "geo_altitude", "squawk",
    "spi", "position_source", "category",
]

# I test on Ohio because a small box costs fewer API credits than the whole world
# Order: (lat_min, lat_max, lon_min, lon_max)
OHIO_BBOX = (38.4, 42.3, -84.8, -80.5)
