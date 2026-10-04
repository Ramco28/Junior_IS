"""Getting live aircraft positions from the OpenSky REST API (features 1 and 2)."""
import time

import pandas as pd
import requests

from .config import API_BASE, STATE_FIELDS


class RateLimited(Exception):
    """OpenSky answers with HTTP 429 when I run out of daily API credits."""

    def __init__(self, retry_after_s: int):
        super().__init__(f"Rate limited, retry after {retry_after_s} s")
        self.retry_after_s = retry_after_s


def fetch_states(auth, bbox=None) -> dict:
    """Get one snapshot of every aircraft OpenSky is tracking right now.

    auth is a TokenManager or AnonymousAccess (see auth.get_auth). Its
    headers() gives the Authorization header, or nothing in anonymous mode.
    bbox is (lat_min, lat_max, lon_min, lon_max), or None for the whole world.
    A small box like Ohio costs fewer credits. I return the JSON the way it
    comes, {"time": ..., "states": [[...], ...]}, and add "credits_remaining"
    when OpenSky sends it.
    """
    # extended=1 asks OpenSky to add the aircraft category (light airplane,
    # rotorcraft, ...) as an 18th field. Without it the category is left out.
    params = {"extended": 1}
    if bbox is not None:
        params.update(zip(["lamin", "lamax", "lomin", "lomax"], bbox))
    resp = requests.get(f"{API_BASE}/states/all", params=params,
                        headers=auth.headers(), timeout=30)
    if resp.status_code == 429:
        # OpenSky tells me how many seconds to wait before trying again
        retry = int(resp.headers.get("X-Rate-Limit-Retry-After-Seconds", 600))
        raise RateLimited(retry)
    resp.raise_for_status()
    data = resp.json()
    data["states"] = data.get("states") or []  # OpenSky sends null instead of [] when there are no aircraft
    data["fetched_at"] = time.time()
    remaining = resp.headers.get("X-Rate-Limit-Remaining")  # credits I have left today
    if remaining is not None:
        data["credits_remaining"] = int(remaining)
    return data


def states_to_frame(snapshot: dict) -> pd.DataFrame:
    """Put a snapshot in a pandas table with named columns so I can print it.

    This only names the columns. The real cleaning (units, missing values)
    is feature 4.
    """
    # some rows are shorter than STATE_FIELDS (no category), so I pad them with None
    rows = [row + [None] * (len(STATE_FIELDS) - len(row)) for row in snapshot["states"]]
    df = pd.DataFrame(rows, columns=STATE_FIELDS)
    df["callsign"] = df["callsign"].str.strip()
    return df
