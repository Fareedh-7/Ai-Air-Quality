from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from geopy.geocoders import Nominatim, RateLimiter


CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
SHORT_NAME = "MOD04_L2"
CACHE_DIR = Path("data/modis_cache")


class ModisLiveError(RuntimeError):
    pass


def _earthdata_auth() -> tuple[str, str]:
    username = os.getenv("EARTHDATA_USERNAME")
    password = os.getenv("EARTHDATA_PASSWORD")
    if not username or not password:
        raise ModisLiveError(
            "EARTHDATA_USERNAME and EARTHDATA_PASSWORD are required for live MODIS."
        )
    return username, password


def geocode_city(city: str) -> tuple[float, float]:
    geolocator = Nominatim(user_agent="ai-air-quality-modis")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    location = geocode(city)
    if not location:
        raise ModisLiveError(f"Unable to geocode city: {city}")
    return float(location.latitude), float(location.longitude)


def _time_window(date: datetime | None) -> str:
    if date is None:
        date = datetime.now(timezone.utc)
    start = datetime(date.year, date.month, date.day, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return f"{start.isoformat()},{end.isoformat()}"


def search_granule(lat: float, lon: float, date: datetime | None = None) -> dict:
    bbox = f"{lon - 0.5},{lat - 0.5},{lon + 0.5},{lat + 0.5}"
    params = {
        "short_name": SHORT_NAME,
        "temporal": _time_window(date),
        "bounding_box": bbox,
        "page_size": 1,
        "sort_key": "-start_date",
    }
    resp = requests.get(CMR_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    entries = data.get("feed", {}).get("entry", [])
    if not entries:
        raise ModisLiveError("No MODIS granules found for this location/date.")

    entry = entries[0]
    links = entry.get("links", [])
    url = ""
    for link in links:
        href = link.get("href", "")
        if href.endswith(".hdf"):
            url = href
            break
    if not url:
        for link in links:
            if link.get("rel", "").endswith("/data#"):
                url = link.get("href", "")
                break
    if not url:
        raise ModisLiveError("No downloadable HDF link found for MODIS granule.")

    return {
        "granule_id": entry.get("title", ""),
        "time_start": entry.get("time_start", ""),
        "download_url": url,
    }


def download_granule(url: str, auth: tuple[str, str]) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    filename = url.split("?")[0].split("/")[-1]
    out_path = CACHE_DIR / filename

    if out_path.exists():
        return out_path

    with requests.get(url, stream=True, auth=auth, timeout=60) as resp:
        if resp.status_code == 401:
            raise ModisLiveError("Earthdata authentication failed (401).")
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return out_path


def _apply_scale(data: np.ndarray, attrs: dict) -> np.ndarray:
    scale = attrs.get("scale_factor", 1.0)
    offset = attrs.get("add_offset", 0.0)
    return data.astype(np.float32) * float(scale) + float(offset)


def read_aod_from_hdf(hdf_path: Path, lat: float, lon: float) -> float:
    try:
        from pyhdf.SD import SD, SDC
    except ImportError as exc:
        raise ModisLiveError(
            "pyhdf is required to read MODIS HDF4 files. Install pyhdf."
        ) from exc

    hdf = SD(str(hdf_path), SDC.READ)

    aod_ds = hdf.select("Optical_Depth_Land_And_Ocean")
    aod = _apply_scale(aod_ds[:], aod_ds.attributes())

    lat_ds = hdf.select("Latitude")
    lon_ds = hdf.select("Longitude")
    lat_arr = lat_ds[:].astype(np.float32)
    lon_arr = lon_ds[:].astype(np.float32)

    fill = aod_ds.attributes().get("_FillValue")
    if fill is not None:
        aod = np.where(aod == fill, np.nan, aod)

    dist = (lat_arr - lat) ** 2 + (lon_arr - lon) ** 2
    idx = np.unravel_index(np.nanargmin(dist), dist.shape)
    value = float(aod[idx])
    if np.isnan(value):
        raise ModisLiveError("MODIS AOD is missing at the nearest pixel.")

    return value


def fetch_live_aod(
    city: str,
    out_csv: Path | None = None,
    date: datetime | None = None,
) -> dict:
    lat, lon = geocode_city(city)
    granule = search_granule(lat, lon, date)
    auth = _earthdata_auth()
    hdf_path = download_granule(granule["download_url"], auth)
    aod = read_aod_from_hdf(hdf_path, lat, lon)

    payload = {
        "city": city,
        "date": granule["time_start"][:10] if granule["time_start"] else "",
        "latitude": lat,
        "longitude": lon,
        "modis_aod": aod,
        "granule_id": granule["granule_id"],
        "source_url": granule["download_url"],
    }

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([payload]).to_csv(out_csv, index=False)

    return payload