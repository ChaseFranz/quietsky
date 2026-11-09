"""Pre-recording exposure check for YOUR OWN recording spot.

Reads a flight-path GeoJSON produced by rannmann/flight-path-mapper
(one MultiLineString per aircraft, [lon, lat] points, 24 h sample) and
reports how much air traffic passes within earshot of a given point.
More flyovers in earshot = more chances a recording there picks up
Doppler traces worth scrubbing.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

EARTH_R_KM = 6371.0


def _closest_km(lat0: float, lon0: float, coords) -> float:
    """Min distance from point to a polyline, local equirectangular projection."""
    pts = np.asarray(coords, float)
    if pts.ndim != 2 or len(pts) == 0:
        return math.inf
    kx = math.cos(math.radians(lat0)) * math.pi / 180 * EARTH_R_KM
    ky = math.pi / 180 * EARTH_R_KM
    xy = np.column_stack(((pts[:, 0] - lon0) * kx, (pts[:, 1] - lat0) * ky))
    if len(xy) == 1:
        return float(np.hypot(*xy[0]))
    a, b = xy[:-1], xy[1:]
    ab = b - a
    denom = np.maximum((ab ** 2).sum(1), 1e-12)
    u = np.clip(-(a * ab).sum(1) / denom, 0, 1)
    proj = a + u[:, None] * ab
    return float(np.hypot(proj[:, 0], proj[:, 1]).min())


def exposure(geojson: str | Path, lat: float, lon: float, radius_km: float = 5.0) -> dict:
    data = json.loads(Path(geojson).read_text())
    passes = []
    for feat in data.get("features", []):
        geom = feat.get("geometry") or {}
        parts = geom.get("coordinates", [])
        if geom.get("type") == "LineString":
            parts = [parts]
        d = min((_closest_km(lat, lon, p) for p in parts), default=math.inf)
        if d <= radius_km:
            props = feat.get("properties") or {}
            passes.append((d, props.get("type") or "unknown"))
    dists = sorted(d for d, _ in passes)
    within = lambda r: sum(1 for d in dists if d <= r)
    n = len(dists)
    level = "high" if n >= 48 else "moderate" if n >= 8 else "low"
    return {
        "location": [lat, lon], "radius_km": radius_km,
        "aircraft_per_day": n,
        "per_hour_avg": round(n / 24, 1),
        "within_1km": within(1.0), "within_2km": within(2.0),
        "median_closest_km": round(float(np.median(dists)), 2) if dists else None,
        "top_types": Counter(t for _, t in passes).most_common(8),
        "exposure": level,
        "note": ("24 h sample from flight-path-mapper (no timestamps/altitude in its GeoJSON), "
                 "so this is a daily volume estimate, not an hourly schedule."),
    }
