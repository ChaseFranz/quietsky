"""quietsky CLI.

  quietsky scan  INPUT [--json]
  quietsky scrub INPUT OUTPUT [--hum] [--passes N] [--json]
  quietsky exposure --geojson CITY.geojson --lat LAT --lon LON [--radius-km 5]
"""
import argparse
import json
import sys

from .core import DetectConfig, ScrubConfig, detect, scrub_file
from .exposure import exposure


def _print_report(r, title):
    d = r.to_dict()
    badge = {"low": "LOW", "moderate": "MODERATE", "high": "HIGH"}[d["risk"]]
    print(f"\n{title}: {d['file']}  ({d['duration_s']} s)")
    print(f"  Aircraft-trace geolocation risk: {badge}")
    for i, e in enumerate(d["aircraft_events"], 1):
        print(f"  [{i}] flyover {e['t_start']:>7.1f}–{e['t_end']:<7.1f}s  closest≈{e['t_closest_approach']}s  "
              f"{e['freq_range_hz'][0]}–{e['freq_range_hz'][1]} Hz  harmonics={e['harmonics']}  "
              f"~{e['est_ground_speed_mps']} m/s  conf={e['confidence']}")
    if d["inter_event_gaps_s"]:
        print(f"  Gaps between flyovers (s): {d['inter_event_gaps_s']}  <- strongest triangulation clue")
    for s in d["stationary_tones"]:
        tag = f"  [{s['mains_hum']}: reveals grid region; ENF can reveal recording time]" if s["mains_hum"] else ""
        print(f"  steady tone {s['freq_hz']} Hz{tag}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="quietsky", description="Find and remove aircraft Doppler traces that can geolocate a recording.")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan"); s.add_argument("input"); s.add_argument("--json", action="store_true")
    s.add_argument("--margin", type=float, default=1.0, help="z-score margin over the rising-pitch null (lower = more sensitive)")
    c = sub.add_parser("scrub"); c.add_argument("input"); c.add_argument("output")
    c.add_argument("--hum", action="store_true", help="also remove steady tones incl. mains hum")
    c.add_argument("--passes", type=int, default=2, help="scrub rounds; each is followed by a verification scan, stops early when clean")
    c.add_argument("--margin", type=float, default=1.0)
    c.add_argument("--json", action="store_true")
    e = sub.add_parser("exposure", help="air traffic near your own recording spot (flight-path-mapper GeoJSON)")
    e.add_argument("--geojson", required=True); e.add_argument("--lat", type=float, required=True)
    e.add_argument("--lon", type=float, required=True); e.add_argument("--radius-km", type=float, default=5.0)
    a = p.parse_args(argv)
    if a.cmd == "exposure":
        print(json.dumps(exposure(a.geojson, a.lat, a.lon, a.radius_km), indent=2))
        return 0
    dcfg = DetectConfig(null_margin=a.margin)

    if a.cmd == "scan":
        r = detect(a.input, cfg=dcfg)
        print(json.dumps(r.to_dict(), indent=2)) if a.json else _print_report(r, "Scan")
        return 2 if r.risk == "high" else 1 if r.risk == "moderate" else 0

    before, after = scrub_file(a.input, a.output, dcfg=dcfg, scfg=ScrubConfig(remove_hum=a.hum), passes=a.passes)
    if a.json:
        print(json.dumps({"before": before.to_dict(), "after": after.to_dict(), "output": a.output}, indent=2))
    else:
        _print_report(before, "Before"); _print_report(after, "After ")
        print(f"\nWrote {a.output}")
        if after.events:
            print("  Residual traces remain. For sensitive footage, replace the audio track entirely.")
    return 0 if not after.events else 1


if __name__ == "__main__":
    sys.exit(main())
