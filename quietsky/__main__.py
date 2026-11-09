"""quietsky CLI.

  quietsky scan  INPUT [--json]
  quietsky scrub INPUT OUTPUT [--hum] [--passes N] [--json]
  quietsky exposure --geojson CITY.geojson --lat LAT --lon LON [--radius-km 5]
  quietsky stream INPUT OUTPUT [--latency 30] [--aggressive]   (experimental live mode)
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


def _stream_cmd(a):
    import subprocess, tempfile
    from pathlib import Path
    import numpy as np
    from .core import load_audio, probe, write_wav
    from .stream import StreamConfig, StreamProcessor
    info = probe(a.input)
    x, sr = load_audio(a.input)
    cfg = StreamConfig(sr=sr, latency_s=a.latency, confirm=2 if a.aggressive else 3)
    y = np.empty_like(x)
    for ch in range(x.shape[0]):              # channels processed independently
        sp = StreamProcessor(cfg)
        blocks = [sp.process(x[ch, i:i + 1024]) for i in range(0, x.shape[1], 1024)]
        y[ch] = np.concatenate(blocks + [sp.flush()])[: x.shape[1]]
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "live.wav"
        write_wav(wav, y, sr)
        if info["has_video"]:
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", a.input, "-i", str(wav), "-map", "0:v",
                            "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
                            "-map_metadata", "-1", "-shortest", a.output], check=True)
        else:
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-map_metadata", "-1", a.output], check=True)
    r = detect(a.output)
    print(f"Wrote {a.output}  (live mode, {a.latency:.0f} s delay). Re-scan risk: {r.risk.upper()}")
    if a.latency < 20:
        print("  Note: under ~20 s delay, each flyover's approach can't be identified in time and leaks through.")
    return 0 if r.risk == "low" else 1


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
    st = sub.add_parser("stream", help="experimental: process a file block-by-block exactly as the live mode would")
    st.add_argument("input"); st.add_argument("output")
    st.add_argument("--latency", type=float, default=30.0,
                    help="output delay in seconds; ~30+ needed to cover a flyover's approach")
    st.add_argument("--aggressive", action="store_true",
                    help="cut after 2 agreeing detections instead of 3 (catches more, may alter clean audio)")
    e = sub.add_parser("exposure", help="air traffic near your own recording spot (flight-path-mapper GeoJSON)")
    e.add_argument("--geojson", required=True); e.add_argument("--lat", type=float, required=True)
    e.add_argument("--lon", type=float, required=True); e.add_argument("--radius-km", type=float, default=5.0)
    a = p.parse_args(argv)
    if a.cmd == "stream":
        return _stream_cmd(a)
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
