import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from synth import make_scene  # noqa: E402

from quietsky import ScrubConfig, detect, scrub_array  # noqa: E402
from quietsky.core import write_wav  # noqa: E402
from quietsky.exposure import exposure  # noqa: E402

SEEDS = [1, 2, 3]


@pytest.mark.parametrize("seed", SEEDS)
def test_no_false_positives_on_noise_and_hum(seed):
    y, sr = make_scene(planes=False, hum=True, seed=seed, tone_db=-14)
    r = detect(y, sr=sr)
    assert r.risk == "low"
    assert any(s["mains_hum"] for s in r.to_dict()["stationary_tones"])


@pytest.mark.parametrize("seed", SEEDS)
def test_detects_two_flyovers(seed):
    y, sr = make_scene(seed=seed, tone_db=-14)
    r = detect(y, sr=sr)
    assert r.risk in ("moderate", "high")
    # truth: closest approach heard at ~42 s and ~108 s (incl. propagation delay)
    hits = [e.t_closest for e in r.events]
    assert any(abs(h - 105) < 15 for h in hits)


def test_scrub_removes_traces():
    y, sr = make_scene(seed=2, tone_db=-14)
    before = detect(y, sr=sr)
    assert before.events
    clean, rep = y, before
    for p in range(2):                       # mirrors scrub_file's default 2 passes
        if not rep.events:
            break
        clean = scrub_array(clean, sr, rep, ScrubConfig(seed=p))
        rep = detect(clean[0], sr=sr)
    after = rep
    assert after.risk == "low"
    # scrub must not wreck the rest of the audio
    rms = lambda a: float(np.sqrt(np.mean(np.square(a))))
    # dB-matched refill trades some background energy for an invisible scrub (see README Limits)
    assert abs(20 * np.log10(rms(clean) / rms(y))) < 4.0


def test_file_roundtrip(tmp_path):
    from quietsky.core import scrub_file
    y, sr = make_scene(seed=3, tone_db=-14, duration=120)
    src, dst = tmp_path / "in.wav", tmp_path / "out.wav"
    write_wav(src, y, sr)
    before, after = scrub_file(src, dst, scfg=ScrubConfig(seed=0))
    assert dst.exists() and before.events and not after.events


def test_exposure(tmp_path):
    lat, lon = 44.85, -93.47
    feats = [
        {"type": "Feature", "properties": {"hex": "a", "type": "C172"},
         "geometry": {"type": "MultiLineString", "coordinates": [[[lon - 0.1, lat + 0.005], [lon + 0.1, lat + 0.005]]]}},
        {"type": "Feature", "properties": {"hex": "b", "type": "B738"},
         "geometry": {"type": "MultiLineString", "coordinates": [[[lon - 0.1, lat + 0.5], [lon + 0.1, lat + 0.5]]]}},
    ]
    p = tmp_path / "paths.geojson"
    p.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    e = exposure(p, lat, lon, radius_km=5)
    assert e["aircraft_per_day"] == 1 and e["within_1km"] == 1
    assert e["top_types"][0][0] == "C172"


# ---------------------------------------------------------------- live mode

from quietsky.stream import StreamConfig, StreamProcessor  # noqa: E402


def _run_stream(x, cfg, block=1024):
    sp = StreamProcessor(cfg)
    return np.concatenate([sp.process(x[i:i + block]) for i in range(0, len(x), block)] + [sp.flush()])


def _curve_energy(x, sr, flyover):
    from scipy import signal as sig
    from quietsky.core import DetectConfig, _whiten, spectrogram
    dc = DetectConfig()
    xx = sig.resample_poly(x, dc.analysis_sr, sr)
    f, t, P = spectrogram(xx, dc.analysis_sr, dc)
    W = _whiten(P, 6)
    tc, v, d, fb = flyover
    vals = []
    for n in (1, 2, 3, 4):
        for j, tj in enumerate(t):
            te = tj
            for _ in range(8):
                te = tj - np.hypot(d, v * (te - tc)) / 343
            r = np.hypot(d, v * (te - tc))
            fr = fb * n / (1 + v * v * (te - tc) / r / 343)
            if abs(te - tc) < 25 and fr < 1500:
                i = np.argmin(abs(f - fr))
                vals.append(W[max(i - 1, 0):i + 2, j].max())
    return float(np.mean(vals))


@pytest.mark.parametrize("n,block", [(48000 * 3, 1000), (48000 * 3 + 777, 4096), (50000, 333)])
def test_stream_passthrough_is_exact(n, block):
    x = np.random.default_rng(0).standard_normal(n) * 0.1
    cfg = StreamConfig(latency_s=0.5)
    y = _run_stream(x, cfg, block)
    assert len(y) == n
    assert np.sqrt(np.mean((y - x) ** 2)) / 0.1 < 1e-4      # whole array, edges included


def test_stream_leaves_plane_free_audio_untouched():
    y, sr = make_scene(planes=False, hum=True, seed=7, tone_db=-14)
    out = _run_stream(y, StreamConfig(sr=sr, latency_s=3.0))
    assert np.sqrt(np.mean((out - y) ** 2)) / np.sqrt(np.mean(y ** 2)) < 1e-4


def test_stream_with_long_delay_suppresses_loud_flyover():
    y, sr = make_scene(seed=2, tone_db=-14)
    out = _run_stream(y, StreamConfig(sr=sr, latency_s=35.0), block=4096)
    loud = (105, 55, 1100, 118)
    assert _curve_energy(y, sr, loud) > 4.5          # clearly present before
    assert _curve_energy(out, sr, loud) < 2.5        # near the noise floor after
