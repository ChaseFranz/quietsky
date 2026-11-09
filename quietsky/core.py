"""quietsky core: detect and scrub aircraft Doppler traces from audio.

Threat model (Colsto, Nov 2026): faint propeller/engine tones in a recording
trace Doppler pitch curves. Curve shape + timing between flyovers, matched
against historical ADS-B data, can geolocate the recording to ~100 m.
quietsky finds those curves before you publish, and removes them.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve
from scipy.ndimage import maximum_filter

C_SOUND = 343.0

# ---------------------------------------------------------------- audio io


def _require_ffmpeg() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg/ffprobe not found on PATH")


def probe(path: str | Path) -> dict:
    _require_ffmpeg()
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    streams = json.loads(out)["streams"]
    audio = next((s for s in streams if s["codec_type"] == "audio"), None)
    if audio is None:
        raise ValueError(f"{path}: no audio stream")
    return {
        "sr": int(audio["sample_rate"]),
        "channels": int(audio["channels"]),
        "has_video": any(s["codec_type"] == "video" for s in streams),
    }


def load_audio(path: str | Path, sr: int | None = None, mono: bool = False) -> tuple[np.ndarray, int]:
    """Decode any ffmpeg-readable file to float32 array of shape (channels, n)."""
    info = probe(path)
    sr = sr or info["sr"]
    ch = 1 if mono else info["channels"]
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", str(ch),
         "-ar", str(sr), "-f", "f32le", "-"],
        check=True, capture_output=True,
    ).stdout
    x = np.frombuffer(raw, dtype="<f4").reshape(-1, ch).T.copy()
    return x, sr


def write_wav(path: str | Path, x: np.ndarray, sr: int) -> None:
    _require_ffmpeg()
    x = np.atleast_2d(x).astype("<f4")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(sr),
         "-ac", str(x.shape[0]), "-i", "-", "-c:a", "pcm_s24le", str(path)],
        input=x.T.tobytes(), check=True,
    )


# ---------------------------------------------------------------- models


def doppler_tanh(t, f_mid, half_shift, t0, tau):
    """Smooth descending S-curve; good proxy for a straight-line flyover."""
    return f_mid - half_shift * np.tanh((t - t0) / tau)


@dataclass
class Track:
    times: list[float] = field(default_factory=list)
    freqs: list[float] = field(default_factory=list)
    snrs: list[float] = field(default_factory=list)
    gap: int = 0

    @property
    def duration(self) -> float:
        return self.times[-1] - self.times[0] if self.times else 0.0


@dataclass
class TrackFit:
    t_start: float
    t_end: float
    f_mid: float
    half_shift: float
    t0: float
    tau: float
    r2: float
    mean_snr_db: float
    kind: str  # "doppler" | "stationary"

    @property
    def rel_shift(self) -> float:
        return 2 * self.half_shift / self.f_mid

    def freq_at(self, t):
        if self.kind == "stationary":
            return np.full_like(np.asarray(t, float), self.f_mid)
        return doppler_tanh(np.asarray(t, float), self.f_mid, self.half_shift, self.t0, self.tau)


@dataclass
class Event:
    """One aircraft flyover: a fundamental and its harmonics."""
    tracks: list[TrackFit]

    @property
    def t_closest(self) -> float:
        return float(np.median([t.t0 for t in self.tracks]))

    @property
    def t_start(self) -> float:
        return min(t.t_start for t in self.tracks)

    @property
    def t_end(self) -> float:
        return max(t.t_end for t in self.tracks)

    @property
    def confidence(self) -> float:
        best_r2 = max(t.r2 for t in self.tracks)
        harm_bonus = min(len(self.tracks) - 1, 3) * 0.08
        return float(min(0.99, 0.55 * best_r2 + harm_bonus + 0.2))

    @property
    def est_speed_mps(self) -> float:
        # Far-field Doppler ratio f_approach/f_recede = (c+v)/(c-v)
        t = max(self.tracks, key=lambda k: k.r2)
        hi, lo = t.f_mid + t.half_shift, t.f_mid - t.half_shift
        return float(C_SOUND * (hi - lo) / (hi + lo))


@dataclass
class Report:
    file: str
    duration_s: float
    events: list[Event]
    stationary: list[TrackFit]

    @property
    def risk(self) -> str:
        n = len(self.events)
        return "high" if n >= 2 else "moderate" if n == 1 else "low"

    def to_dict(self) -> dict:
        def ev(e: Event):
            return {
                "t_start": round(e.t_start, 2), "t_end": round(e.t_end, 2),
                "t_closest_approach": round(e.t_closest, 2),
                "harmonics": len(e.tracks),
                "freq_range_hz": [round(min(t.f_mid - t.half_shift for t in e.tracks), 1),
                                  round(max(t.f_mid + t.half_shift for t in e.tracks), 1)],
                "est_ground_speed_mps": round(e.est_speed_mps, 1),
                "confidence": round(e.confidence, 2),
            }
        def st(t: TrackFit):
            mains = _mains_label(t.f_mid)
            return {"freq_hz": round(t.f_mid, 1), "t_start": round(t.t_start, 1),
                    "t_end": round(t.t_end, 1), "mains_hum": mains}
        gaps = [round(b.t_closest - a.t_closest, 1) for a, b in zip(self.events, self.events[1:])]
        return {
            "file": self.file, "duration_s": round(self.duration_s, 1), "risk": self.risk,
            "aircraft_events": [ev(e) for e in self.events],
            "inter_event_gaps_s": gaps,
            "stationary_tones": [st(t) for t in self.stationary],
        }


def _mains_label(f: float, tol: float = 1.2) -> str | None:
    for base in (50.0, 60.0):
        n = round(f / base)
        if 1 <= n <= 8 and abs(f - n * base) <= tol:
            return f"{int(base)} Hz grid, harmonic {n}"
    return None


# ---------------------------------------------------------------- detection


@dataclass
class DetectConfig:
    analysis_sr: int = 4000
    nperseg: int = 4096          # ~1 Hz resolution at 4 kHz
    hop: int = 1000              # 0.25 s
    fmin: float = 30.0
    fmax: float = 1500.0
    peak_db: float = 4.5         # above local spectral floor, after integration
    integrate_frames: int = 6    # ~1.5 s power averaging; tones drift slowly
    max_gap_frames: int = 4
    min_density: float = 0.6     # fraction of frames in a track with a hit
    min_track_s: float = 8.0
    min_r2: float = 0.75
    min_rel_shift: float = 0.17         # ~30 m/s; slower "flyovers" are almost always noise
    max_rel_shift: float = 0.75
    stationary_std_hz: float = 0.6
    # matched-filter search
    log_step: float = 0.002                       # 0.2% per log-frequency bin
    shifts: tuple = (0.12, 0.17, 0.23, 0.3, 0.4, 0.52, 0.64)  # total rel. shift ~ 2v/c
    taus: tuple = (2.0, 4.0, 7.0, 12.0, 20.0)     # seconds; ~ altitude / speed
    z_thresh: float = 7.0
    null_margin: float = 1.0                      # z above best rising-pitch (impossible) match


def spectrogram(x: np.ndarray, sr: int, cfg: DetectConfig):
    f, t, Z = signal.stft(x, fs=sr, window="hann", nperseg=cfg.nperseg,
                          noverlap=cfg.nperseg - cfg.hop, boundary=None, padded=False)
    band = (f >= cfg.fmin) & (f <= cfg.fmax)
    P = 10 * np.log10(np.abs(Z[band]) ** 2 + 1e-20)
    return f[band], t, P


def _whiten(P: np.ndarray, integrate: int = 1) -> np.ndarray:
    if integrate > 1:
        L = uniform_filter1d(10 ** (P / 10), integrate, axis=1, mode="nearest")
        P = 10 * np.log10(L)
    floor = median_filter(P, size=(41, 1), mode="nearest")
    return P - floor


def _track_peaks(f, t, W, cfg: DetectConfig) -> list[Track]:
    df = f[1] - f[0]
    active: list[Track] = []
    done: list[Track] = []
    for j, tj in enumerate(t):
        col = W[:, j]
        idx, props = signal.find_peaks(col, height=cfg.peak_db, distance=3)
        # parabolic interpolation for sub-bin frequency
        peaks = []
        for i in idx:
            if 0 < i < len(col) - 1:
                a, b, c = col[i - 1], col[i], col[i + 1]
                denom = a - 2 * b + c
                off = 0.5 * (a - c) / denom if denom != 0 else 0.0
            else:
                off = 0.0
            peaks.append((f[i] + off * df, col[i]))
        used = set()
        for tr in sorted(active, key=lambda k: -len(k.freqs)):
            if len(tr.freqs) >= 3:
                slope = (tr.freqs[-1] - tr.freqs[-3]) / 2
            else:
                slope = 0.0
            pred = tr.freqs[-1] + slope * (tr.gap + 1)
            tol = max(1.5 * df, 0.004 * pred) * (1 + 0.3 * tr.gap)
            best, bestd = None, tol
            for k, (pf, _) in enumerate(peaks):
                if k in used:
                    continue
                d = abs(pf - pred)
                if d <= bestd:
                    best, bestd = k, d
            if best is None:
                tr.gap += 1
            else:
                used.add(best)
                tr.times.append(float(tj)); tr.freqs.append(peaks[best][0])
                tr.snrs.append(float(peaks[best][1])); tr.gap = 0
        still = []
        for tr in active:
            (done if tr.gap > cfg.max_gap_frames else still).append(tr)
        active = still
        for k, (pf, s) in enumerate(peaks):
            if k not in used:
                active.append(Track([float(tj)], [pf], [float(s)]))
    done.extend(active)
    hop_s = t[1] - t[0] if len(t) > 1 else 1.0
    keep = []
    for tr in done:
        if tr.duration < cfg.min_track_s:
            continue
        if len(tr.times) / (tr.duration / hop_s + 1) < cfg.min_density:
            continue
        keep.append(tr)
    return keep


def _fit_track(tr: Track, cfg: DetectConfig) -> TrackFit | None:
    t = np.array(tr.times); fr = np.array(tr.freqs)
    snr = float(np.mean(tr.snrs))
    if np.std(fr) < cfg.stationary_std_hz:
        return TrackFit(t[0], t[-1], float(np.mean(fr)), 0.0, float(t.mean()), 1.0,
                        1.0, snr, "stationary")
    span = fr.max() - fr.min()
    p0 = [fr.mean(), span / 2, t[np.argmin(np.abs(fr - fr.mean()))], max(2.0, tr.duration / 6)]
    lo = [fr.min(), 0.0, t[0] - tr.duration, 0.3]
    hi = [fr.max(), fr.max(), t[-1] + tr.duration, 10 * tr.duration]
    try:
        popt, _ = curve_fit(doppler_tanh, t, fr, p0=p0, bounds=(lo, hi), maxfev=4000)
    except (RuntimeError, ValueError):
        return None
    resid = fr - doppler_tanh(t, *popt)
    ss_tot = np.sum((fr - fr.mean()) ** 2)
    r2 = 1 - np.sum(resid ** 2) / ss_tot if ss_tot > 0 else 0.0
    fit = TrackFit(t[0], t[-1], *map(float, popt), float(r2), snr, "doppler")
    observed_shift = (fr[: max(3, len(fr) // 10)].mean() - fr[-max(3, len(fr) // 10):].mean()) / fit.f_mid
    if (r2 < cfg.min_r2 or not (cfg.min_rel_shift <= fit.rel_shift <= cfg.max_rel_shift)
            or observed_shift < cfg.min_rel_shift / 2):
        return None
    return fit


def _group_events(fits: list[TrackFit]) -> list[Event]:
    """Merge fragments/harmonics of the same flyover: overlapping time spans
    with nearby closest-approach times."""
    groups: list[list[TrackFit]] = []
    for ft in sorted(fits, key=lambda k: k.t0):
        for g in groups:
            a0, a1 = min(k.t_start for k in g), max(k.t_end for k in g)
            ov = min(a1, ft.t_end) - max(a0, ft.t_start)
            shorter = min(a1 - a0, ft.t_end - ft.t_start)
            tc = float(np.median([k.t0 for k in g]))
            if ov > 0.5 * shorter and abs(ft.t0 - tc) < 15.0:
                g.append(ft)
                break
        else:
            groups.append([ft])
    return [Event(sorted(g, key=lambda k: k.f_mid)) for g in groups]


def _merge_stationary(st: list[TrackFit]) -> list[TrackFit]:
    out: list[TrackFit] = []
    for ft in sorted(st, key=lambda k: k.f_mid):
        if out and abs(ft.f_mid - out[-1].f_mid) < 1.5:
            o = out[-1]
            o.t_start, o.t_end = min(o.t_start, ft.t_start), max(o.t_end, ft.t_end)
        else:
            out.append(ft)
    return out

def _to_logfreq(f, W, cfg: DetectConfig):
    lf = np.arange(np.log(cfg.fmin), np.log(cfg.fmax), cfg.log_step)
    Wl = np.empty((lf.size, W.shape[1]), np.float32)
    ff = np.exp(lf)
    for j in range(W.shape[1]):
        Wl[:, j] = np.interp(ff, f, W[:, j])
    return lf, Wl


def _template(shift, tau, hop_s, cfg: DetectConfig, max_frames, guard=0.06):
    """Contrast template on log-frequency grid: +1 on the Doppler curve,
    -0.5 on parallel guard curves at +/- `guard` (log units)."""
    half = min(int(2.5 * tau / hop_s), max_frames // 2)
    tt = np.arange(-half, half + 1) * hop_s
    curve = np.log1p(-(shift / 2) * np.tanh(tt / tau))
    g = int(round(guard / cfg.log_step))
    rows = np.round(curve / cfg.log_step).astype(int)
    r0 = rows.min() - g
    T = np.zeros((rows.max() + g - r0 + 1, tt.size), np.float32)
    cols = np.arange(tt.size)
    T[rows - r0, cols] += 1.0
    T[rows - g - r0, cols] -= 0.5
    T[rows + g - r0, cols] -= 0.5
    T /= tt.size
    return T, -r0, half


def _corr(Wl, T, crow):
    """Correlation map indexed so M[r, j] = score for f_mid at row r, t0 at frame j."""
    M = fftconvolve(Wl, T[::-1, ::-1], mode="same")
    return np.roll(M, crow - T.shape[0] // 2, axis=0)


def _rowz(M, half):
    Z = np.full_like(M, -np.inf)
    valid = M[:, half: M.shape[1] - half]
    if valid.shape[1] < 5:
        return Z
    med = np.median(valid, axis=1, keepdims=True)
    mad = 1.4826 * np.median(np.abs(valid - med), axis=1, keepdims=True)
    mad = np.maximum(mad, np.median(mad) * 0.5) + 1e-9
    Z[:, half: M.shape[1] - half] = (valid - med) / mad
    return Z


def _matched_candidates(f, t, W, cfg: DetectConfig):
    lf, Wl = _to_logfreq(f, W, cfg)
    Wl = maximum_filter(Wl, size=(3, 1))   # tolerate tone width / curve mismatch
    Wl = np.clip(Wl, -6, 12)
    hop_s = t[1] - t[0]
    h_off = [int(round(np.log(n) / cfg.log_step)) for n in (2, 3, 4)]
    def zmap(T, crow, half):
        M = _corr(Wl, T, crow)
        # harmonic stack: fundamental + 2f + 3f + 4f share the same log-shape
        H = M.copy()
        for o in h_off:
            Hs = np.zeros_like(M); Hs[: M.shape[0] - o] = M[o:]
            H += Hs
        return np.maximum(_rowz(M, half), _rowz(H, half))

    cands, null_max = [], 0.0
    for sh in cfg.shifts:
        for tau in cfg.taus:
            T, crow, half = _template(sh, tau, hop_s, cfg, W.shape[1])
            if 2 * half * hop_s < cfg.min_track_s:
                continue
            # Null: rising-pitch templates. Real flyovers only fall in pitch, so
            # the best rising match measures this file's false-alarm level.
            null_max = max(null_max, float(np.max(zmap(T[:, ::-1], crow, half))))
            Z = zmap(T, crow, half)
            pk = (Z == maximum_filter(Z, size=(15, 9))) & (Z > cfg.z_thresh)
            for i, j in zip(*np.nonzero(pk)):
                cands.append((float(Z[i, j]), float(np.exp(lf[i])), sh, tau, float(t[j])))
    thresh = max(cfg.z_thresh, null_max + cfg.null_margin)
    cands = [c for c in cands if c[0] > thresh]
    cands.sort(reverse=True)
    kept = []
    for z, fm, sh, tau, t0 in cands:
        if any(abs(np.log(fm / k[1])) < 0.04 and abs(t0 - k[4]) < max(3.0, 0.5 * k[3]) for k in kept):
            continue
        kept.append((z, fm, sh, tau, t0))
    return kept


def _ridge_track(f, t, W, fm, sh, tau, t0, cfg: DetectConfig) -> Track:
    tr = Track()
    df = f[1] - f[0]
    for j, tj in enumerate(t):
        if abs(tj - t0) > 2.5 * tau:
            continue
        pred = fm * (1 - (sh / 2) * np.tanh((tj - t0) / tau))
        tol = max(2 * df, 0.01 * pred)
        m = np.abs(f - pred) <= tol
        if not m.any():
            continue
        i = np.flatnonzero(m)[np.argmax(W[m, j])]
        if W[i, j] > 1.0:
            tr.times.append(float(tj)); tr.freqs.append(float(f[i])); tr.snrs.append(float(W[i, j]))
    return tr


def detect(path_or_array, sr: int | None = None, cfg: DetectConfig | None = None,
           name: str = "<array>") -> Report:
    cfg = cfg or DetectConfig()
    if isinstance(path_or_array, (str, Path)):
        x, sr = load_audio(path_or_array, sr=cfg.analysis_sr, mono=True)
        x = x[0]; name = str(path_or_array)
    else:
        x = np.asarray(path_or_array, float)
        if x.ndim > 1:
            x = x.mean(axis=0)
        if sr != cfg.analysis_sr:
            x = signal.resample_poly(x, cfg.analysis_sr, sr)
        sr = cfg.analysis_sr
    f, t, P = spectrogram(x, sr, cfg)
    W = _whiten(P, cfg.integrate_frames)
    # steady tones (hum, appliances) via frame tracking
    stat = [ft for ft in (_fit_track(tr, cfg) for tr in _track_peaks(f, t, W, cfg))
            if ft is not None and ft.kind == "stationary"]
    # flyovers via matched filter, confirmed by ridge extraction + curve fit
    dop = []
    for z, fm, sh, tau, t0 in _matched_candidates(f, t, W, cfg):
        tr = _ridge_track(f, t, W, fm, sh, tau, t0, cfg)
        if tr.duration < cfg.min_track_s:
            continue
        ft = _fit_track(tr, cfg)
        if ft is not None and ft.kind == "doppler":
            ft.mean_snr_db = z   # store detection z-score
            dop.append(ft)
    return Report(name, len(x) / sr, _group_events(dop), _merge_stationary(stat))


# ---------------------------------------------------------------- scrubbing


@dataclass
class ScrubConfig:
    pad_s: float = 6.0            # extend masks past detected ends (faint tails)
    bw_hz: float = 4.0            # half-width of notch
    bw_rel: float = 0.04          # plus relative half-width (covers fit error at curve tails)
    atten_db: float = 45.0
    fmax_harmonics: float = 1200.0   # above this, %-width masks merge and erase the voice band
    max_harmonic: int = 6
    remove_hum: bool = False
    nperseg_s: float = 0.5
    seed: int | None = None
    refill_trim: float = 0.77   # calibrated so a scrubbed curve reads as flat background on a dB spectrogram (untrimmed: faint bright ridge)


def _mask_curves(event: Event, cfg: ScrubConfig):
    """Curves to remove: every detected track, plus integer multiples of every
    track >= 60 Hz (catches harmonics below the detection threshold without
    depending on which fragment was detected most strongly). Near-duplicate
    curves are dropped."""
    curves = [(t, 1.0) for t in event.tracks]
    seen = [t.f_mid for t in event.tracks]
    for base in sorted(event.tracks, key=lambda k: -k.mean_snr_db):
        if base.f_mid < 60.0:
            continue
        n = 2
        while base.f_mid * n <= cfg.fmax_harmonics and n <= cfg.max_harmonic:
            fm = base.f_mid * n
            if all(abs(fm - s) > 0.02 * fm for s in seen):
                curves.append((base, float(n)))
                seen.append(fm)
            n += 1
    return curves


def scrub_array(x: np.ndarray, sr: int, report: Report, cfg: ScrubConfig | None = None) -> np.ndarray:
    cfg = cfg or ScrubConfig()
    x = np.atleast_2d(np.asarray(x, np.float64))
    nper = int(2 ** np.ceil(np.log2(cfg.nperseg_s * sr)))
    hop = nper // 4
    out = np.empty_like(x)
    gain_floor = 10 ** (-cfg.atten_db / 20)
    for c in range(x.shape[0]):
        f, t, Z = signal.stft(x[c], fs=sr, window="hann", nperseg=nper, noverlap=nper - hop)
        G = np.ones(Z.shape)
        targets: list[tuple[TrackFit, float, float, float]] = []
        for ev in report.events:
            tau = max(t.tau for t in ev.tracks)
            pad = max(cfg.pad_s, 3.0 * tau)          # faint tails extend well past detection
            for fit, mult in _mask_curves(ev, cfg):
                targets.append((fit, mult, ev.t_start - pad, ev.t_end + pad))
        if cfg.remove_hum:
            for st in report.stationary:
                targets.append((st, 1.0, st.t_start - cfg.pad_s, st.t_end + cfg.pad_s))
        for fit, mult, ta, tb in targets:
            cols = np.where((t >= ta) & (t <= tb))[0]
            if cols.size == 0:
                continue
            fc = fit.freq_at(t[cols]) * mult
            hw = cfg.bw_hz + cfg.bw_rel * fc
            d = np.abs(f[:, None] - fc[None, :]) / hw[None, :]
            # raised-cosine edge over d in [1, 3]: a hard edge is itself a Doppler-shaped feature
            ramp = 0.5 - 0.5 * np.cos(np.pi * np.clip((d - 1) / 2, 0, 1))
            g = np.where(d < 1, gain_floor, gain_floor + (1 - gain_floor) * ramp)
            G[:, cols] = np.minimum(G[:, cols], g)
        # Refill removed bins with noise at the local floor level. A bare notch
        # leaves a dark Doppler-shaped groove that is itself a fingerprint.
        floor = median_filter(np.abs(Z), size=(61, 1), mode="nearest")
        rng = np.random.default_rng(None if cfg.seed is None else cfg.seed + c)
        phase = np.exp(2j * np.pi * rng.random(Z.shape))
        # Rayleigh-distributed magnitude with median == local floor
        # x sqrt(nper/hop): random-phase frames add incoherently in overlap-add,
        # losing that factor in power vs. a consistent signal.
        mag = floor * rng.rayleigh(1.0, Z.shape) / np.sqrt(2 * np.log(2)) * np.sqrt(nper / hop) * cfg.refill_trim
        Zc = Z * G + (1 - G) * mag * phase
        _, y = signal.istft(Zc, fs=sr, window="hann", nperseg=nper, noverlap=nper - hop)
        y = y[: x.shape[1]]
        if y.size < x.shape[1]:
            y = np.pad(y, (0, x.shape[1] - y.size))
        out[c] = y
    return out.astype(np.float32)


def scrub_file(src: str | Path, dst: str | Path, report: Report | None = None,
               dcfg: DetectConfig | None = None, scfg: ScrubConfig | None = None,
               passes: int = 2) -> tuple[Report, Report]:
    """Detect, scrub, and re-detect (up to `passes` rounds). Remuxes video untouched."""
    src, dst = Path(src), Path(dst)
    info = probe(src)
    x, sr = load_audio(src)
    first = report or detect(src, cfg=dcfg)
    cur, y = first, x
    for _ in range(passes):
        if not cur.events and not (scfg and scfg.remove_hum and cur.stationary):
            break
        y = scrub_array(y, sr, cur, scfg)
        cur = detect(y, sr=sr, cfg=dcfg, name=str(dst))
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "clean.wav"
        write_wav(wav, y, sr)
        if info["has_video"]:
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-i", str(wav),
                 "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
                 "-map_metadata", "-1", "-shortest", str(dst)], check=True)
        else:
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav),
                            "-map_metadata", "-1", str(dst)], check=True)
    return first, cur
