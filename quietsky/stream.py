"""quietsky streaming: causal, bounded-latency suppression of aircraft Doppler
traces, for live use (streamers, calls).

The batch scrubber in core.py sees the whole file. A live tool can't: at time t
it has only the past plus whatever fixed lookahead you allow. Streaming splits
latency in two:

  * throughput latency  — fixed input->output delay, set by `latency_s`. The
    audio stream flows through with this delay and no more.
  * detection latency   — how much of a NEW flyover must be heard before the
    detector is sure. The curve has to glide a while before a matched filter
    can fit it; a few to a few-tens of seconds.

Design: rather than hand-roll a weaker online detector, this runs the tested
core.detect() on a rolling buffer every `refresh_s`, and applies the resulting
fitted curves to the delayed output via overlap-add. So streaming detection is
exactly as sensitive as batch detection on the same span of audio.

What `latency_s` buys you:
  * latency_s small (~0.3 s): the stream is near-live, but a flyover is only
    suppressed from the moment the detector locks (seconds in), so its faint
    onset leaks. The loud, dangerous center is still removed.
  * latency_s large (a few s, within a streamer's usual stream delay): the
    output is held long enough that, by the time a frame is due, the detector
    has already seen enough of that flyover to suppress it from its onset.

Either way the whole-file guarantee of the batch scrubber is NOT reached live;
for archival or sensitive footage, scrub the file with core.scrub_file.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter

from .core import DetectConfig, detect


@dataclass
class StreamConfig:
    sr: int = 48000
    win: int = 8192                 # 170 ms at 48 kHz -> ~5.9 Hz bins
    hop: int = 2048                 # 75% overlap (Hann is COLA here)
    latency_s: float = 3.0          # throughput delay (output held this long)
    window_s: float = 48.0          # rolling buffer the detector sees
    refresh_s: float = 3.0          # how often the detector re-runs
    fmax_harm: float = 1200.0
    max_harm: int = 6
    atten_db: float = 40.0
    notch_bw_hz: float = 6.0
    notch_bw_rel: float = 0.04
    pad_s: float = 4.0              # extend each curve's active span
    smear_k: float = 0.75           # widen notch by this x the in-frame sweep
    confirm: int = 3                # refreshes whose fits must trace the same tone before it's cut.
                                    # 3 = conservative (left plane-free test audio bit-exact);
                                    # 2 = aggressive (catches more, sometimes alters clean audio)
    null_margin: float = 1.0        # per-window margin over the rising-pitch noise floor
    min_curve_hz: float = 60.0      # below this, background rumble and hum produce false glides
    min_buffer_s: float = 20.0      # short early buffers give unreliable fits; wait for this much
    refill_trim: float = 0.77       # match core: refill reads as flat on a dB spectrogram
    seed: int | None = 0


def _same_curve(b1, tr1, b2, tr2, min_overlap_s=3.0, tol=0.04):
    """Do two fits trace the same tone? Compare predicted frequency over the
    time both actually observed. Partial-curve fits wander in t0 and f_mid,
    but where both saw the tone they agree."""
    lo = max(b1 + tr1.t_start, b2 + tr2.t_start)
    hi = min(b1 + tr1.t_end, b2 + tr2.t_end)
    if hi - lo < min_overlap_s:
        return False
    ts = np.linspace(lo, hi, 12)
    f1 = tr1.freq_at(ts - b1)
    f2 = tr2.freq_at(ts - b2)
    return float(np.median(np.abs(f1 - f2) / f2)) < tol


class StreamProcessor:
    """Feed blocks to process(); get back delayed, suppressed audio. Causal:
    output sample n depends only on input samples <= n + latency_samples."""

    def __init__(self, cfg: StreamConfig | None = None, dcfg: DetectConfig | None = None):
        self.cfg = cfg or StreamConfig()
        self.dcfg = dcfg or DetectConfig(null_margin=self.cfg.null_margin)
        c = self.cfg
        self.awin = np.hanning(c.win)
        norm = np.zeros(c.win)
        w2 = self.awin ** 2
        for s in range(0, c.win, c.hop):
            norm[: c.win - s] += w2[s:]
            norm[c.win - s:] += w2[:s]
        self._synth_norm = np.median(norm[norm > 0])
        self.freqs = np.fft.rfftfreq(c.win, 1 / c.sr)
        self.lat_frames = max(1, round(c.latency_s * c.sr / c.hop))
        self._rng = np.random.default_rng(c.seed)

        self._skip = c.win - c.hop                 # silent prefix length (internal timeline)
        self._leftover = np.zeros(self._skip)
        self._n_in = 0
        self._n_out = 0
        self._k = 0
        self._pending: list[tuple[int, np.ndarray]] = []
        self._ola = np.zeros(0)
        self._ola_start = 0
        self._emit_upto = 0
        # detection
        self._buf = np.zeros(0)                 # rolling raw audio
        self._buf_start = 0                     # absolute sample index of _buf[0]
        self._since_detect = 0
        self._curves: list = []                 # confirmed: (abs_base_sec, TrackFit)
        self._cands: list = []                  # all: (abs_base_sec, TrackFit, hits)

    # ---- detection on the rolling buffer --------------------------------
    def _maybe_detect(self):
        c = self.cfg
        win_n = int(c.window_s * c.sr)
        if len(self._buf) > win_n:
            drop = len(self._buf) - win_n
            self._buf = self._buf[drop:]
            self._buf_start += drop
        if self._since_detect < c.refresh_s * c.sr or len(self._buf) < c.min_buffer_s * c.sr:
            return
        self._since_detect = 0
        rep = detect(self._buf, sr=c.sr, cfg=self.dcfg)
        base = self._buf_start / c.sr
        # Accumulate across refreshes. Replacing the list would let one refresh
        # that misses a flyover wipe out its suppression. A newer fit of the same
        # flyover (close in time and frequency) replaces the older one.
        # A curve is applied only after `confirm` separate refreshes agree on it:
        # the detector runs on many overlapping windows, so single-window flukes
        # are likely, while a real flyover reappears in window after window.
        for ev in rep.events:
            for tr in ev.tracks:
                if tr.f_mid < c.min_curve_hz:
                    continue
                t0_abs = base + tr.t0
                hits, keep = 1, []
                for (b, old, h) in self._cands:
                    if _same_curve(b, old, base, tr):
                        hits = max(hits, h + 1)          # same flyover: newer fit wins
                    else:
                        keep.append((b, old, h))
                self._cands = keep + [(base, tr, hits)]
        emit_t = (self._emit_upto - self._skip) / c.sr
        self._cands = [(b, tr, h) for (b, tr, h) in self._cands
                       if b + tr.t_end + self._pad(tr) >= emit_t]
        self._curves = [(b, tr) for (b, tr, h) in self._cands if h >= c.confirm]

    def _pad(self, tr):
        # the tanh fit extrapolates well; faint tails extend ~3 tau past the observed span
        return max(self.cfg.pad_s, 3.0 * tr.tau if tr.kind == "doppler" else 0.0)

    # ---- synthesis ------------------------------------------------------
    def _gain(self, k0):
        c = self.cfg
        t = (k0 * c.hop - self._skip) / c.sr
        g = np.ones(self.freqs.shape)
        floor_lin = 10 ** (-c.atten_db / 20)
        for base, tr in self._curves:
            lt = t - base
            pad = self._pad(tr)
            if not (tr.t_start - pad <= lt <= tr.t_end + pad):
                continue
            f0 = float(tr.freq_at(np.array([lt]))[0])
            # the tone sweeps |df/dt| * window-length within one analysis frame;
            # near closest approach that smear is wider than a fixed notch
            if tr.kind == "doppler":
                sech2 = 1.0 / np.cosh((lt - tr.t0) / tr.tau) ** 2
                slope = tr.half_shift / tr.tau * sech2          # Hz/s at the fundamental
            else:
                slope = 0.0
            smear = slope * c.win / c.sr                         # Hz swept per frame
            for m in range(1, c.max_harm + 1):
                fc = f0 * m
                if fc > c.fmax_harm:
                    break
                hw = c.notch_bw_hz + c.notch_bw_rel * fc + c.smear_k * smear * m
                d = np.abs(self.freqs - fc) / hw
                gg = np.where(d < 1, floor_lin, np.where(d < 2, floor_lin + (1 - floor_lin) * (d - 1), 1.0))
                g = np.minimum(g, gg)
        return g

    def _synth_frame(self, S, g):
        if np.all(g >= 1.0):
            return S
        floor = median_filter(np.abs(S), size=41, mode="nearest")
        phase = np.exp(2j * np.pi * self._rng.random(S.shape))
        refill = floor * self.cfg.refill_trim * np.sqrt(self.cfg.win / self.cfg.hop) / np.sqrt(2) * phase
        return S * g + (1 - g) * refill

    def _emit_ready(self, out):
        c = self.cfg
        while self._pending and self._pending[0][0] <= self._k - self.lat_frames:
            k0, S = self._pending.pop(0)
            y = np.fft.irfft(self._synth_frame(S, self._gain(k0)), n=c.win) * self.awin / self._synth_norm
            pos = k0 * c.hop
            end = pos + c.win
            if end - self._ola_start > len(self._ola):
                self._ola = np.pad(self._ola, (0, end - self._ola_start - len(self._ola)))
            a = pos - self._ola_start
            self._ola[a : a + c.win] += y
            final = (k0 + 1) * c.hop
            n = final - self._emit_upto
            if n > 0:
                b = self._emit_upto - self._ola_start
                out.append(self._ola[b : b + n].copy())
                self._emit_upto = final
        drop = self._emit_upto - self._ola_start
        if drop > 0:
            self._ola = self._ola[drop:]
            self._ola_start = self._emit_upto

    def process(self, block):
        c = self.cfg
        block = np.asarray(block, np.float64).ravel()
        self._n_in += len(block)
        self._buf = np.concatenate([self._buf, block])
        self._since_detect += len(block)
        self._maybe_detect()
        data = np.concatenate([self._leftover, block])
        out = []
        while len(data) >= c.win:
            frame = data[: c.win]
            self._pending.append((self._k, np.fft.rfft(frame * self.awin)))
            self._k += 1
            data = data[c.hop:]
            self._emit_ready(out)
        self._leftover = data
        return self._real(out)

    def _real(self, out):
        y = np.concatenate(out) if out else np.zeros(0, np.float64)
        drop = max(0, min(len(y), self._skip - (self._emit_upto - len(y))))
        y = y[drop:]
        self._n_out += len(y)
        return y

    def flush(self):
        c = self.cfg
        out = []
        # zero-pad so every remaining input sample is covered by full frame overlap
        tail = np.concatenate([self._leftover, np.zeros(c.win)])
        while len(tail) >= c.win:
            self._pending.append((self._k, np.fft.rfft(tail[: c.win] * self.awin)))
            self._k += 1
            tail = tail[c.hop:]
        self._leftover = np.zeros(0)
        self.lat_frames = 0
        self._emit_ready(out)
        prev = self._n_out
        y = self._real(out)
        need = self._n_in - prev                          # real samples still owed
        self._n_out = self._n_in
        return y[:need] if len(y) >= need else np.pad(y, (0, need - len(y)))

    @property
    def latency_samples(self):
        return self.lat_frames * self.cfg.hop + self.cfg.win
