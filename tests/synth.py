"""Synthetic recordings: physically modeled propeller flyovers buried in noise."""
import numpy as np
from scipy import signal

C = 343.0


def flyover(t, t_cpa, v, d, f_blade, harmonics, amp, sr):
    """Straight-line pass. Received frequency includes sound propagation delay."""
    # emission times te such that te + r(te)/c = t  (solve by fixed-point iteration)
    te = t.copy()
    for _ in range(8):
        r = np.sqrt(d ** 2 + (v * (te - t_cpa)) ** 2)
        te = t - r / C
    r = np.sqrt(d ** 2 + (v * (te - t_cpa)) ** 2)
    v_radial = v ** 2 * (te - t_cpa) / r            # >0 receding
    f_ratio = 1.0 / (1.0 + v_radial / C)
    env = (d / r) ** 1.0
    y = np.zeros_like(t)
    for n, a in harmonics:
        f_inst = f_blade * n * f_ratio
        y += a * np.sin(2 * np.pi * np.cumsum(f_inst) / sr)
    return amp * env * y


def make_scene(duration=150.0, sr=48000, seed=0, planes=True, hum=True, tone_db=-32, fast=False):
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * sr)) / sr
    # brown-ish ambient noise (wind/river): white noise, 1-pole lowpass + some white
    w = rng.standard_normal(t.size)
    b, a = signal.butter(1, 300 / (sr / 2))
    noise = signal.lfilter(b, a, w) * 3 + 0.3 * w
    noise /= np.sqrt(np.mean(noise ** 2))
    noise *= 0.1
    amp = 0.1 * 10 ** (tone_db / 20)
    y = noise.copy()
    if planes:
        harm = [(1, 1.0), (2, 0.6), (3, 0.4), (4, 0.25)]
        y += flyover(t, 40.0, 70.0, 700.0, 92.0, harm, amp, sr)
        if fast:   # ~100 m/s, like the fit in Colsto's example
            y += flyover(t, 105.0, 100.0, 1500.0, 130.0, harm, amp, sr)
        else:
            y += flyover(t, 105.0, 55.0, 1100.0, 118.0, harm, amp, sr)
    if hum:
        y += amp * 0.8 * np.sin(2 * np.pi * 60.0 * t) + amp * 0.4 * np.sin(2 * np.pi * 180.0 * t)
    return y.astype(np.float32), sr
