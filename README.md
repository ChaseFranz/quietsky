# quietsky

Find and remove the faint aircraft sounds that can reveal where a recording was made, before you publish it.

## What this is, in plain English

Almost every outdoor recording picks up aircraft passing miles away, even when nobody can hear them. In November 2026, the creator Colsto [showed](https://www.youtube.com/watch?v=UytYrhYLOKI) that those sounds, compared with public flight-tracking data, can locate where a recording was made, sometimes to within about 100 meters, with nothing identifiable on screen.

quietsky checks your audio and video for those traces and removes them, so a stream, a source's footage or a home video doesn't give away a location. It's meant for streamers, journalists and their sources, activists, people in sensitive situations, and anyone posting from a fixed camera.

A few things that **don't** protect you, based on his demonstration: cutting the low frequencies (the traces sit in the same range as voice), talking or playing music over them, and phone-call compression. Cutting the recording into pieces does help, because it breaks up the timing between planes.

quietsky is defensive only. It never works out where a recording was made.

Right now it runs from the command line, so the rest of this page assumes some comfort with a terminal. A simpler app is on the roadmap.

## Install

```bash
pip install numpy scipy        # plus ffmpeg/ffprobe on PATH
pip install -e .
```

## Use

```bash
quietsky scan  clip.mp4                  # risk report (exit code 0 low / 1 moderate / 2 high)
quietsky scan  clip.mp4 --json
quietsky scrub clip.mp4 clip_clean.mp4   # removes traces, re-scans the output, copies video untouched, strips metadata
quietsky scrub clip.mp4 out.mp4 --hum    # also remove steady tones incl. 50/60 Hz mains hum
quietsky exposure --geojson minneapolis-80mi.geojson --lat 44.85 --lon -93.47 --radius-km 5
quietsky stream clip.mp4 clip_live.mp4 --latency 30   # experimental: what live mode would output
```

Sample scan output:

```
Aircraft-trace geolocation risk: HIGH
  [1] flyover  10.5–70.0 s   closest≈40.1s   77.7–114.4 Hz  harmonics=1  ~65 m/s
  [2] flyover  47.0–148.8 s  closest≈105.8s  30.0–155.9 Hz  harmonics=5  ~69 m/s
  Gaps between flyovers (s): [65.7]  <- strongest triangulation clue
  steady tone 60.0 Hz  [60 Hz grid: reveals grid region; ENF can reveal recording time]
```

## Live mode (experimental)

`quietsky.stream.StreamProcessor` processes audio block by block with a fixed output delay, the shape a streaming plugin needs. `quietsky stream IN OUT --latency 30` runs a file through it exactly as live mode would.

**The physics sets the price.** Before closest approach, an approaching plane's tone is a steady, high, nearly flat pitch, indistinguishable from any other steady tone. The falling Doppler glide that identifies it only appears as the plane passes. A live tool can't cut what it can't identify yet, so how much of each flyover gets removed depends on how long the output is held back:

| Output delay | Loud flyover residual (orig 5.1, noise floor ~1.3) | Quiet flyover (orig 3.4) | What's covered |
|---|---|---|---|
| 3 s | ~4.2 | ~2.7 | only after the pass is identified; the approach leaks |
| 20 s | 2.3 | 1.9 | most of the approach |
| 35 s | 1.9 | 1.6 | matches the whole-file scrubber (~2.0 / ~1.6) |

Runs about 2.2× faster than real time on one Python core.

How it works: every 3 s, the batch detector (`core.detect`) runs on the last 48 s of audio. Its fitted curves are notched out of the delayed output by overlap-add, with the same dB-matched noise refill as the batch scrubber, and extended by 3τ because the fit extrapolates well. Because overlapping windows re-test nearly the same audio dozens of times, a curve is only cut once **three** refreshes produce fits that trace the same tone, and curves below 60 Hz (where rumble and hum make false glides) are ignored. In testing, that left plane-free audio bit-exact in every scene tried. `--aggressive` cuts after two agreeing refreshes: it catches more, but altered plane-free audio in every scene tested.

For a streamer, the practical setup is a ~30 s broadcast delay. For anything sensitive or archived, scrub the file afterwards with `quietsky scrub`. A real plugin (OBS/VST) would port the DSP core to native code; channels are currently processed independently.

## How it works

1. **Spectrogram:** audio is analyzed at 4 kHz with about 1 Hz resolution (30–1500 Hz) and whitened against the local spectral floor.
2. **Matched filter:** on a log-frequency axis, every straight-line flyover is the same falling S-curve, `f(t) = f_mid·(1 − s/2·tanh((t−t0)/τ))`, shifted in time and frequency. One 2D correlation per (shift, τ) template scans every position. Two details matter:
   - The template is a *contrast* template (the curve minus the parallel bands beside it), so smooth background can't score.
   - Template scores are summed across harmonics 1–4, because aircraft tones come in harmonic families and noise doesn't.
3. **Per-file false-alarm floor:** flyover pitch only ever *falls*, so the best match of a time-reversed (rising) template measures how high noise scores in that specific recording. A detection must beat that floor by a set margin.
4. **Confirmation:** each candidate's ridge is extracted and curve-fit (R² ≥ 0.75, implied speed ≥ ~30 m/s). The fit is loose on purpose: real aircraft turn and change speed, so their curves aren't ideal. The per-file noise floor carries the false-alarm control. Fragments and harmonics are then merged into flyover events.
5. **Scrub:** masks follow the fitted curves, padded by 3τ past the detected ends (real curves last about 2 minutes, and faint tails still score), plus integer harmonics of the strongest track. Removed bins are **refilled with noise at the local floor level**. The refill level is calibrated on a dB spectrogram, and the mask edges are raised-cosine tapered. An untrimmed refill or a hard edge leaves a faint curve-shaped ridge or step, and so does a plain notch: it leaves a dark Doppler-shaped groove that a person can trace by eye, which is how Colsto handled his faintest curve. Refill also beats his "boost it back" test: the tone is replaced, not just attenuated. The output is then re-scanned to verify.
6. **Exposure:** reads flight-path-mapper's per-city GeoJSON and counts aircraft passing within N km of *your* spot per day, with nearest-pass distances and aircraft types.

## Validation (synthetic, `pytest tests`)

The test scenes are physically modeled: straight-line flyovers with propagation delay, 1/r falloff, four blade-pass harmonics, brown + white ambient noise, and 60/180 Hz hum.

| | Result |
|---|---|
| False positives, noise + hum only | 0 / 10 scenes |
| Detection, tones at −14 dB vs. ambient | both flyovers found, 5 / 5 seeds |
| Detection, tones at −17 dB | at least one flyover found; often misses the other |
| Detection, ~100 m/s flyover (his example's speed) | found in 4 / 5 seeds |
| Trace energy on the true flyover curves (noise-only baseline ≈ 1.2–1.3) | 3.4 / 5.1 → 1.6 / 2.0 after scrub (video → AAC round trip); re-scan: LOW |
| Refilled curve vs. untouched background, dB spectrogram | +0.01 vs −0.04 dB, i.e. indistinguishable |
| Re-scan after scrub, 6 refill seeds × 2 scenes | LOW every time |
| Background level after 2-pass scrub (2 scenes) | 300–3000 Hz: −1.6 / −2.8 dB; below 300 Hz: −1.3 / −3.5 dB (see Limits) |

## Limits

- **Validated on synthetic audio only.** The next step is real field recordings with known flyovers (e.g. your own recording plus a flight-tracker log).
- Detects traces from about −17 dB below ambient. Colsto located a recording from a curve he could only see by hand-tracing it, which is likely fainter than this detector catches. Treat LOW as "no obvious traces," not as a guarantee.
- Partial scrubbing interacts with his silence filter. If quietsky removes some flyovers and misses a faint one, the leftover curve plus the false silence around it will mislead a search more than it helps one. Still, don't rely on that.
- Targets tonal propeller traces, matching his method (he filtered jets out). Jet fan tones above 1.5 kHz are out of the analysis band.
- Speed: about 5 s to scan 150 s of audio and about 70 s to scrub it. The scrub has not been optimized yet.
- Residue: about 0.3–0.7 above the noise baseline remains on the true curves. That's below what the detector can see, but it isn't zero.
- Audio cost: masks are ±(4 Hz + 4%) around each curve and its harmonics, padded by 3τ, and harmonics are extended to the 6th or 1.2 kHz, whichever is lower. Narrow bands of other sound along those curves (voice included) are replaced with matched noise. The background measured 1.3–3.5 dB quieter in my two test scenes. That's the core trade-off: refill dB-matched to look like background on a spectrogram carries less energy than the sound it replaces. An earlier version extended harmonics to 2.5 kHz; the masks merged and cut the voice band by 4.5 dB, so it's capped now. The refill level was calibrated against this detector's spectrogram settings; an analyst using very different resolution could see a slightly different picture.
- Two scrub passes are the default and stop early once the re-scan is clean. Removing a loud flyover can expose a faint one the first scan missed, and the second pass catches it.
- For truly sensitive footage the only complete defense is replacing the ambient track (music, voice-over, or room tone recorded elsewhere). Cutting and re-editing also helps, because it breaks the time gaps between flyovers.
- `exposure`: flight-path-mapper's GeoJSON has no timestamps or altitude, so this gives daily volume, not an hourly schedule.

## License

MIT. See [LICENSE](LICENSE).
