"""
audio_pipeline.py  —  Task 3: Sound Data Collection & Processing
----------------------------------------------------------------
- Generates placeholder voice recordings with espeak-ng (distinct voice per
  identity) saying the two required phrases:
      "Yes, approve"  and  "Confirm transaction"
  REPLACE: record yourself and drop WAVs into data/audio/raw/ named
      <member>_<phrase>.wav   e.g.  member1_yes_approve.wav
- Plots waveform + spectrogram for each raw sample.
- Applies 3 augmentations per sample: pitch shift (+2 semitones),
  time stretch (0.85x), additive background noise.
- Extracts features -> audio_features.csv:
      13 MFCC means + 13 MFCC stds, spectral roll-off, RMS energy,
      zero-crossing rate, spectral centroid.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import librosa, librosa.display
import soundfile as sf
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_AUD, AUG_AUD = ROOT / "data/audio/raw", ROOT / "data/audio/augmented"
PLOTS = ROOT / "outputs/plots"
for p in (RAW_AUD, AUG_AUD, PLOTS): p.mkdir(parents=True, exist_ok=True)

SR = 22050
PHRASES = {"yes_approve": "Yes, approve", "confirm_transaction": "Confirm transaction"}
# identity -> (espeak voice, pitch, speed): distinct "voiceprints"
VOICES = {
    "member1":  ("en-us", 50, 150),
    "impostor1": ("en+f3", 70, 170),
    "impostor2": ("en+m5", 30, 130),
}
RNG = np.random.default_rng(11)


def collect_audio():
    existing = list(RAW_AUD.glob("*.wav"))
    if existing:
        print(f"Found {len(existing)} real recordings in {RAW_AUD} — using those.")
        return sorted(existing)
    paths = []
    for member, (voice, pitch, speed) in VOICES.items():
        for key, text in PHRASES.items():
            p = RAW_AUD / f"{member}_{key}.wav"
            subprocess.run(["espeak-ng", "-v", voice, "-p", str(pitch), "-s", str(speed),
                            "-w", str(p), text], check=True)
            paths.append(p)
    print(f"Generated {len(paths)} placeholder recordings -> {RAW_AUD} (REPLACE with real voice)")
    return paths


def plot_wave_spec(path: Path):
    y, sr = librosa.load(path, sr=SR)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=axes[0])
    axes[0].set(title=f"Waveform — {path.stem}", xlabel="Time (s)", ylabel="Amplitude")
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(S, sr=sr, x_axis="time", y_axis="hz", ax=axes[1])
    axes[1].set(title=f"Spectrogram — {path.stem}")
    fig.colorbar(img, ax=axes[1], format="%+2.0f dB")
    fig.tight_layout()
    fig.savefig(PLOTS / f"audio_{path.stem}.png", dpi=110)
    plt.close(fig)


def augment(path: Path) -> list[Path]:
    y, sr = librosa.load(path, sr=SR)
    variants = {
        "pitch": librosa.effects.pitch_shift(y, sr=sr, n_steps=2),
        "stretch": librosa.effects.time_stretch(y, rate=0.85),
        "noise": y + 0.005 * RNG.standard_normal(len(y)),
    }
    out = []
    for tag, ya in variants.items():
        p = AUG_AUD / f"{path.stem}_{tag}.wav"
        sf.write(p, ya, sr)
        out.append(p)
    return out


def extract_features(path: Path) -> dict:
    y, sr = librosa.load(path, sr=SR)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    member, phrase = path.stem.split("_")[0], "_".join(path.stem.split("_")[1:3])
    row = {"file": path.name, "member": member, "phrase": phrase,
           "label_authorized": int(member.startswith("member")),
           "spectral_rolloff": float(librosa.feature.spectral_rolloff(y=y, sr=sr).mean()),
           "spectral_centroid": float(librosa.feature.spectral_centroid(y=y, sr=sr).mean()),
           "rms_energy": float(librosa.feature.rms(y=y).mean()),
           "zcr": float(librosa.feature.zero_crossing_rate(y).mean())}
    row |= {f"mfcc_mean_{i}": float(v) for i, v in enumerate(mfcc.mean(axis=1))}
    row |= {f"mfcc_std_{i}": float(v) for i, v in enumerate(mfcc.std(axis=1))}
    return row


def main():
    raw = collect_audio()
    for p in raw:
        plot_wave_spec(p)
    print(f"Saved waveform+spectrogram plots for {len(raw)} samples -> {PLOTS}")

    all_paths = list(raw)
    for p in raw:
        all_paths += augment(p)
    print(f"Augmented: 3 variants per sample -> {len(all_paths)} total clips")

    df = pd.DataFrame([extract_features(p) for p in all_paths])
    out = ROOT / "data/processed/audio_features.csv"
    df.to_csv(out, index=False)
    print(f"Saved audio features {df.shape} -> {out}")


if __name__ == "__main__":
    main()
