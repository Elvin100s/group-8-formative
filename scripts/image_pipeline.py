"""
image_pipeline.py  —  Task 2: Image Data Collection & Processing
----------------------------------------------------------------
- Generates synthetic placeholder "face" images for member1 (YOU — replace with
  real photos) in 3 expressions: neutral, smiling, surprised, plus 2 impostor
  identities so the face-recognition model has negative classes.
  REPLACE: drop real photos into data/images/raw/ named
      <member>_<expression>.jpg   e.g.  member1_neutral.jpg
  and this script will use them instead of generating placeholders.
- Applies 3 augmentations per image: rotation(±15°), horizontal flip, grayscale
  (+ brightness jitter on the rotated copy).
- Extracts features per image: 32-bin grayscale histogram (normalized) +
  16x16 downsampled pixel embedding (256 dims) -> image_features.csv
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageEnhance

ROOT = Path(__file__).resolve().parents[1]
RAW_IMG = ROOT / "data/images/raw"
AUG_IMG = ROOT / "data/images/augmented"
PLOTS = ROOT / "outputs/plots"
for p in (RAW_IMG, AUG_IMG, PLOTS): p.mkdir(parents=True, exist_ok=True)

MEMBERS = {   # identity -> (skin RGB, face width factor) ; member1 = authorized user
    "member1":  ((224, 172, 138), 1.00),
    "impostor1": ((188, 140, 110), 0.90),
    "impostor2": ((140, 100, 80), 1.10),
}
EXPRESSIONS = ["neutral", "smiling", "surprised"]
SIZE = 256
RNG = np.random.default_rng(7)


def draw_face(skin, wfac, expression) -> Image.Image:
    """Placeholder cartoon face — replace with real photos for submission."""
    img = Image.new("RGB", (SIZE, SIZE), (235, 235, 245))
    d = ImageDraw.Draw(img)
    cx, cy, w, h = SIZE // 2, SIZE // 2, int(90 * wfac), 110
    d.ellipse([cx - w, cy - h, cx + w, cy + h], fill=skin, outline=(60, 40, 30), width=3)
    ew = 26 if expression == "surprised" else 16          # eyes
    for ex in (cx - 40, cx + 40):
        d.ellipse([ex - 14, cy - 40 - 10, ex + 14, cy - 40 + 10], fill="white", outline="black", width=2)
        d.ellipse([ex - 6, cy - 40 - ew // 3, ex + 6, cy - 40 + ew // 3], fill=(40, 30, 30))
    yb = cy - 68                                          # eyebrows
    lift = 14 if expression == "surprised" else 0
    for ex in (cx - 40, cx + 40):
        d.line([ex - 16, yb - lift, ex + 16, yb - lift + (4 if expression == "neutral" else -4)],
               fill=(60, 40, 30), width=5)
    if expression == "smiling":                            # mouth
        d.arc([cx - 40, cy + 20, cx + 40, cy + 80], 10, 170, fill=(120, 30, 30), width=6)
    elif expression == "surprised":
        d.ellipse([cx - 18, cy + 40, cx + 18, cy + 80], fill=(120, 30, 30))
    else:
        d.line([cx - 30, cy + 55, cx + 30, cy + 55], fill=(120, 30, 30), width=6)
    return img


def collect_images():
    """Use real photos if present in data/images/raw, else generate placeholders."""
    existing = list(RAW_IMG.glob("*.jpg")) + list(RAW_IMG.glob("*.png"))
    if existing:
        print(f"Found {len(existing)} real images in {RAW_IMG} — using those.")
        return sorted(existing)
    paths = []
    for member, (skin, wfac) in MEMBERS.items():
        for expr in EXPRESSIONS:
            p = RAW_IMG / f"{member}_{expr}.jpg"
            draw_face(skin, wfac, expr).save(p, quality=95)
            paths.append(p)
    print(f"Generated {len(paths)} placeholder images -> {RAW_IMG} (REPLACE with real photos)")
    return paths


def augment(path: Path) -> list[Path]:
    img = Image.open(path).convert("RGB")
    stem = path.stem
    out = []
    rot = img.rotate(float(RNG.uniform(-15, 15)), fillcolor=(235, 235, 245))
    rot = ImageEnhance.Brightness(rot).enhance(float(RNG.uniform(0.8, 1.2)))
    variants = {
        "rot": rot,
        "flip": img.transpose(Image.FLIP_LEFT_RIGHT),
        "gray": img.convert("L").convert("RGB"),
    }
    for tag, im in variants.items():
        p = AUG_IMG / f"{stem}_{tag}.jpg"
        im.save(p, quality=95)
        out.append(p)
    return out


def extract_features(path: Path) -> dict:
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    gray = cv2.resize(gray, (SIZE, SIZE))
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-9)
    emb = cv2.resize(gray, (16, 16)).flatten() / 255.0
    member, expression = path.stem.split("_")[0], path.stem.split("_")[1]
    row = {"file": path.name, "member": member, "expression": expression,
           "label_authorized": int(member.startswith("member"))}
    row |= {f"hist_{i}": v for i, v in enumerate(hist)}
    row |= {f"emb_{i}": v for i, v in enumerate(emb)}
    return row


def main():
    raw = collect_images()

    # display grid of each identity's expressions (members first, then impostors)
    identities = sorted({p.stem.split("_")[0] for p in raw},
                        key=lambda m: (not m.startswith("member"), m))
    fig, axes = plt.subplots(len(identities), len(EXPRESSIONS),
                             figsize=(7, 2.4 * len(identities)), squeeze=False)
    for i, member in enumerate(identities):
        for j, expr in enumerate(EXPRESSIONS):
            for ext in ("jpg", "png"):
                p = RAW_IMG / f"{member}_{expr}.{ext}"
                if p.exists():
                    axes[i, j].imshow(Image.open(p))
                    break
            axes[i, j].set_title(f"{member} — {expr}", fontsize=9)
            axes[i, j].axis("off")
    fig.suptitle("Collected face images (3 expressions per identity)")
    fig.tight_layout(); fig.savefig(PLOTS / "image_samples_grid.png", dpi=120); plt.close(fig)

    all_paths = list(raw)
    for p in raw:
        all_paths += augment(p)
    print(f"Augmented: 3 variants per image -> {len(all_paths)} total images")

    df = pd.DataFrame([extract_features(p) for p in all_paths])
    out = ROOT / "data/processed/image_features.csv"
    df.to_csv(out, index=False)
    print(f"Saved image features {df.shape} -> {out}")


if __name__ == "__main__":
    main()
