import os
import shutil
import subprocess
import tempfile
import warnings

import numpy as np
import pyroomacoustics as pra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")  

import librosa  
from sklearn.pipeline import make_pipeline  
from sklearn.preprocessing import StandardScaler 
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score  


WORDS = ["yes", "no", "stop", "go"]     
FS = 16000
N_PER_WORD = 30
AUG_PER_TRAIN_CLIP = 3                   
SEED = 0


def generate_clean_dataset(words, fs, n_per_word, rng):
    """Real `say` voices on macOS; synthetic word-like signals otherwise."""
    if shutil.which("say"):
        print("  Using macOS `say` voices for real speech.")
        return _with_say(words, fs, n_per_word)
    print("  No `say` found — synthesising word-like signals (still runs fine).")
    return _synthetic(words, fs, n_per_word, rng)


def _english_voices():
    """List installed English `say` voices."""
    try:
        out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
    except Exception:
        return ["Alex"]
    voices = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1].startswith("en"):
            voices.append(parts[0])
    return voices or ["Alex"]


def _with_say(words, fs, n_per_word):
    voices = _english_voices()
    rates = [150, 175, 200, 225]
    audio, labels = [], []
    for label, word in enumerate(words):
        made = 0
        for voice in voices:
            for rate in rates:
                if made >= n_per_word:
                    break
                with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as tf:
                    path = tf.name
                try:
                    subprocess.run(
                        ["say", "-v", voice, "-r", str(rate), "-o", path, word],
                        check=True, capture_output=True,
                    )
                    y, _ = librosa.load(path, sr=fs)
                    if len(y) > 0:
                        audio.append(y.astype(np.float32))
                        labels.append(label)
                        made += 1
                except Exception:
                    pass
                finally:
                    if os.path.exists(path):
                        os.remove(path)
            if made >= n_per_word:
                break
    return audio, np.array(labels)


def _synthetic(words, fs, n_per_word, rng):
    """Each word = a distinct formant pattern with per-sample jitter."""
    formant_sets = {
        "yes":  [520, 1500, 2500],
        "no":   [400, 900, 2200],
        "stop": [700, 1250, 2650],
        "go":   [350, 820, 2000],
    }
    audio, labels = [], []
    for label, word in enumerate(words):
        base = formant_sets.get(word, [500, 1200, 2400])
        for _ in range(n_per_word):
            dur = rng.uniform(0.45, 0.7)
            t = np.linspace(0, dur, int(fs * dur), endpoint=False)
            sig = np.zeros_like(t)
            for f in base:
                f_j = f * rng.uniform(0.95, 1.05)          
                for h in (1, 2):                       
                    sig += (1.0 / h) * np.sin(2 * np.pi * f_j * h * t)
            env = np.hanning(len(t))                  
            sig = sig * env
            sig += 0.01 * rng.standard_normal(len(t))    
            sig = sig.astype(np.float32)
            sig /= (np.max(np.abs(sig)) + 1e-9)
            audio.append(sig)
            labels.append(label)
    return audio, np.array(labels)


def apply_environment(clip, fs, rng):
    """Convolve with a random room RIR and add noise at a random SNR."""
    rt60 = rng.uniform(0.2, 0.6)
    dim = [rng.uniform(2.5, 6.0), rng.uniform(2.5, 5.0), rng.uniform(2.2, 3.0)]
    try:
        e_abs, max_order = pra.inverse_sabine(rt60, dim)
        room = pra.ShoeBox(dim, fs=fs, materials=pra.Material(e_abs),
                           max_order=min(int(max_order), 8))
        src = [rng.uniform(0.5, dim[0] - 0.5), rng.uniform(0.5, dim[1] - 0.5), 1.0]
        mic = [rng.uniform(0.5, dim[0] - 0.5), rng.uniform(0.5, dim[1] - 0.5), 1.2]
        room.add_source(src, signal=clip)
        room.add_microphone(np.array(mic).reshape(3, 1))
        room.simulate()
        rev = room.mic_array.signals[0, :]
    except Exception:
        rev = clip.copy()
    snr_db = rng.uniform(0, 12)
    sig_power = np.mean(rev ** 2) + 1e-12
    noise = rng.standard_normal(len(rev)).astype(np.float32)
    scale = np.sqrt(sig_power / (np.mean(noise ** 2) * 10 ** (snr_db / 10)))
    return (rev + scale * noise).astype(np.float32)


def features(clip, fs):
    """20 MFCCs -> mean + std over time = a fixed-length 40-d vector."""
    mfcc = librosa.feature.mfcc(y=clip, sr=fs, n_mfcc=20)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])


def feature_matrix(clips, fs):
    return np.vstack([features(c, fs) for c in clips])


def main():
    rng = np.random.default_rng(SEED)
    print("\n=== Does synthetic acoustic augmentation improve robustness? ===\n")

    print("[1] Building a clean keyword dataset:", ", ".join(WORDS))
    audio, labels = generate_clean_dataset(WORDS, FS, N_PER_WORD, rng)
    print(f"    {len(audio)} clean clips total.")

    idx = np.arange(len(audio))
    rng.shuffle(idx)
    n_test = int(0.3 * len(idx))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    train_clips = [audio[i] for i in train_idx]
    test_clips = [audio[i] for i in test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    print("\n[2] Creating the noisy/reverberant TEST set (the 'in-car' condition)...")
    test_noisy = [apply_environment(c, FS, rng) for c in test_clips]

    print("[3] Extracting MFCC features (librosa)...")
    Xtr_clean = feature_matrix(train_clips, FS)
    Xte_clean = feature_matrix(test_clips, FS)
    Xte_noisy = feature_matrix(test_noisy, FS)

    print("\n[4] Training Model A (clean data only)...")
    model_a = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10))
    model_a.fit(Xtr_clean, y_train)

    print("[5] Training Model B (clean + synthetic reverb/noise augmentation)...")
    aug_clips, aug_labels = [], []
    for c, lab in zip(train_clips, y_train):
        for _ in range(AUG_PER_TRAIN_CLIP):
            aug_clips.append(apply_environment(c, FS, rng))
            aug_labels.append(lab)
    Xtr_aug = np.vstack([Xtr_clean, feature_matrix(aug_clips, FS)])
    ytr_aug = np.concatenate([y_train, np.array(aug_labels)])
    model_b = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10))
    model_b.fit(Xtr_aug, ytr_aug)

    print("\n[6] Evaluating both models...\n")
    results = {
        "A (clean-only)": {
            "clean test": accuracy_score(y_test, model_a.predict(Xte_clean)),
            "noisy test": accuracy_score(y_test, model_a.predict(Xte_noisy)),
        },
        "B (augmented)": {
            "clean test": accuracy_score(y_test, model_b.predict(Xte_clean)),
            "noisy test": accuracy_score(y_test, model_b.predict(Xte_noisy)),
        },
    }

    print(f"    {'model':<18}{'clean test':>12}{'noisy test':>12}")
    print(f"    {'-'*42}")
    for name, r in results.items():
        print(f"    {name:<18}{r['clean test']*100:>10.1f}%{r['noisy test']*100:>11.1f}%")

    gap = (results["B (augmented)"]["noisy test"]
           - results["A (clean-only)"]["noisy test"]) * 100
    print(f"\n    >>> On the noisy 'in-car' test set, augmentation changed accuracy "
          f"by {gap:+.1f} points.")
    print("    This is the thesis hypothesis: synthetic acoustic data -> robustness.\n")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels_x = ["clean test", "noisy test"]
    a_vals = [results["A (clean-only)"][k] * 100 for k in labels_x]
    b_vals = [results["B (augmented)"][k] * 100 for k in labels_x]
    x = np.arange(len(labels_x))
    w = 0.35
    ax.bar(x - w/2, a_vals, w, label="A: clean-only", color="#9CA3AF")
    ax.bar(x + w/2, b_vals, w, label="B: + synthetic augmentation", color="#2563EB")
    ax.set_ylabel("keyword accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_x)
    ax.set_ylim(0, 105)
    ax.set_title("Synthetic acoustic augmentation improves robustness", weight="bold")
    ax.legend()
    for i, v in enumerate(a_vals):
        ax.text(i - w/2, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9)
    for i, v in enumerate(b_vals):
        ax.text(i + w/2, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig("augmentation_result.png", dpi=120)
    print("    Saved chart -> augmentation_result.png\n")


if __name__ == "__main__":
    main()
