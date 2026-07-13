
import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt


def load_or_make_clean(path="clean.wav", fs_default=16000):
    """Load clean.wav if it exists; otherwise synthesise a fallback test tone."""
    try:
        fs, sig = wavfile.read(path)
        if sig.ndim > 1:                 
            sig = sig.mean(axis=1)
        sig = sig.astype(np.float32)
        sig /= (np.max(np.abs(sig)) + 1e-9)  
        print(f"  Loaded real speech from {path}  (fs={fs} Hz, {len(sig)/fs:.1f}s)")
        return fs, sig
    except FileNotFoundError:
        fs = fs_default
        t = np.linspace(0, 2.0, int(fs * 2.0), endpoint=False)
        sig = 0.5 * np.sin(2 * np.pi * 220 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))
        sig = sig.astype(np.float32)
        print(f"  No {path} found — using a synthetic test tone (fs={fs} Hz).")
        print(f"  TIP: create real speech first, then re-run (see the run guide).")
        return fs, sig


def simulate_room(clean, fs, room_dim, rt60_target, src_pos, mic_pos):
    """
    Place the clean signal in a shoebox room and record it at a mic.

    rt60_target = how long sound takes to die away (seconds).
       small/absorbent room -> short RT60 (e.g. 0.3s, like a car cabin)
       large/hard room       -> long  RT60 (e.g. 0.8s, like a hall)

    Returns (reverberant_signal, room_impulse_response, measured_rt60).
    """
    e_absorption, max_order = pra.inverse_sabine(rt60_target, room_dim)

    room = pra.ShoeBox(
        room_dim, fs=fs,
        materials=pra.Material(e_absorption),
        max_order=max_order,
    )
    room.add_source(src_pos, signal=clean)
    room.add_microphone(np.array(mic_pos).reshape(3, 1))
    room.simulate()

    reverberant = room.mic_array.signals[0, :]
    rir = room.rir[0][0]                    
    measured_rt60 = room.measure_rt60()[0, 0]
    return reverberant, rir, measured_rt60


def add_noise(signal, snr_db):
    """Mix in white noise so that the speech-to-noise ratio = snr_db decibels.
    Lower SNR = noisier. In-car conditions are often a harsh 0–10 dB."""
    sig_power = np.mean(signal ** 2)
    noise = np.random.randn(len(signal)).astype(np.float32)
    noise_power = np.mean(noise ** 2)
    scale = np.sqrt(sig_power / (noise_power * 10 ** (snr_db / 10)))
    return signal + scale * noise


def save_wav(path, fs, sig):
    sig = sig / (np.max(np.abs(sig)) + 1e-9)       
    wavfile.write(path, fs, (sig * 32767).astype(np.int16))
    print(f"  saved {path}")


def main():
    print("\n=== Synthetic acoustic environment demo ===\n")

    print("[1] Getting a clean speech signal...")
    fs, clean = load_or_make_clean()

    print("\n[2] Simulating the SAME speech in two different rooms...")
    small_dim = [3.0, 2.0, 1.2]      
    large_dim = [25.0, 20.0, 12.0]     
    src = [1.0, 1.0, 0.6]
    mic_small = [2.0, 1.5, 0.6]
    mic_large = [6.0, 5.0, 1.5]

    rev_small, rir_small, rt60_small = simulate_room(
        clean, fs, small_dim, 0.3, src, mic_small)
    print(f"    Small room: target RT60 0.30s -> measured {rt60_small:.2f}s "
          f"(short tail, like a car cabin)")

    rev_large, rir_large, rt60_large = simulate_room(
        clean, fs, large_dim, 0.8, [2, 2, 1], mic_large)
    print(f"    Large room: target RT60 0.80s -> measured {rt60_large:.2f}s "
          f"(long echoey tail)")

    print("\n[3] Adding background noise at SNR = 5 dB (noisy, in-car-like)...")
    noisy = add_noise(rev_small, snr_db=5)

    print("\n[4] Saving audio files (open them and listen in order)...")
    save_wav("1_clean.wav", fs, clean)
    save_wav("2_reverberant_small_room.wav", fs, rev_small)
    save_wav("3_reverberant_large_room.wav", fs, rev_large)
    save_wav("4_noisy.wav", fs, noisy)

    print("\n[5] Plotting the room impulse responses -> rir.png")
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    for ax, rir, title in [
        (axes[0], rir_small, f"Small room RIR (RT60 ~ {rt60_small:.2f}s)"),
        (axes[1], rir_large, f"Large room RIR (RT60 ~ {rt60_large:.2f}s)"),
    ]:
        t = np.arange(len(rir)) / fs
        ax.plot(t, rir, linewidth=0.7)
        ax.set_title(title)
        ax.set_ylabel("amplitude")
        ax.grid(alpha=0.3)
    axes[1].set_xlabel("time (seconds)")
    fig.suptitle("Room Impulse Response = the room's acoustic fingerprint", weight="bold")
    fig.tight_layout()
    fig.savefig("rir.png", dpi=120)

    print("[6] Plotting spectrograms -> spectrograms.png")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, sig, title in [
        (axes[0], clean, "clean"),
        (axes[1], rev_small, "+ reverb (small room)"),
        (axes[2], noisy, "+ reverb + noise"),
    ]:
        ax.specgram(sig, NFFT=512, Fs=fs, noverlap=256)
        ax.set_title(title)
        ax.set_xlabel("time (s)")
    axes[0].set_ylabel("frequency (Hz)")
    fig.suptitle("Same speech, progressively corrupted — this is what you train on",
                 weight="bold")
    fig.tight_layout()
    fig.savefig("spectrograms.png", dpi=120)

    print("\n=== Done. ===")
    print("Listen to the 4 .wav files in order, then look at rir.png and spectrograms.png.")
    print("That progression IS  observed = clean * RIR + noise.\n")


if __name__ == "__main__":
    main()
