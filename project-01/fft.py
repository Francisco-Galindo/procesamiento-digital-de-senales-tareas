import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.io import wavfile
import os
import math


def bit_reverse_index(n: int, num_bits: int) -> int:
    """
    Reverse the binary representation of integer `n` using `num_bits` bits.

    Example (num_bits = 10):
        n = 0b0000000001  (decimal 1)
        reversed = 0b1000000000  (decimal 512)
    """
    result = 0
    for _ in range(num_bits):
        result = (result << 1) | (n & 1)   # shift result left, OR in LSB of n
        n >>= 1                            # next bit
    return result


def bit_reversal_permutation(x: list) -> list:
    """
    Reorder the input sequence `x` according to the bit-reversal permutation
    required by the Decimation-in-Time (DIT) FFT.
    """
    N = len(x)
    num_bits = int(math.log2(N))
    x_br = [0.0] * N

    for i in range(N):
        j = bit_reverse_index(i, num_bits)
        x_br[j] = x[i]

    return x_br


def twiddle_factor(k: int, N: int) -> complex:
    """
    Compute the twiddle factor W_N^k = e^{-j * 2π * k / N}.
    """
    angle = -2.0 * math.pi * k / N
    return complex(math.cos(angle), math.sin(angle))


def fft_radix2_dit_1024(x: list) -> list:
    """
    Compute the 1024-point DFT using the Radix-2 Decimation-in-Time algorithm.
    """
    N = len(x)
    if N != 1024:
        raise ValueError(f"This FFT is hard-coded for N=1024. Got N={N}.")

    NUM_STAGES = 10

    # Pre-populate output with bit_reversal
    X = [complex(v) for v in bit_reversal_permutation(x)]

    for s in range(1, NUM_STAGES + 1): 

        group_size = 2 ** s 
        butterfly_span = group_size // 2 

        for k in range(0, N, group_size):

            for j_idx in range(butterfly_span):
                u_index = k + j_idx
                v_index = k + j_idx + butterfly_span

                W = twiddle_factor(j_idx * (N // group_size), N)

                # Butterfly
                t = W * X[v_index]
                u = X[u_index]
                X[u_index] = u + t
                X[v_index] = u - t

    return X


# ──────────────────────────────────────────────────────────────────────────────
# SECTION Written by Anthropic's Claude. Read an audio file and compare this
# implementation with the numpy.fft
# ──────────────────────────────────────────────────────────────────────────────

def load_audio_window(filepath: str, start_sample: int = 0) -> tuple:
    """
    Load a WAV file and extract a 1024-sample window for analysis.

    Processing chain
    ----------------
    1. Read WAV → convert to mono (average channels if stereo)
    2. Normalise to float64 in [-1, +1]
    3. Slice 1024 samples starting at `start_sample`
    4. Apply a Hanning window to reduce spectral leakage

    Parameters
    ----------
    filepath     : path to the .wav audio file
    start_sample : index of the first sample in the window

    Returns
    -------
    (sample_rate, windowed_samples, raw_samples)
        sample_rate      : integer Hz
        windowed_samples : 1024-element float list (Hanning-weighted)
        raw_samples      : 1024-element float list (unweighted, for display)
    """
    sample_rate, data = wavfile.read(filepath)

    # Convert to float64
    if data.dtype == np.int16:
        data = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float64) - 128.0) / 128.0
    else:
        data = data.astype(np.float64)

    # Mono mix-down
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Ensure we have enough samples
    total_samples = len(data)
    end_sample    = start_sample + 1024
    if end_sample > total_samples:
        start_sample = max(0, total_samples - 1024)
        end_sample   = start_sample + 1024

    raw_window = list(data[start_sample:end_sample])

    # Hanning window: w[n] = 0.5 · (1 − cos(2π·n / (N−1)))
    hanning = [0.5 * (1.0 - math.cos(2.0 * math.pi * n / 1023))
               for n in range(1024)]
    windowed = [raw_window[n] * hanning[n] for n in range(1024)]

    return sample_rate, windowed, raw_window


def generate_synthetic_audio(freqs_hz: list, sample_rate: int = 44100,
                             duration_s: float = 0.05) -> tuple:
    """
    Generate a synthetic multi-tone signal when no WAV file is available.

    Parameters
    ----------
    freqs_hz    : list of sinusoid frequencies in Hz to mix
    sample_rate : samples per second
    duration_s  : signal duration in seconds (must cover ≥ 1024 samples)

    Returns
    -------
    (sample_rate, windowed_samples, raw_samples)  — same shape as load_audio_window
    """
    n_samples = max(1024, int(sample_rate * duration_s))
    t         = [i / sample_rate for i in range(n_samples)]
    signal    = [sum(math.sin(2.0 * math.pi * f * ti) for f in freqs_hz) / len(freqs_hz)
                 for ti in t]

    # Normalise
    peak   = max(abs(v) for v in signal)
    signal = [v / peak for v in signal]

    raw_window = signal[:1024]

    # Hanning window
    hanning  = [0.5 * (1.0 - math.cos(2.0 * math.pi * n / 1023))
                for n in range(1024)]
    windowed = [raw_window[n] * hanning[n] for n in range(1024)]

    return sample_rate, windowed, raw_window


def compute_magnitude_spectrum(X: list, N: int = 1024) -> list:
    """
    Compute the one-sided magnitude spectrum from the complex DFT output.

    Only bins 0 … N/2 are returned (the positive-frequency half); the upper
    half is the complex conjugate mirror for a real-valued input.

    Magnitude is normalised by N so that amplitude is independent of window size.

    Parameters
    ----------
    X : list of N complex DFT coefficients
    N : DFT size (1024)

    Returns
    -------
    List of N//2 + 1 non-negative magnitude values.
    """
    one_sided = N // 2 + 1
    magnitudes = [abs(X[k]) / N for k in range(one_sided)]
    # Double the non-DC, non-Nyquist bins to account for the discarded mirror
    for k in range(1, one_sided - 1):
        magnitudes[k] *= 2.0
    return magnitudes


def verify_against_numpy(x: list, custom_X: list, tol: float = 1e-6) -> dict:
    """
    Compare the custom FFT output to numpy.fft.fft for correctness verification.

    Parameters
    ----------
    x        : original time-domain samples
    custom_X : DFT coefficients from fft_radix2_dit_1024()
    tol      : maximum acceptable absolute error per bin

    Returns
    -------
    Dictionary with keys:
        max_abs_error  : float — worst-case |custom[k] − numpy[k]|
        mean_abs_error : float — average |custom[k] − numpy[k]|
        passed         : bool  — True if max error ≤ tol
    """
    numpy_X        = np.fft.fft(x)
    errors         = [abs(custom_X[k] - numpy_X[k]) for k in range(len(custom_X))]
    max_err        = max(errors)
    mean_err       = sum(errors) / len(errors)
    return {
        "max_abs_error"  : max_err,
        "mean_abs_error" : mean_err,
        "passed"         : max_err <= tol,
    }


def plot_results(raw_samples: list, windowed_samples: list,
                 custom_X: list, sample_rate: int,
                 verification: dict, signal_label: str = "Signal") -> None:
    """
    Produce a four-panel figure:

        Panel 1 — Raw 1024-sample time-domain window
        Panel 2 — Hanning-windowed samples
        Panel 3 — One-sided magnitude spectrum (custom FFT, dB scale)
        Panel 4 — Verification: custom vs numpy magnitude overlay
    """
    N          = 1024
    magnitudes = compute_magnitude_spectrum(custom_X, N)
    freq_bins  = [k * sample_rate / N for k in range(N // 2 + 1)]

    # numpy reference
    numpy_X    = np.fft.fft(windowed_samples)
    numpy_mag  = [abs(numpy_X[k]) / N for k in range(N // 2 + 1)]
    for k in range(1, N // 2):
        numpy_mag[k] *= 2.0

    # dB conversion (floor at -120 dB)
    def to_db(m):
        return [20.0 * math.log10(max(v, 1e-12)) for v in m]

    custom_db = to_db(magnitudes)
    numpy_db  = to_db(numpy_mag)
    time_axis = [n / sample_rate * 1000 for n in range(N)]   # milliseconds

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("#0f0f1a")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    panel_style = dict(facecolor="#1a1a2e")
    label_style = dict(color="#c0c0d8", fontsize=10)
    title_style = dict(color="#e8e8ff", fontsize=12, fontweight="bold", pad=10)
    tick_style  = dict(colors="#8080a0")

    # ── Panel 1: Raw signal ────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0], **panel_style)
    ax1.plot(time_axis, raw_samples, color="#4fc3f7", linewidth=0.8, alpha=0.9)
    ax1.set_title("Raw 1024-Sample Window", **title_style)
    ax1.set_xlabel("Time (ms)", **label_style)
    ax1.set_ylabel("Amplitude", **label_style)
    ax1.tick_params(colors=tick_style["colors"])
    ax1.spines[:].set_color("#3a3a5c")
    ax1.grid(True, color="#2a2a45", linestyle="--", linewidth=0.5)

    # ── Panel 2: Hanning-windowed signal ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1], **panel_style)
    ax2.plot(time_axis, windowed_samples, color="#f48fb1", linewidth=0.8, alpha=0.9)
    ax2.set_title("After Hanning Window (spectral leakage suppressed)", **title_style)
    ax2.set_xlabel("Time (ms)", **label_style)
    ax2.set_ylabel("Amplitude", **label_style)
    ax2.tick_params(colors=tick_style["colors"])
    ax2.spines[:].set_color("#3a3a5c")
    ax2.grid(True, color="#2a2a45", linestyle="--", linewidth=0.5)

    # ── Panel 3: Custom FFT magnitude spectrum ─────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0], **panel_style)
    ax3.plot(freq_bins, custom_db, color="#69f0ae", linewidth=1.0)
    ax3.fill_between(freq_bins, min(custom_db), custom_db,
                     alpha=0.15, color="#69f0ae")
    ax3.set_title("Magnitude Spectrum — Custom Radix-2 DIT FFT (10 stages)", **title_style)
    ax3.set_xlabel("Frequency (Hz)", **label_style)
    ax3.set_ylabel("Magnitude (dB)", **label_style)
    ax3.tick_params(colors=tick_style["colors"])
    ax3.spines[:].set_color("#3a3a5c")
    ax3.grid(True, color="#2a2a45", linestyle="--", linewidth=0.5)
    ax3.set_xlim(0, sample_rate / 2)
    ax3.set_ylim(bottom=-100)

    # ── Panel 4: Verification overlay ─────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1], **panel_style)
    ax4.plot(freq_bins, numpy_db,  color="#ffb300", linewidth=2.5,
             alpha=0.7, label="numpy.fft (reference)", linestyle="--")
    ax4.plot(freq_bins, custom_db, color="#69f0ae", linewidth=1.2,
             alpha=0.9, label="Custom DIT FFT")
    ax4.set_title("Verification: Custom FFT vs numpy.fft", **title_style)
    ax4.set_xlabel("Frequency (Hz)", **label_style)
    ax4.set_ylabel("Magnitude (dB)", **label_style)
    ax4.tick_params(colors=tick_style["colors"])
    ax4.spines[:].set_color("#3a3a5c")
    ax4.grid(True, color="#2a2a45", linestyle="--", linewidth=0.5)
    ax4.set_xlim(0, sample_rate / 2)
    ax4.set_ylim(bottom=-100)
    leg = ax4.legend(facecolor="#1a1a2e", edgecolor="#3a3a5c",
                     labelcolor="#c0c0d8", fontsize=9)

    # Verification badge
    status_color = "#69f0ae" if verification["passed"] else "#ff5252"
    status_text  = ("✓  PASS" if verification["passed"] else "✗  FAIL")
    fig.text(0.5, 0.02,
             f"{status_text}  |  Max error: {verification['max_abs_error']:.2e}"
             f"  |  Mean error: {verification['mean_abs_error']:.2e}",
             ha="center", fontsize=11, color=status_color,
             bbox=dict(facecolor="#12122a", edgecolor=status_color,
                       boxstyle="round,pad=0.4"))

    fig.suptitle(
        f"1024-Point Radix-2 DIT FFT  —  {signal_label}\n"
        f"N=1024 · 10 stages · Sample rate={sample_rate} Hz",
        color="#ffffff", fontsize=14, fontweight="bold", y=0.98
    )

    out_path = "./outputs/fft_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n[✓] Figure saved → {out_path}")
    plt.close()


def main():
    print("=" * 72)
    print("  1024-Point Radix-2 Decimation-in-Time (DIT) FFT — from scratch")
    print("=" * 72)

    # ── Try to load a real WAV file; fall back to synthetic tone ──────────────
    WAV_PATH = "audio_sample.wav"   # ← replace with your own WAV file path

    if os.path.exists(WAV_PATH):
        print(f"\n[✓] Loading WAV file: {WAV_PATH}")
        sample_rate, windowed_samples, raw_samples = load_audio_window(
            WAV_PATH, start_sample=0
        )
        signal_label = os.path.basename(WAV_PATH)
    else:
        print("\n[!] No WAV file found — generating synthetic 3-tone signal:")
        SYNTH_FREQS = [440, 1760, 8000]   # A4, an overtone, high partial
        print(f"    Frequencies: {SYNTH_FREQS} Hz")
        sample_rate  = 44100
        sample_rate, windowed_samples, raw_samples = generate_synthetic_audio(
            SYNTH_FREQS, sample_rate=sample_rate
        )
        signal_label = f"Synthetic tones: {SYNTH_FREQS} Hz"

    print(f"    Sample rate : {sample_rate} Hz")
    print(f"    Window size : 1024 samples")
    print(f"    Stages      : 10  (log₂(1024) = 10)")

    # ── Run the custom FFT ────────────────────────────────────────────────────
    print("\n[…] Running custom Radix-2 DIT FFT …")
    custom_X = fft_radix2_dit_1024(windowed_samples)
    print("[✓] FFT complete.")

    # ── Verify against numpy ──────────────────────────────────────────────────
    print("\n[…] Verifying against numpy.fft.fft …")
    verification = verify_against_numpy(windowed_samples, custom_X)
    status = "PASS ✓" if verification["passed"] else "FAIL ✗"
    print(f"    Max absolute error  : {verification['max_abs_error']:.6e}")
    print(f"    Mean absolute error : {verification['mean_abs_error']:.6e}")
    print(f"    Verification status : {status}")

    # ── Show dominant frequency bins ──────────────────────────────────────────
    N          = 1024
    magnitudes = compute_magnitude_spectrum(custom_X, N)
    freq_bins  = [k * sample_rate / N for k in range(N // 2 + 1)]

    # Top-5 peaks
    indexed_mags  = sorted(enumerate(magnitudes), key=lambda x: x[1], reverse=True)
    print("\n[i] Top-5 spectral peaks (custom FFT):")
    print(f"    {'Bin':>5}  {'Freq (Hz)':>10}  {'Magnitude':>12}")
    print(f"    {'─'*5}  {'─'*10}  {'─'*12}")
    for i, (bin_idx, mag) in enumerate(indexed_mags[:5]):
        print(f"    {bin_idx:>5}  {freq_bins[bin_idx]:>10.2f}  {mag:>12.6f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    print("\n[…] Generating plots …")
    plot_results(raw_samples, windowed_samples, custom_X,
                 sample_rate, verification, signal_label)

    print("\n" + "=" * 72)
    print("  Done.  All outputs written to ./outputs/")
    print("=" * 72)


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
