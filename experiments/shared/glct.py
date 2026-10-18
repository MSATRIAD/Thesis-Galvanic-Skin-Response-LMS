"""
Generalized Lin Chirplet Transform (GLCT).

Formula:
    GLCT(t, f, β) = ∫ x(τ) · w(τ-t) · exp(-j(2πfτ + πβτ²)) dτ

Dimana:
    - x(τ) : sinyal input
    - w(τ-t): fungsi window (Gaussian)
    - f     : frekuensi
    - β     : chirp rate (laju perubahan frekuensi)
"""

import numpy as np
from scipy.signal import windows


def _gaussian_window(length):
    return windows.gaussian(length, std=length / 6)


def _chirplet_atom(length, freq, chirp_rate, fs):
    t = np.arange(length) / fs
    t_centered = t - t[len(t) // 2]

    win = _gaussian_window(length)

    phase = 2 * np.pi * freq * t_centered + np.pi * chirp_rate * t_centered ** 2
    atom = win * np.exp(-1j * phase)

    return atom


def glct_transform(signal, fs, n_freqs=128, chirp_rates=None,
                   window_length=64, hop_length=None):
    n_signal = len(signal)

    if hop_length is None:
        hop_length = window_length // 4

    if chirp_rates is None:
        chirp_rates = np.linspace(-10, 10, 21)

    freqs = np.linspace(0, fs / 2, n_freqs)

    time_positions = np.arange(0, n_signal - window_length, hop_length)
    n_times = len(time_positions)

    tfr = np.zeros((n_freqs, n_times))

    for beta in chirp_rates:
        tfr_beta = np.zeros((n_freqs, n_times))

        for t_idx, t_pos in enumerate(time_positions):
            segment = signal[t_pos: t_pos + window_length]

            for f_idx, freq in enumerate(freqs):
                atom = _chirplet_atom(window_length, freq, beta, fs)

                coeff = np.abs(np.sum(segment * np.conj(atom)))
                tfr_beta[f_idx, t_idx] = coeff

        tfr = np.maximum(tfr, tfr_beta)

    return tfr


def glct_transform_fast(signal, fs, n_freqs=128, chirp_rates=None,
                        window_length=64, hop_length=None):
    n_signal = len(signal)

    if hop_length is None:
        hop_length = window_length // 4

    if chirp_rates is None:
        chirp_rates = np.linspace(-10, 10, 21)

    win = _gaussian_window(window_length)

    t = np.arange(window_length) / fs
    t_centered = t - t[window_length // 2]

    time_positions = np.arange(0, n_signal - window_length, hop_length)
    n_times = len(time_positions)

    tfr = np.zeros((n_freqs, n_times))

    for beta in chirp_rates:
        dechirp = np.exp(-1j * np.pi * beta * t_centered ** 2)

        tfr_beta = np.zeros((n_freqs, n_times))

        for t_idx, t_pos in enumerate(time_positions):
            segment = signal[t_pos: t_pos + window_length]
            windowed = segment * win * dechirp

            spectrum = np.fft.fft(windowed, n=max(n_freqs * 2, window_length))
            magnitude = np.abs(spectrum[:n_freqs])

            tfr_beta[:, t_idx] = magnitude

        tfr = np.maximum(tfr, tfr_beta)

    return tfr
