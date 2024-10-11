from matplotlib import pyplot as plt
import numpy as np
from numpy.fft import fft

from util import format_to_4_digits, get_label, get_vec


def make_plots_waveform(start: int, end: int):
    for i in range(start, end):
        T_index = format_to_4_digits(i)
        
        vec = get_vec(i)
        t = np.linspace(0, len(vec) / 44100, len(vec))

        plt.plot(t, vec)
        plt.xlabel("Time, s")
        plt.ylabel("Amplitude")
        plt.title(f"Waveform of the signal, T{T_index}, class={get_label(i)}")

        plt.savefig(f".\\fig\\waveform\\T{T_index}_plot.png")

        plt.clf()


def make_plots_fft(start: int, end: int):
    for i in range(start, end):
        T_index = format_to_4_digits(i)
        
        f = 15000
        vec = get_vec(i)
        t = np.linspace(0, f, f)

        freq = fft(vec, f)

        plt.plot(t, abs(freq))
        plt.xlabel("Frequency, Hz")
        plt.ylabel("Amplitude")
        plt.title(f"Spectrum of the signal, T{T_index}, class={get_label(i)}")

        plt.savefig(f".\\fig\\frequency-domain\\T{T_index}_plot_freq.png")

        plt.clf()


if __name__ == "__main__":
    make_plots_waveform(start=1, end=11)
    make_plots_fft(start=1, end=11)
