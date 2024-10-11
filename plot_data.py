from matplotlib import pyplot as plt
import numpy as np
from numpy.fft import fft

from util import format_to_4_digits, get_labels, get_vec


def make_plots_waveform(start: int, end: int):
    labels = get_labels()
    for i in range(start, end):
        new_index = format_to_4_digits(i)
        
        vec = get_vec(i)
        t = np.linspace(0, len(vec) / 44100, len(vec))

        plt.plot(t, vec)
        plt.xlabel("Time, s")
        plt.ylabel("Amplitude")
        plt.title(f"Waveform of the signal, T{new_index}, class={labels[i - 1]}")

        plt.savefig(f".\\fig\\waveform\\T{new_index}_plot.png")

        plt.clf()


def make_plots_fft(start: int, end: int):
    labels = get_labels()
    for i in range(start, end):
        new_index = format_to_4_digits(i)
        
        f = 15000
        vec = get_vec(i)
        t = np.linspace(0, f, f)

        freq = fft(vec, f)

        plt.plot(t, abs(freq))
        plt.xlabel("Frequency, Hz")
        plt.ylabel("Amplitude")
        plt.title(f"Spectrum of the signal, T{new_index}, class={int(labels[i - 1])}")

        plt.savefig(f".\\fig\\frequency-domain\\T{new_index}_plot_freq.png")

        plt.clf()


if __name__ == "__main__":
    make_plots_waveform(start=20, end=40)
    make_plots_fft(start=20, end=40)
