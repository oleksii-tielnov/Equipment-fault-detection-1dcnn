from matplotlib import pyplot as plt
import numpy as np
from numpy.fft import fft

from util import format_to_4_digits, get_label, get_vec, extract_features


def make_plots(start: int, end: int, mode: str):
    match mode:
        case "fft":  # Fast Fourie Transform
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
        case "wf":  # waveform
            for i in range(start, end):
                T_index = format_to_4_digits(i)
                
                f = 15000
                vec, label = get_vec(i), get_label(i)
                t = np.linspace(0, f, f)

                freq = fft(vec, f)

                plt.plot(t, abs(freq))
                plt.xlabel("Frequency, Hz")
                plt.ylabel("Amplitude")
                plt.title(f"Spectrum of the signal, T{T_index}, class={label}")

                plt.savefig(f".\\fig\\frequency-domain\\T{T_index}_plot_freq.png")

                plt.clf()
        case "ef":  # extracted feature
            for i in range(start, end):
                T_index = format_to_4_digits(i)

                step = 100
                vec, label = get_vec(i), get_label(i)

                vec = extract_features(vec, step)
                t = np.linspace(0, 1, len(vec))

                plt.plot(t, vec, c='g')
                plt.xlabel("Samples")
                plt.ylabel("Amplitude")
                plt.title(f"Max pooling T{T_index}, class={label}")

                plt.savefig(f".\\fig\\extract-features\\T{T_index}_plot.png")

                plt.clf()


if __name__ == "__main__":
    make_plots(1, 2, "ef")
    