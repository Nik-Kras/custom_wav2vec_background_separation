# Place to write a data generation bit to create two waveforms, a nignal and a noise
import numpy as np
import soundfile
import matplotlib.pyplot as plt

BIT_RATE = 16
SAMPLING_RATE = 16_000
DURATION = 2 # Seconds
NUMBER_OF_SAMPLES = SAMPLING_RATE * DURATION
FREQUENCY = 440 * 2 * np.pi
MAX_AMP = np.iinfo(np.int16).max  # Max amplitude for 16-bit audio

NOISE_STRENGTH = 0.1

def generate_wave():
    clock_ticks = np.arange(start=0, stop=NUMBER_OF_SAMPLES, step=1, dtype=np.int16)
    x = MAX_AMP * np.sin(clock_ticks*FREQUENCY / NUMBER_OF_SAMPLES)
    soundfile.write(file='data_gen/data/sine.wav', data=x.astype(np.int16), samplerate=SAMPLING_RATE, subtype='PCM_16')
    return x

def generate_wave_with_noise():
    clock_ticks = np.arange(start=0, stop=NUMBER_OF_SAMPLES, step=1, dtype=np.int16)
    x = MAX_AMP * np.sin(clock_ticks*FREQUENCY / NUMBER_OF_SAMPLES)
    mu, sigma = 0, MAX_AMP * NOISE_STRENGTH  # mean and standard deviation
    s = np.random.normal(mu, sigma, NUMBER_OF_SAMPLES)
    x = x + s
    
    # Normalize after noise is added
    x = np.clip(x, a_min=-MAX_AMP, a_max=MAX_AMP)
    
    soundfile.write(file='data_gen/data/sine_noisy.wav', data=x.astype(np.int16), samplerate=SAMPLING_RATE, subtype='PCM_16')
    return x


def plot_signal(signal: np.ndarray, steps: int = SAMPLING_RATE/20):
    plt.plot(np.arange(0, len(signal), 1), signal)
    if steps:
        plt.xlim(0, steps)
    plt.hlines(0, 0, len(signal), color='red')
    plt.show()


if __name__ == "__main__":
    x = generate_wave()
    x = generate_wave_with_noise()
