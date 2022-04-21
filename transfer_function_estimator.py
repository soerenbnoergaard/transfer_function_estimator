import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

ABSMODES = ["median", "mean", "full"]
PHASEMODES = ["median", "mean", "full", "zero"]

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("infile", help="Input stereo wav file (default channels: left=PICKUP, right=MIC)")
    parser.add_argument("--swapchannels", "-s", help="Swap input channels so left=MIC and right=PICKUP", action="store_true")
    parser.add_argument("--length", "-n", help="Impulse response length", default=2048, type=int)
    parser.add_argument("--output", "-o", help="Label used for the output file", default="ir_" + str(int(time.time())), type=str)
    parser.add_argument("--absmode", "-a", help=f"Mode for estimating absolute value ({'|'.join(ABSMODES)})", default="median", type=str)
    parser.add_argument("--phasemode", "-p", help=f"Mode for estimating phase ({'|'.join(PHASEMODES)})", default="full", type=str)
    parser.add_argument("--normfreq", "-f", help="Frequency where the impulse response is normalized to zero gain (Hz)", default=200, type=float)
    parser.add_argument("--all", help="Generate and impulse response for each absmode/phasemode combination", action="store_true")
    parser.add_argument("--show", help="Show graphs after completion", action="store_true")
    parser.add_argument("--stats", help="Calculate correlation statistics", action="store_true")

    args = parser.parse_args()

    if args.all:
        output = args.output
        for a in ABSMODES:
            for p in PHASEMODES:
                args.absmode = a
                args.phasemode = p
                args.output = f"{output}_{a}_{p}"
                run(args)
    else:
        run(args)

    if args.show:
        plt.show()

def run(args):
    print("")
    print(f"Input file: {args.infile}")
    print(f"Magnitude estimation mode: {args.absmode}")
    print(f"Phase estimation mode: {args.phasemode}")

    data, fs = sf.read(args.infile)
    if args.swapchannels:
        ydata, xdata = data.T
    else:
        xdata, ydata = data.T

    # Estimate frequency response for time-blocks of samples
    H_blocks = np.array([
        estimate_frequency_response(x, y)
        for x, y in block_generator(xdata, ydata, args.length)
    ])

    # Estimate the frequency response based on all the recorded data
    H_full = estimate_frequency_response(xdata, ydata, args.length)

    # Estimate magnitude response
    if args.absmode == "median":
        H_abs = np.median(np.abs(H_blocks), axis=0)
    elif args.absmode == "mean":
        H_abs = np.mean(np.abs(H_blocks), axis=0)
    elif args.absmode == "full":
        H_abs = np.abs(H_full)
    else:
        raise ValueError(f"Unknown abs mode {args.absmode!r} - valid options: {ABSMODES!r}")

    # Estimate phase response
    if args.phasemode == "median":
        H_angle = np.median(np.unwrap(np.angle(H_blocks)), axis=0)
    elif args.phasemode == "mean":
        H_angle = np.mean(np.unwrap(np.angle(H_blocks)), axis=0)
    elif args.phasemode == "full":
        H_angle = np.angle(H_full)
    elif args.phasemode == "zero":
        H_angle = np.zeros(args.length)
    else:
        raise ValueError(f"Unknown phase mode {args.phasemode!r} - valid options: {PHASEMODES!r}")

    # Combine frequency response
    H = H_abs * np.exp(1j*H_angle)

    # Calculate impulse response
    h = np.real(np.fft.ifft(H))

    # Apply fade to impulse response
    N = len(h)
    n = np.arange(N)
    # fade = ((-n + N)*(n>N//2) + (N//2)*(n <= N//2)) / (N//2)
    fade = (-n + N)/N
    h *= fade

    # Normalize impulse response
    w = 2*np.pi * args.normfreq/fs
    k = np.arange(len(h))
    h /= np.abs(np.sum(h*np.exp(-1j*w*k)))

    # Update frequency response with the impulse response changes
    H = np.fft.fft(h)

    # Save results
    save_impulse_response(f"{args.output}.wav", fs, h)
    plot_results(H, h, fs, f"{args.output}.png")

    # Calculate statistics
    if args.stats:
        ydata_est = np.convolve(xdata, h, "same")
        print("Correlation coefficient: {:11.6f}".format(np.corrcoef(ydata, ydata_est)[0, 1]))
        print("Mean square error:       {:11.6f}".format(np.square(ydata - ydata_est).mean()))

def plot_results(H, h, fs, filename):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=[8, 6])

    N = len(H)
    f = np.linspace(0, fs, N)
    end = N//2
    ax1_2 = ax1.twinx()
    ax1.semilogx(f[0:end], 20*np.log10(np.abs(H[0:end])), "C0")
    ax1_2.semilogx(f[0:end], np.rad2deg(np.unwrap(np.angle(H[0:end]))), "C1")

    # Impulse response
    ax2.plot(h)

    fig.tight_layout()
    fig.savefig(filename)

def save_impulse_response(filename, fs, h):
    sf.write(filename, h, samplerate=fs, subtype="PCM_24")

def estimate_frequency_response(x, y, size=None):
    if size is None:
        size = len(x)
    # w = np.hanning(len(x))
    # w = np.ones(len(x))
    # w = np.hamming(len(x))
    w = np.blackman(len(x))
    X = np.fft.fft(x * w, size)
    Y = np.fft.fft(y * w, size)
    H = Y/X
    return H

def block_generator(xdata, ydata, size):
    N = len(xdata)
    assert N == len(xdata) == len(ydata)
    for n in range(0, N, size):
        start = n
        stop = n + size
        if stop > N:
            break
        yield xdata[start:stop], ydata[start:stop]

if __name__ == "__main__":
    main()

