import numpy as np
import scipy.fftpack
import librosa
import matplotlib.pyplot as plt


# 1. Pre-emphasis
# Pre-emphasis enhances high frequencies by attenuating low frequencies to balance the spectrum, commonly used to improve the clarity of speech signals.
def pre_emphasis(signal, alpha=0.97):
    # np.append keeps the first sample unchanged, applies the formula y(t) = x(t) - α*x(t-1) for the subsequent samples
    emphasized_signal = np.append(signal[0], signal[1:] - alpha * signal[:-1])
    return emphasized_signal

# 2. Frame signal (Framing)
# Framing is used to segment continuous speech signals into small segments, assuming that each frame is approximately stationary over a short period.
def framing(signal, frame_size, frame_stride, sample_rate):
    # Frame length and frame shift: frame_size = 0.025 seconds, frame_stride = 0.01 seconds
    frame_length = int(frame_size * sample_rate)
    frame_step = int(frame_stride * sample_rate)
    signal_length = len(signal)
    
    # Calculate the number of frames needed
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step)) + 1

    # Zero-padding the signal to ensure all frames have the same length
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(signal, z)

    # Create indices for the frames
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    return frames

# 3. Apply Hamming window (Windowing)
# Windowing is used to reduce the discontinuity at the frame edges, usually done with a Hamming window function.
def windowing(frames, frame_length):
    # Apply a Hamming window to each frame to gradually attenuate the signal at the ends
    frames *= np.hamming(frame_length)
    return frames

# 4. Short-Time Fourier Transform (STFT) and Power Spectrum
# STFT transforms time-domain signals into their frequency representation, computing the power spectrum.
def stft(frames, NFFT):
    # Calculate the magnitude of the Fourier Transform for each frame
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    # Compute the power spectrum (normalized square)
    pow_frames = ((1.0 / NFFT) * (mag_frames ** 2))
    return pow_frames

# 5. Mel Filter Banks (Mel Filter)
# The Mel filter converts linear frequencies into the Mel scale, which approximates human auditory perception, and applies filtering.
def mel_filter_banks(pow_frames, nfilt, NFFT, sample_rate):
    # Convert frequencies to the Mel scale
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Create equally spaced points on the Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel scale points back to linear frequencies
    bin = np.floor((NFFT + 1) * hz_points / sample_rate).astype(int)  # Calculate the corresponding FFT bin values for the frequency points

    # Create filter banks
    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = bin[m - 1]   # Left frequency point
        f_m = bin[m]             # Center frequency point
        f_m_plus = bin[m + 1]    # Right frequency point

        # Left slope
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        # Right slope
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    # Calculate the output of the filter banks
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Avoid log(0)
    filter_banks = 20 * np.log10(filter_banks)  # Convert to dB
    return filter_banks

# 6. Discrete Cosine Transform (DCT) to get MFCC
# Use the Discrete Cosine Transform to convert the output of the Mel filter banks into MFCC features.
def mfcc(filter_banks, num_ceps):
    # Apply DCT to each frame and take the first num_ceps coefficients
    mfcc = scipy.fftpack.dct(filter_banks, type=2, axis=1, norm='ortho')[:, :num_ceps]
    return mfcc

# 7. Dynamic Feature Extraction (First-order difference)
# Extract dynamic features by calculating the first-order difference over time from the MFCC.
def delta(mfcc, N=2):
    num_frames = len(mfcc)
    denominator = 2 * sum([i**2 for i in range(1, N+1)])  # Normalization factor
    delta_feat = np.empty_like(mfcc)
    
    # Boundary padding
    padded = np.pad(mfcc, ((N, N), (0, 0)), mode='edge')

    # Calculate the difference for each frame
    for t in range(num_frames):
        delta_feat[t] = np.dot(np.arange(-N, N+1), padded[t: t+2*N+1]) / denominator
    return delta_feat


# Load the audio and process MFCC
def process_mfcc(audio_path):
    # Load the audio file
    signal, sample_rate = librosa.load(audio_path, sr=None)
    
    # Parameter settings
    pre_emphasis_coeff = 0.97  # Pre-emphasis coefficient
    frame_size = 0.025         # Frame length (25 ms)
    frame_stride = 0.01        # Frame stride (10 ms)
    num_filters = 26           # Number of Mel filters
    NFFT = 512                 # FFT size
    num_ceps = 13              # Extract the first 13 MFCC coefficients

    # 1. Pre-emphasis
    emphasized_signal = pre_emphasis(signal, pre_emphasis_coeff)

    # 2. Frame segmentation
    frames = framing(emphasized_signal, frame_size, frame_stride, sample_rate)

    # 3. Windowing
    frames = windowing(frames, int(frame_size * sample_rate))

    # 4. STFT and power spectrum calculation
    pow_frames = stft(frames, NFFT)

    # 5. Mel filter banks
    filter_banks = mel_filter_banks(pow_frames, num_filters, NFFT, sample_rate)

    # 6. DCT to compute MFCC
    mfcc_feat = mfcc(filter_banks, num_ceps)

    # 7. Dynamic feature extraction
    delta_feat = delta(mfcc_feat)
    delta_delta_feat = delta(delta_feat)

    return mfcc_feat, delta_feat, delta_delta_feat

# Compare with the MFCC function in the librosa library
def compare_with_librosa(audio_path):
    # Use the librosa library to load the audio and compute MFCC
    signal, sample_rate = librosa.load(audio_path, sr=None)
    mfcc_librosa = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13, hop_length=int(0.01 * sample_rate), n_fft=512)
    return mfcc_librosa.T

# Main function
if __name__ == "__main__":
    wav_path = './recording_files/noisy_recording.wav'  # Input audio file path

    # Compute MFCC with custom implementation
    mfcc_feat, delta_feat, delta_delta_feat = process_mfcc(wav_path)
    
    # Compare with the MFCC computed by the librosa library
    mfcc_librosa = compare_with_librosa(wav_path)
    
    # Output the shapes of the custom and librosa MFCC results
    print("Shape of custom MFCC:", mfcc_feat.shape)
    print("Shape of librosa MFCC:", mfcc_librosa.shape)
    
    # Visualize and compare custom MFCC results with librosa's
    plt.figure(figsize=(10, 6))
    
    # Custom MFCC
    plt.subplot(1, 2, 1)
    plt.imshow(mfcc_feat.T, aspect='auto', origin='lower')
    plt.title('MFCC (Custom)')
    
    # Librosa MFCC
    plt.subplot(1, 2, 2)
    plt.imshow(mfcc_librosa.T, aspect='auto', origin='lower')
    plt.title('MFCC (Librosa)')
    
    plt.tight_layout()
    plt.show()
