import json
import torch
import torchaudio
import librosa
from speechbrain.augment.time_domain import AddNoise, AddReverb
import parselmouth
from parselmouth.praat import call
import numpy as np
import pysptk
from scipy.signal import lfilter, get_window
from scipy.linalg import toeplitz
import os
import urllib.request
import csv

def _download_augmentation_resources(resource_dir="RNN/resources"):
    """Downloads noise and RIR files if they don't exist."""
    os.makedirs(resource_dir, exist_ok=True)

    noise_dir = os.path.join(resource_dir, "noise")
    rir_dir = os.path.join(resource_dir, "rir")

    os.makedirs(noise_dir, exist_ok=True)
    os.makedirs(rir_dir, exist_ok=True)

    # URLs from the SpeechBrain repository
    noise_files = {"noise1.wav": "https://raw.githubusercontent.com/speechbrain/speechbrain/develop/tests/samples/noise/noise1.wav"}
    rir_files = {"rir1.wav": "https://raw.githubusercontent.com/speechbrain/speechbrain/develop/tests/samples/RIRs/rir1.wav"}

    # Download noise
    for filename, url in noise_files.items():
        path = os.path.join(noise_dir, filename)
        if not os.path.exists(path):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, path)

    # Download RIR
    for filename, url in rir_files.items():
        path = os.path.join(rir_dir, filename)
        if not os.path.exists(path):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, path)

    # Create manifest files
    noise_manifest_path = os.path.join(resource_dir, "noise_manifest.csv")
    rir_manifest_path = os.path.join(resource_dir, "rir_manifest.csv")

    if not os.path.exists(noise_manifest_path):
        with open(noise_manifest_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "duration", "wav", "wav_format", "wav_opts"])
            writer.writerow(["noise_0", "0.0", os.path.join(noise_dir, "noise1.wav"), "wav", ""])

    if not os.path.exists(rir_manifest_path):
        with open(rir_manifest_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "duration", "wav", "wav_format", "wav_opts"])
            writer.writerow(["rir_0", "0.0", os.path.join(rir_dir, "rir1.wav"), "wav", ""])

    return noise_manifest_path, rir_manifest_path


class SpeechProcessor:
    def __init__(self, hyperparams_file):
        """
        Initializes the processor by loading hyperparameters and setting up augmentation modules.

        Arguments:
            hyperparams_file (str): Path to the JSON file with hyperparameters.
        """
        with open(hyperparams_file, 'r') as f:
            self.hparams = json.load(f)

        # Ensure augmentation resources are available
        noise_manifest, rir_manifest = _download_augmentation_resources()

        # Initialize SpeechBrain augmentation modules
        self.noise_augmenter = AddNoise(
            noise_manifest,
            snr_low=self.hparams["augmentation"]["environmental_corruption"]["noise_snr_low_db"],
            snr_high=self.hparams["augmentation"]["environmental_corruption"]["noise_snr_high_db"],
        )
        self.reverb_augmenter = AddReverb(
             rir_manifest,
            rir_scale_factor=self.hparams["augmentation"]["environmental_corruption"]["rir_scale_factor"],
        )

    def process_audio(self, wav_path, augment=True):
        """
        Applies the full processing pipeline to a single audio file.

        Arguments:
            wav_path (str): The path to the input .wav file.
            augment (bool): Whether to apply data augmentation.

        Returns:
            torch.Tensor: A tensor containing the extracted features.
        """
        # 1. Load audio
        signal, sample_rate = self._load_audio(wav_path)

        # 2. Preprocessing
        signal = self._remove_silence(signal, sample_rate)
        signal = self._normalize_loudness(signal, sample_rate)

        # 3. Augmentation (if enabled)
        if augment:
            signal = self._apply_augmentation(signal, sample_rate)

        # 4. Feature Extraction
        features = self._extract_features(signal.numpy(), sample_rate)

        return torch.tensor(features, dtype=torch.float32)

    def _load_audio(self, wav_path):
        """Loads an audio file, resamples it to 16kHz, and converts it to a mono tensor."""
        signal, sample_rate = torchaudio.load(wav_path)

        target_sr = 16000
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
            signal = resampler(signal)

        if signal.ndim > 1:
            signal = torch.mean(signal, dim=0)

        return signal.squeeze(), target_sr

    def _remove_silence(self, signal, sample_rate):
        """Removes silence from the signal using librosa."""
        params = self.hparams["preprocessing"]["vad"]

        # Convert torch tensor to numpy array for librosa
        signal_np = signal.numpy()

        # librosa.effects.split returns a list of [start, end] sample indices
        clips = librosa.effects.split(
            signal_np,
            top_db=abs(params["threshold_db"]), # top_db is positive
            frame_length=int(params["frame_length_ms"] / 1000 * sample_rate),
            hop_length=int(params["hop_length_ms"] / 1000 * sample_rate)
        )

        # Concatenate the non-silent clips
        if len(clips) > 0:
            non_silent_signal = np.concatenate([signal_np[start:end] for start, end in clips])
            return torch.from_numpy(non_silent_signal)

        # If no speech is detected, return the original signal
        return signal

    def _normalize_loudness(self, signal, sample_rate):
        """Normalizes the loudness of the signal to a target level."""
        params = self.hparams["preprocessing"]["loudness_normalization"]

        # Ensure signal is floating point
        signal = signal.float()

        current_rms = torch.sqrt(torch.mean(signal**2))
        if current_rms == 0: return signal # Avoid division by zero for silent signals

        target_rms = 10**(params["target_dbfs"] / 20.0)
        gain = target_rms / current_rms

        return signal * gain

    def _apply_augmentation(self, signal, sample_rate):
        """Applies configured augmentations."""
        params = self.hparams["augmentation"]["environmental_corruption"]

        if torch.rand(1) < params["noise_prob"]:
            # AddNoise requires a lengths tensor.
            signal = self.noise_augmenter(
                signal.unsqueeze(0), lengths=torch.tensor([1.0])
            ).squeeze(0)

        if torch.rand(1) < params["reverb_prob"]:
            # AddReverb does not require the lengths tensor.
            signal = self.reverb_augmenter(signal.unsqueeze(0)).squeeze(0)

        return signal

    def _extract_features(self, signal_np, sample_rate):
        """Extracts features over frames to create a sequence."""

        # Use parameters from the JSON file
        params_js = self.hparams["feature_extraction"]["pathological_features"]["jitter_shimmer"]
        frame_length_s = params_js["window_size_s"]
        hop_length_s = params_js["step_size_s"]

        frame_length = int(frame_length_s * sample_rate)
        hop_length = int(hop_length_s * sample_rate)

        # Use librosa to frame the signal
        frames = librosa.util.frame(signal_np, frame_length=frame_length, hop_length=hop_length, axis=0)

        feature_sequence = []

        for frame in frames:
            sound = parselmouth.Sound(frame, sampling_frequency=sample_rate)

            f0_params = self.hparams["feature_extraction"]["pathological_features"]["f0"]
            pitch = sound.to_pitch(pitch_floor=f0_params["min_f0_hz"], pitch_ceiling=f0_params["max_f0_hz"])
            f0_mean = call(pitch, "Get mean", 0.0, 0.0, "Hertz")

            point_process = call(sound, "To PointProcess (periodic, cc)", f0_params["min_f0_hz"], f0_params["max_f0_hz"])
            jitter = call(point_process, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3) * 100
            shimmer = call([sound, point_process], "Get shimmer (local)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
            hnr = call(sound, "To Harmonicity (cc)", 0.01, f0_params["min_f0_hz"], 0.1, 1.0)
            hnr_mean = call(hnr, "Get mean", 0.0, 0.0)

            gne_mean = self._calculate_gne(frame, sample_rate)

            frame_features = np.array([f0_mean, jitter, shimmer, hnr_mean, gne_mean])
            frame_features = np.nan_to_num(frame_features, nan=0.0)
            feature_sequence.append(frame_features)

        return np.array(feature_sequence)

    def _calculate_gne(self, signal, fs):
        """
        Calculates the Glottal-to-Noise Excitation (GNE) ratio.
        Implementation based on the algorithm described by Michaelis et al.
        """
        params = self.hparams["feature_extraction"]["gne"]

        # Resample to the target rate for GNE analysis
        if fs != params["new_sample_rate_hz"]:
            resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=params["new_sample_rate_hz"])
            signal = resampler(torch.from_numpy(signal).float()).numpy()
            fs = params["new_sample_rate_hz"]

        # 1. Pre-emphasis (parameters are symbolic but use JSON values)
        pre_emphasis_coeff = 1.0 - (params["bandwidth_hz"] / fs)
        signal = lfilter([1, -pre_emphasis_coeff], 1, signal)
        win = get_window("hann", int(0.02 * fs)) # 20ms window

        # 2. LPC analysis to get the inverse filter
        lpc_order = params["lpc_order"]

        # We need at least lpc_order + 1 samples to compute LPC
        if len(signal) < lpc_order + 1:
            return 0.0

        a = pysptk.sptk.lpc(signal.astype(np.float64), order=lpc_order)

        # 3. Inverse filtering to get the excitation signal
        excitation = lfilter(a, 1, signal)

        # 4. Decompose the excitation signal into glottal (periodic) and noise components
        # This is a simplified decomposition. A full implementation is more complex.
        # We find the pitch to estimate the periodic part.
        pitch_period = int(fs / 75) # Assume a pitch around 75 Hz

        if len(excitation) < pitch_period:
            return 0.0

        num_periods = len(excitation) // pitch_period

        if num_periods == 0:
            return 0.0

        excitation_matrix = excitation[:num_periods * pitch_period].reshape(num_periods, pitch_period)

        glottal_component = np.mean(excitation_matrix, axis=0)
        glottal_component = np.tile(glottal_component, num_periods)

        noise_component = excitation[:len(glottal_component)] - glottal_component

        # 5. Calculate the GNE ratio
        energy_glottal = np.sum(glottal_component ** 2)
        energy_noise = np.sum(noise_component ** 2)

        if energy_noise == 0:
            return 0.0 # Or a large number, as noise is zero

        gne_ratio = 10 * np.log10(energy_glottal / energy_noise)

        return gne_ratio

if __name__ == "__main__":
    import os

    # Get the absolute path to the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define paths relative to the script's location
    hyperparams_path = os.path.join(script_dir, "hyperparameters.json")
    dummy_wav_path = os.path.join(script_dir, "dummy.wav")


    # 1. Create a dummy wav file for testing
    dummy_signal = torch.randn(16000 * 2) # 2 seconds of noise
    torchaudio.save(dummy_wav_path, dummy_signal.unsqueeze(0), 16000)

    # 2. Load hyperparameters using an absolute path
    try:
        processor = SpeechProcessor(hyperparams_path)
        features = processor.process_audio(dummy_wav_path, augment=True)
        print("Processing successful!")
        print(f"Extracted features tensor: {features}")
        print(f"Feature shape: {features.shape}")

    except FileNotFoundError as e:
        print(f"\nAn unexpected FileNotFoundError occurred.")
        print(f"Error details: {e}")

    # 3. Clean up the dummy file
    if os.path.exists(dummy_wav_path):
        os.remove(dummy_wav_path)
