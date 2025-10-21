# RNN/features.py
import torch
import torch.nn as nn
import numpy as np
import parselmouth
from parselmouth.praat import call
from scipy.signal import filtfilt
import librosa

class PathologyFeatureExtractor(nn.Module):
    """
    Estrae feature patologiche classiche da un segnale audio.
    """
    def __init__(
        self,
        sample_rate,
        praat_step_size_s,
        min_f0_hz,
        max_f0_hz,
        gne_resample_rate,
        gne_lpc_order,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.praat_step_size_s = praat_step_size_s
        self.min_f0_hz = min_f0_hz
        self.max_f0_hz = max_f0_hz
        self.gne_resample_rate = gne_resample_rate
        self.gne_lpc_order = gne_lpc_order

    def _extract_praat_features(self, audio_numpy):
        snd = parselmouth.Sound(audio_numpy, sampling_frequency=self.sample_rate)
        pitch = call(snd, "To Pitch", self.praat_step_size_s, self.min_f0_hz, self.max_f0_hz)
        f0 = call(pitch, "Get mean", 0, 0, "Hertz")
        if np.isnan(f0): f0 = 0.0

        hnr = call(snd, "To Harmonicity (cc)", 0.01, self.min_f0_hz, 0.1, 1.0)
        hnr_mean = call(hnr, "Get mean", 0, 0)
        if np.isnan(hnr_mean): hnr_mean = 0.0

        point_process = call(snd, "To PointProcess (periodic, cc)", self.min_f0_hz, self.max_f0_hz)
        jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = call(point_process, "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        return [f0, hnr_mean, jitter, shimmer]

    def _extract_gne_feature(self, audio_numpy):
        try:
            target_sr = self.gne_resample_rate
            if self.sample_rate != target_sr:
                secs = len(audio_numpy) / self.sample_rate
                samps = int(secs * target_sr)
                audio_numpy = np.interp(np.linspace(0, secs, samps), np.linspace(0, secs, len(audio_numpy)), audio_numpy)

            A = librosa.lpc(audio_numpy, order=self.gne_lpc_order)
            excitation = filtfilt([1], A, audio_numpy)
            gne_proxy = np.std(excitation)
            return [gne_proxy]
        except Exception:
            return [0.0]

    def forward(self, wav, wav_lens):
        batch_size = wav.shape[0]
        output_features = []

        for i in range(batch_size):
            audio_sample = wav[i].cpu().numpy()
            praat_feats = self._extract_praat_features(audio_sample)
            gne_feat = self._extract_gne_feature(audio_sample)
            all_feats = praat_feats + gne_feat
            output_features.append(all_feats)

        feature_tensor = torch.tensor(output_features, device=wav.device).float()
        return feature_tensor.unsqueeze(1)
