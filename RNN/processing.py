# RNN/processing.py
import torch
import speechbrain as sb
from speechbrain.augment.time_domain import AddNoise, AddReverb
from speechbrain.dataio.dataio import read_audio

class PreprocessingChain(torch.nn.Module):
    """
    Una classe che incapsula l'intera catena di preprocessing audio.
    """
    def __init__(
        self,
        sample_rate,
        vad_threshold_db,
        vad_frame_length_ms,
        vad_hop_length_ms,
        noise_data_folder=None,
        noise_snr_low_db=5,
        noise_snr_high_db=20,
        noise_prob=1.0,
        reverb_data_folder=None,
        rir_scale_factor=1.0,
        reverb_prob=1.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.vad_threshold_db = vad_threshold_db
        self.vad_frame_length_ms = vad_frame_length_ms
        self.vad_hop_length_ms = vad_hop_length_ms
        self.noise_prob = noise_prob
        self.reverb_prob = reverb_prob

        # Inizializza i componenti di augmentation
        self.noise_augment = None
        if noise_data_folder:
            self.noise_augment = AddNoise(
                wavs_folder=noise_data_folder,
                snr_low=noise_snr_low_db,
                snr_high=noise_snr_high_db
            )

        self.reverb_augment = None
        if reverb_data_folder:
            self.reverb_augment = AddReverb(
                rir_folder=reverb_data_folder,
                rir_scale_factor=rir_scale_factor
            )

    def forward(self, wav, wav_lens, stage='train'):
        frame_length_samples = int(self.sample_rate * self.vad_frame_length_ms / 1000)
        hop_length_samples = int(self.sample_rate * self.vad_hop_length_ms / 1000)

        rms = sb.processing.features.STFT(
            sample_rate=self.sample_rate,
            win_length=frame_length_samples,
            hop_length=hop_length_samples
        )(wav).abs().mean(dim=-1)

        is_speech = (rms > (rms.max() - abs(self.vad_threshold_db)))

        upsampled_mask = torch.nn.functional.interpolate(
            is_speech.float().unsqueeze(1),
            size=wav.shape[1],
            mode='nearest'
        ).squeeze(1)

        wav = wav * upsampled_mask
        wav = wav / torch.max(torch.abs(wav), dim=-1, keepdim=True)[0]

        if stage == 'train':
            if self.noise_augment and torch.rand(1) < self.noise_prob:
                wav = self.noise_augment(wav, wav_lens)

            if self.reverb_augment and torch.rand(1) < self.reverb_prob:
                wav = self.reverb_augment(wav, wav_lens)

        return wav
