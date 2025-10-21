# RNN/dataio.py
import speechbrain as sb
from speechbrain.dataio.dataset import DynamicItemDataset
import torch

def dataio_prepare(hparams):
    """
    Prepara i dataset per il training, la validazione e il test.

    Questa funzione definisce la pipeline di elaborazione dinamica per ogni
    campione audio.

    Args:
        hparams (dict): Il dizionario degli iperparametri caricato dal file .yaml.

    Returns:
        dict: Un dizionario contenente i dataset di train, validation e test.
    """

    # 1. Carica il file JSON generato da prepare_data.py
    # SpeechBrain creerà un dataset da questo file.
    data_json_path = hparams["json_data_file"]

    # Crea i dataset per train, validation e test
    # NOTA: Per ora usiamo lo stesso JSON. In un esperimento reale, si dovrebbero
    # creare JSON separati per train/valid/test. La logica di split verrà
    # gestita nello script di training.
    datasets = {}
    for dataset_name in ["train", "valid", "test"]:
        datasets[dataset_name] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_json_path
        )

    # 2. Definisci la pipeline di elaborazione dinamica (per tutti i set)
    # Questa pipeline viene eseguita "al volo" per ogni campione.

    @sb.utils.data_pipeline.takes("wav_path")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav_path):
        """Carica il segnale audio dal percorso specificato."""
        # Carica l'audio e lo resampla alla frequenza desiderata
        sig = sb.dataio.dataio.read_audio(wav_path)
        return sig

    @sb.utils.data_pipeline.takes("sig")
    @sb.utils.data_pipeline.provides("processed_sig")
    def preprocessing_pipeline(sig):
        """Applica la catena di preprocessing (VAD, Normalizzazione, Augmentation)."""
        # La logica effettiva è incapsulata nella classe PreprocessingChain
        processed_sig = hparams["preprocessing_chain"](sig.unsqueeze(0), torch.ones(1))
        return processed_sig.squeeze(0)

    @sb.utils.data_pipeline.takes("processed_sig")
    @sb.utils.data_pipeline.provides("features")
    def feature_pipeline(processed_sig):
        """Estrae le feature patologiche dal segnale pre-processato."""
        # La logica effettiva è incapsulata nella classe PathologyFeatureExtractor
        features = hparams["feature_extractor"](processed_sig.unsqueeze(0), torch.ones(1))
        # Rimuove la dimensione del batch (B, T, F) -> (T, F)
        return features.squeeze(0)

    @sb.utils.data_pipeline.takes("class_label")
    @sb.utils.data_pipeline.provides("label_tensor")
    def label_pipeline(class_label):
        """Converte l'etichetta in un tensore."""
        if hparams["task_type"] == 'classification':
            # Per la classificazione, l'etichetta è un intero
            label_tensor = torch.LongTensor([class_label])
        elif hparams["task_type"] == 'regression':
            # Per la regressione, l'etichetta è un float
            label_tensor = torch.FloatTensor([class_label])
        else:
            raise ValueError("`task_type` in hparams deve essere 'classification' o 'regression'")
        return label_tensor

    # Aggiungi le funzioni della pipeline ai dataset
    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
    sb.dataio.dataset.add_dynamic_item(datasets, preprocessing_pipeline)
    sb.dataio.dataset.add_dynamic_item(datasets, feature_pipeline)
    sb.dataio.dataset.add_dynamic_item(datasets, label_pipeline)

    # 3. Definisci le colonne di output
    # Specifica quali elementi dinamici devono essere restituiti dal dataloader.
    # 'features' sarà l'input del modello, 'label_tensor' l'output atteso.
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "features", "label_tensor"],
    )

    return datasets["train"], datasets["valid"], datasets["test"]
