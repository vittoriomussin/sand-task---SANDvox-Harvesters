# Pipeline di Analisi Vocale con RNN

Questo progetto implementa una pipeline completa per l'analisi di file audio vocali, l'estrazione di feature patologiche e l'addestramento di un modello di deep learning (RNN) per compiti di classificazione e regressione.

## Struttura del Progetto

-   `hyperparameters.json`: File di configurazione centrale per tutti i parametri di preprocessing, augmentation e feature extraction.
-   `requirements.txt`: Elenco delle dipendenze Python necessarie per eseguire il progetto.
-   `prepare_data.py`: Script per preparare i dati. Scansiona i file audio, li unisce ai metadati e crea i file manifest per i set di training, validazione e test.
-   `speech_processor.py`: Modulo principale che contiene la logica per l'elaborazione dell'audio. Carica i file, rimuove il silenzio, normalizza il volume, applica augmentation e estrae le feature.
-   `dataset.py`: Definisce la classe `AudioDataset` di PyTorch, che carica i dati dai manifest e li elabora al volo.
-   `model.py`: Definisce l'architettura del modello `SimpleRNN` (basato su LSTM).
-   `train.py`: Script principale per lanciare l'addestramento e la valutazione del modello.

## Istruzioni per l'Uso

### 1. Installazione delle Dipendenze

Prima di tutto, è necessario installare tutte le librerie Python richieste. Assicurarsi di avere anche le librerie di sistema per l'audio come `libsndfile` e `sox`.

```bash
pip install -r requirements.txt
```

### 2. Preparazione dei Dati

Lo script `prepare_data.py` crea i file `train.csv`, `valid.csv` e `test.csv` necessari per l'addestramento.

**Uso:**

```bash
python3 prepare_data.py --metadata_file <percorso_del_file_metadati.csv> --data_folder <percorso_della_cartella_audio> --output_folder <cartella_di_output_manifests>
```

**Esempio:**

```bash
python3 prepare_data.py --metadata_file data/metadata.csv --data_folder data/audio --output_folder RNN/manifests
```

### 3. Addestramento del Modello

Una volta che i manifest sono pronti, è possibile lanciare l'addestramento con lo script `train.py`.

**Argomenti:**

-   `--hyperparams_file`: Percorso del file JSON di iperparametri.
-   `--train_manifest`: Percorso del file `train.csv`.
-   `--valid_manifest`: Percorso del file `valid.csv`.
-   `--task_type`: Il tipo di compito (`classification` o `regression`).
-   `--target_column`: Il nome della colonna nel manifest da usare come etichetta (es. `Class` o `Age`).
-   `--save_path`: (Opzionale) Directory dove salvare il modello migliore.

**Esempio di Classificazione:**

```bash
python3 train.py \
    --hyperparams_file RNN/hyperparameters.json \
    --train_manifest RNN/manifests/train.csv \
    --valid_manifest RNN/manifests/valid.csv \
    --task_type classification \
    --target_column Class
```

**Esempio di Regressione:**

```bash
python3 train.py \
    --hyperparams_file RNN/hyperparameters.json \
    --train_manifest RNN/manifests/train.csv \
    --valid_manifest RNN/manifests/valid.csv \
    --task_type regression \
    --target_column Age
```
