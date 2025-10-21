# Progetto RNN per l'Analisi della Voce Patologica

Questo progetto implementa un sistema basato su Reti Neurali Ricorrenti (RNN) per l'analisi di segnali vocali, con un focus sull'estrazione di feature patologiche classiche. Il sistema è costruito utilizzando la libreria `speechbrain` ed è progettato per essere altamente modulare e configurabile.

Può essere utilizzato sia per compiti di **classificazione** (es. sano vs. patologico) che di **regressione** (es. predire un punteggio di severità).

## Struttura del Progetto

```
RNN/
├── hparams.yaml          # File di configurazione centrale per tutti gli iperparametri
├── prepare_data.py       # Script per preparare il file JSON dei dati
├── dataio.py             # Pipeline di caricamento e processamento dati
├── processing.py         # Modulo per il preprocessing audio (VAD, Normalizzazione, Augmentation)
├── features.py           # Modulo per l'estrazione delle feature (F0, Jitter, Shimmer, GNE)
├── model.py              # Definizione del modello RNN
├── train.py              # Script principale per lanciare l'addestramento
└── requirements.txt      # Lista delle dipendenze Python
```

## Guida Rapida

### 1. Installazione delle Dipendenze

Per installare tutte le librerie necessarie, esegui questo comando dalla cartella radice del progetto:

```bash
pip install -r RNN/requirements.txt
```

### 2. Preparazione dei Dati

Il sistema si aspetta una struttura specifica per i dati di input:

1.  **File Audio**: Tutti i file `.wav` devono essere organizzati in sottocartelle. Ad esempio:
    ```
    dati/
    ├── phonationA/
    │   ├── ID000_phonationA.wav
    │   └── ID001_phonationA.wav
    └── rhythmKA/
        ├── ID000_rhythmKA.wav
        └── ID001_rhythmKA.wav
    ```

2.  **File Metadati**: Le etichette e le informazioni sui pazienti devono essere in un file Excel (`.xlsx`) con colonne specifiche come `ID`, `Age`, `Sex`, e `Class`. L'ID deve corrispondere al prefisso del nome dei file audio.

    | ID    | Age | Sex | Class |
    |-------|-----|-----|-------|
    | ID000 | 80  | M   | 5     |
    | ID001 | 61  | F   | 2     |

### 3. Configurazione dell'Esperimento

Tutti gli aspetti dell'esperimento sono controllati dal file `RNN/hparams.yaml`. Prima di lanciare il training, assicurati di configurare correttamente:

-   **Percorsi dei dati**:
    -   `data_folder`: Il percorso della cartella principale contenente i file audio (es. `dati/`).
    -   `excel_file`: Il percorso del tuo file Excel.
-   **Tipo di Task**:
    -   `task_type`: Impostalo su `'classification'` o `'regression'`.
-   **Iperparametri**:
    -   Modifica i parametri di preprocessing, feature extraction, modello e training secondo le tue necessità. Ogni parametro è commentato per chiarezza.

### 4. Lanciare il Training

Una volta completata la configurazione, puoi lanciare il training con il seguente comando dalla cartella radice:

```bash
python RNN/train.py RNN/hparams.yaml --device="cuda:0"
```

-   `--device`: Specifica il dispositivo su cui eseguire il training (es. `"cuda:0"` per la prima GPU, `"cpu"` per la CPU).

Lo script si occuperà automaticamente di:
1.  Leggere i tuoi dati e creare un file `data.json`.
2.  Inizializzare il dataset, il modello e l'ottimizzatore.
3.  Lanciare il ciclo di training e validazione.
4.  Salvare i modelli migliori e i log nella cartella specificata in `output_folder`.
