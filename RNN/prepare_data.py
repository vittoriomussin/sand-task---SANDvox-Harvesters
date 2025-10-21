# RNN/prepare_data.py
import os
import json
import pandas as pd
import glob
import logging

# Configurazione del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_data(data_folder, excel_file, output_json_file):
    """
    Scansiona le cartelle audio e legge il file Excel per creare un file JSON
    nel formato richiesto da SpeechBrain.

    Args:
        data_folder (str): Il percorso della cartella radice contenente le
                           sottocartelle con i file audio (es. 'phonationA').
        excel_file (str): Il percorso del file Excel con i metadati.
        output_json_file (str): Il percorso del file JSON da generare.
    """
    try:
        # 1. Carica i metadati dal file Excel e crea un dizionario di lookup
        logging.info(f"Lettura del file Excel: {excel_file}")
        metadata_df = pd.read_excel(excel_file)
        # Usiamo l'ID come indice per una ricerca veloce
        metadata_dict = metadata_df.set_index('ID').to_dict('index')
        logging.info(f"Trovati {len(metadata_dict)} record nel file Excel.")

        # 2. Scansiona le cartelle per trovare tutti i file .wav
        logging.info(f"Scansione della cartella dati: {data_folder}")
        # Cerca i file .wav in tutte le sottodirectory
        wav_files = glob.glob(os.path.join(data_folder, '**', '*.wav'), recursive=True)
        if not wav_files:
            logging.warning("Nessun file .wav trovato. Controlla il percorso di `data_folder`.")
            return

        logging.info(f"Trovati {len(wav_files)} file .wav.")

        # 3. Crea il dizionario JSON finale
        data_json = {}
        for wav_path in wav_files:
            # Normalizza il percorso per essere cross-platform
            wav_path = os.path.normpath(wav_path)
            filename = os.path.basename(wav_path)

            # Estrai l'ID del paziente (es. 'ID000' da 'ID000_phonationA.wav')
            patient_id = filename.split('_')[0]

            if patient_id in metadata_dict:
                # Crea un ID unico per ogni registrazione (es. 'ID000_phonationA')
                entry_id = os.path.splitext(filename)[0]

                # Prendi i metadati dal dizionario
                metadata = metadata_dict[patient_id]

                data_json[entry_id] = {
                    "wav_path": wav_path,
                    "age": metadata['Age'],
                    "sex": metadata['Sex'],
                    "class_label": metadata['Class']
                }
            else:
                logging.warning(f"ID paziente '{patient_id}' dal file '{filename}' non trovato nel file Excel. Il file verrà saltato.")

        # 4. Scrivi il file JSON
        logging.info(f"Creazione del file JSON di output: {output_json_file}")
        with open(output_json_file, 'w') as f:
            json.dump(data_json, f, indent=4)

        logging.info("Preparazione dei dati completata con successo!")
        logging.info(f"Creati {len(data_json)} record nel file JSON.")

    except FileNotFoundError:
        logging.error(f"Errore: Uno dei file non è stato trovato. Controlla i percorsi:\n- Cartella Dati: {data_folder}\n- File Excel: {excel_file}")
    except Exception as e:
        logging.error(f"Si è verificato un errore inaspettato: {e}")


if __name__ == "__main__":
    # Esempio di come eseguire lo script.
    # Questi percorsi dovrebbero essere letti da hparams.yaml in un contesto reale.
    # Per ora, li mettiamo qui come esempio.
    # NOTA: Assicurati di creare un file Excel e delle cartelle audio dummy per testare.

    # --- Configurazione Esempio ---
    # Crea delle cartelle e file finti per il test
    if not os.path.exists("sample_data/phonationA"):
        os.makedirs("sample_data/phonationA")
        with open("sample_data/phonationA/ID000_phonationA.wav", "w") as f: f.write("dummy wav")
        with open("sample_data/phonationA/ID001_phonationA.wav", "w") as f: f.write("dummy wav")

    if not os.path.exists("metadata.xlsx"):
        df = pd.DataFrame({
            "ID": ["ID000", "ID001"],
            "Age": [80, 61],
            "Sex": ["M", "F"],
            "Class": [5, 2]
        })
        df.to_excel("metadata.xlsx", index=False)
    # -----------------------------

    HPARAMS_FILE = 'RNN/hparams.yaml'
    # In un vero script di training, caricheremmo hparams.
    # Qui leggiamo direttamente i valori necessari per questo script.
    # Questo è un placeholder, in pratica useremo un file .yaml
    class Hparams:
        data_folder = 'sample_data'
        excel_file = 'metadata.xlsx'
        json_data_file = 'RNN/data.json'

    hparams = Hparams()

    prepare_data(
        data_folder=hparams.data_folder,
        excel_file=hparams.excel_file,
        output_json_file=hparams.json_data_file
    )
