# RNN/train.py
import os
import sys
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.utils.distributed import run_on_main

# Importa i moduli che abbiamo creato
from prepare_data import prepare_data

# Definiamo la classe Brain, il cuore del training in SpeechBrain
class PathoBrain(sb.Brain):
    """
    Classe Brain per addestrare il modello RNN su dati vocali patologici.
    Gestisce la logica del forward pass, del calcolo della loss e del ciclo di training.
    """
    def compute_forward(self, batch, stage):
        """Calcola l'output del modello."""
        batch = batch.to(self.device)
        features, wav_lens = batch.features, None # wav_lens non necessario ma per compatibilità

        # L'output dipende dal task
        predictions = self.modules.model(features, wav_lens)
        return predictions

    def compute_objectives(self, predictions, batch, stage):
        """
        Calcola la loss. La funzione di costo dipende dal task (classificazione/regressione).
        """
        batch = batch.to(self.device)
        labels = batch.label_tensor

        if self.hparams.task_type == 'classification':
            # Cross-Entropy Loss per la classificazione
            loss = sb.nnet.losses.nll_loss(
                log_probabilities=torch.log_softmax(predictions, dim=-1),
                targets=labels.squeeze(1)
            )
            # Calcola l'accuracy come metrica aggiuntiva
            if stage != sb.Stage.TRAIN:
                self.acc_metric.append(predictions, labels)

        elif self.hparams.task_type == 'regression':
            # Mean Squared Error (MSE) o L1Loss per la regressione
            loss = torch.nn.functional.mse_loss(predictions, labels)
            if stage != sb.Stage.TRAIN:
                self.error_metric.append(predictions, labels)
        else:
            raise ValueError("`task_type` non valido.")

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Viene chiamato all'inizio di ogni stage (train, valid, test)."""
        if stage != sb.Stage.TRAIN:
            # Inizializza le metriche per validazione e test
            if self.hparams.task_type == 'classification':
                self.acc_metric = sb.nnet.accuracy.Accuracy()
            elif self.hparams.task_type == 'regression':
                self.error_metric = sb.nnet.losses.ErrorRate()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Viene chiamato alla fine di ogni stage."""
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        elif stage == sb.Stage.VALID:
            if self.hparams.task_type == 'classification':
                print(f"Epoch {epoch}: Valid Loss: {stage_loss:.2f}, Accuracy: {self.acc_metric.summarize():.2f}%")
            elif self.hparams.task_type == 'regression':
                 print(f"Epoch {epoch}: Valid Loss: {stage_loss:.2f}, Error Rate: {self.error_metric.summarize():.2f}%")
        elif stage == sb.Stage.TEST:
            if self.hparams.task_type == 'classification':
                print(f"Test Loss: {stage_loss:.2f}, Accuracy: {self.acc_metric.summarize():.2f}%")
            elif self.hparams.task_type == 'regression':
                print(f"Test Loss: {stage_loss:.2f}, Error Rate: {self.error_metric.summarize():.2f}%")

# --- Inizio dello Script Principale ---
if __name__ == "__main__":

    # 1. Carica gli iperparametri dal file YAML
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Crea la cartella di output
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # 2. Prepara i dati (esegue prepare_data.py)
    run_on_main(
        prepare_data,
        kwargs={
            "data_folder": hparams["data_folder"],
            "excel_file": hparams["excel_file"],
            "output_json_file": hparams["json_data_file"],
        },
    )

    # 3. Inizializza i componenti (dataset, modello, ecc.)
    from dataio import dataio_prepare
    train_data, valid_data, test_data = dataio_prepare(hparams)

    # Crea i dataloader
    train_dataloader = SaveableDataLoader(train_data, batch_size=hparams["batch_size"], shuffle=True)
    valid_dataloader = SaveableDataLoader(valid_data, batch_size=hparams["batch_size"])
    test_dataloader = SaveableDataLoader(test_data, batch_size=hparams["batch_size"])

    # Seleziona l'ottimizzatore
    optimizers = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
    }
    opt_class = optimizers[hparams["optimizer"]]

    # Inizializza Brain
    brain = PathoBrain(
        modules=hparams["modules"],
        opt_class=opt_class,
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"]
    )

    # 4. Addestra il modello
    print("Inizio dell'addestramento...")
    brain.fit(
        epoch_counter=brain.hparams.epoch_counter,
        train_set=train_dataloader,
        valid_set=valid_dataloader,
        min_keys=["loss"], # Criterio per salvare il miglior modello
    )

    # 5. Valuta sul test set
    print("Valutazione sul test set...")
    brain.evaluate(test_dataloader, min_key="loss")
