# RNN/model.py
import torch
import torch.nn as nn

class PathoVoiceRNN(nn.Module):
    """
    Un modello RNN per la classificazione o regressione di segnali vocali patologici.
    L'architettura legge i parametri da un file di configurazione per massima flessibilità.
    """
    def __init__(
        self,
        rnn_type,
        input_size,
        hidden_size,
        num_layers,
        bidirectional,
        rnn_dropout,
        task_type,
        num_classes=None,
    ):
        super().__init__()
        self.task_type = task_type

        # Seleziona il tipo di RNN (LSTM o GRU)
        rnn_class = getattr(nn, rnn_type.upper())

        # 1. Strato RNN
        self.rnn = rnn_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=rnn_dropout if num_layers > 1 else 0,
            batch_first=True  # Importante: (Batch, Time, Features)
        )

        # Calcola la dimensione dell'output della RNN
        rnn_output_size = hidden_size * (2 if bidirectional else 1)

        # 2. Strato di Dropout (opzionale)
        self.dropout = nn.Dropout(rnn_dropout)

        # 3. Strato di output (flessibile per classificazione/regressione)
        if self.task_type == 'classification':
            if num_classes is None:
                raise ValueError("`num_classes` deve essere specificato per la classificazione.")
            output_size = num_classes
        elif self.task_type == 'regression':
            output_size = 1
        else:
            raise ValueError("`task_type` deve essere 'classification' o 'regression'")

        self.output_layer = nn.Linear(rnn_output_size, output_size)

    def forward(self, features, wav_lens=None):
        """
        Args:
            features (torch.Tensor): Il tensore di feature in input (Batch, Time, Features).
            wav_lens (torch.Tensor): Non usato direttamente qui, ma mantenuto per compatibilità.

        Returns:
            torch.Tensor: L'output del modello.
                          - Per classificazione: (Batch, NumClasses) - logits
                          - Per regressione: (Batch, 1) - valore predetto
        """
        output, _ = self.rnn(features)
        last_output = output[:, -1, :]
        last_output_dropout = self.dropout(last_output)
        final_output = self.output_layer(last_output_dropout)

        if self.task_type == 'regression':
            return final_output.view(-1, 1)
        else:
            return final_output
