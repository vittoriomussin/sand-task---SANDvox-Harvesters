import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, rnn_type='LSTM', task_type='classification'):
        """
        Initializes the RNN model.

        Arguments:
            input_size (int): The number of input features.
            hidden_size (int): The number of features in the hidden state h.
            num_layers (int): Number of recurrent layers.
            output_size (int): The number of output units.
            rnn_type (str): The type of RNN layer to use ('LSTM', 'GRU', 'RNN').
            task_type (str): The type of task, 'classification' or 'regression'.
        """
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.task_type = task_type
        self.rnn_type = rnn_type

        rnn_class = getattr(nn, rnn_type)
        self.rnn = rnn_class(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Arguments:
            x (torch.Tensor): The input tensor of shape (batch, seq_len, features).

        Returns:
            torch.Tensor: The model output.
        """
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # LSTM has a cell state, GRU and RNN do not
        if self.rnn_type == 'LSTM':
            c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.rnn(x, (h0, c0))
        else:
            out, _ = self.rnn(x, h0)

        # We take the output of the last time step for classification/regression
        out = self.fc(out[:, -1, :])

        if self.task_type == 'regression':
            out = out.squeeze(-1)

        return out

if __name__ == '__main__':
    # --- Example Usage ---

    INPUT_SIZE = 5
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    SEQ_LENGTH = 10 # Example sequence length

    for rnn_type in ['LSTM', 'GRU', 'RNN']:
        print(f"--- Testing {rnn_type} ---")

        # --- Classification Example ---
        NUM_CLASSES = 10
        model_class = SimpleRNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES, rnn_type=rnn_type, task_type='classification')

        dummy_input_class = torch.randn(4, SEQ_LENGTH, INPUT_SIZE)
        output_class = model_class(dummy_input_class)

        print("  Classification Task:")
        print(f"    Input shape: {dummy_input_class.shape}")
        print(f"    Output shape: {output_class.shape}")
        assert output_class.shape == (4, NUM_CLASSES)
        print("    Classification model test successful!")

        # --- Regression Example ---
        OUTPUT_SIZE_REG = 1
        model_reg = SimpleRNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE_REG, rnn_type=rnn_type, task_type='regression')

        dummy_input_reg = torch.randn(4, SEQ_LENGTH, INPUT_SIZE)
        output_reg = model_reg(dummy_input_reg)

        print("  Regression Task:")
        print(f"    Input shape: {dummy_input_reg.shape}")
        print(f"    Output shape: {output_reg.shape}")
        assert output_reg.shape == (4,)
        print("    Regression model test successful!\n")
