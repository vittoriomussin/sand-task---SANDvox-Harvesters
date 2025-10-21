import torch
from torch.utils.data import Dataset
import pandas as pd
from RNN.speech_processor import SpeechProcessor

class AudioDataset(Dataset):
    def __init__(self, manifest_file, speech_processor, target_column, task_type='classification', augment=True):
        """
        Initializes the dataset.

        Arguments:
            manifest_file (str): Path to the manifest CSV file (e.g., 'train.csv').
            speech_processor (SpeechProcessor): An instance of the SpeechProcessor class.
            target_column (str): The name of the column in the manifest to be used as the label.
            task_type (str): The type of task, either 'classification' or 'regression'.
            augment (bool): Whether to apply data augmentation.
        """
        self.manifest = pd.read_csv(manifest_file)
        self.speech_processor = speech_processor
        self.target_column = target_column
        self.task_type = task_type
        self.augment = augment

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.manifest)

    def __getitem__(self, idx):
        """
        Retrieves the features and label for a given index.

        Arguments:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the feature tensor and the label.
        """
        # Get audio path and label from the manifest
        sample = self.manifest.iloc[idx]
        wav_path = sample['wav_path']
        label = sample[self.target_column]

        # Process the audio file to get features
        features = self.speech_processor.process_audio(wav_path, augment=self.augment)

        # Convert label to the appropriate tensor type
        if self.task_type == 'classification':
            # Ensure label is an integer for classification
            label = torch.tensor(int(label), dtype=torch.long)
        else: # regression
            # Ensure label is a float for regression
            label = torch.tensor(float(label), dtype=torch.float32)

        return features, label

def pad_collate(batch):
    """
    Pads sequences to the length of the longest sequence in a batch.
    """
    # Separate features and labels
    features = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # Pad sequences
    features_padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0)

    # Stack labels
    labels = torch.stack(labels)

    return features_padded, labels

if __name__ == '__main__':
    # This is a placeholder for a functional test.
    # To run this, you would need:
    # 1. A 'hyperparameters.json' file.
    # 2. A 'train.csv' manifest file with 'wav_path' and a target column.
    # 3. The audio files referenced in the manifest.

    print("Dataset module created successfully.")
    print("To test, you need to create dummy data and run a script similar to this:")

    print("""
# --- Example Test Script ---
# from RNN.speech_processor import SpeechProcessor
# from RNN.dataset import AudioDataset, pad_collate
# from torch.utils.data import DataLoader

# # 1. Initialize the processor
# processor = SpeechProcessor('RNN/hyperparameters.json')

# # 2. Create a dummy manifest file (e.g., 'dummy_train.csv')
# # This file should have columns 'wav_path' and 'Class' or 'Age'
# # And point to a real (or dummy) .wav file.

# # 3. Initialize the Dataset
# # For classification:
# # train_dataset = AudioDataset(
# #     manifest_file='dummy_train.csv',
# #     speech_processor=processor,
# #     target_column='Class',
# #     task_type='classification'
# # )
# # For regression:
# # train_dataset = AudioDataset(
# #     manifest_file='dummy_train.csv',
# #     speech_processor=processor,
# #     target_column='Age',
# #     task_type='regression'
# # )

# # 4. Initialize the DataLoader
# # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# # 5. Iterate over a batch
# # features, labels = next(iter(train_loader))
# # print("Batch of features shape:", features.shape)
# # print("Batch of labels:", labels)
# # print("Test successful!")
    """)
