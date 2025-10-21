import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

from RNN.speech_processor import SpeechProcessor
from RNN.dataset import AudioDataset, pad_collate
from RNN.model import SimpleRNN

# --- Configuration ---
# Model parameters - could be moved to a config file or args
INPUT_SIZE = 5  # f0, jitter, shimmer, hnr, gne
HIDDEN_SIZE = 128
NUM_LAYERS = 2
LEARNING_RATE = 0.001
BATCH_SIZE = 16
NUM_EPOCHS = 10 # For demonstration; should be higher for real training

def train(model, train_loader, criterion, optimizer, device):
    """Handles the training loop for one epoch."""
    model.train()
    total_loss = 0

    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)

        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def evaluate(model, data_loader, criterion, task_type, device):
    """Handles the evaluation loop."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            if task_type == 'classification':
                preds = torch.argmax(outputs, dim=1)
            else: # regression
                preds = outputs

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)

    # Calculate metrics
    if task_type == 'classification':
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        return avg_loss, accuracy, f1
    else: # regression
        mae = mean_absolute_error(all_labels, all_preds)
        return avg_loss, mae


def main(args):
    """Main function to orchestrate the training and evaluation process."""

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize speech processor
    processor = SpeechProcessor(args.hyperparams_file)

    # --- Data Loading ---
    train_dataset = AudioDataset(
        manifest_file=args.train_manifest,
        speech_processor=processor,
        target_column=args.target_column,
        task_type=args.task_type,
        augment=True
    )
    valid_dataset = AudioDataset(
        manifest_file=args.valid_manifest,
        speech_processor=processor,
        target_column=args.target_column,
        task_type=args.task_type,
        augment=False
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate)

    # --- Model, Criterion, Optimizer ---
    if args.task_type == 'classification':
        # Determine number of classes from the training data
        num_classes = pd.read_csv(args.train_manifest)[args.target_column].nunique()
        output_size = num_classes
        criterion = nn.CrossEntropyLoss()
        best_metric = 0.0 # We want to maximize accuracy/F1
    else: # regression
        output_size = 1
        criterion = nn.MSELoss()
        best_metric = float('inf') # We want to minimize loss/MAE

    model = SimpleRNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, output_size, args.task_type).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Starting training for {args.task_type} on target '{args.target_column}'...")

    # --- Training Loop ---
    for epoch in range(NUM_EPOCHS):
        train_loss = train(model, train_loader, criterion, optimizer, device)

        if args.task_type == 'classification':
            valid_loss, valid_acc, valid_f1 = evaluate(model, valid_loader, criterion, args.task_type, device)
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}, Valid F1: {valid_f1:.4f}")

            # Save the best model based on F1 score
            if valid_f1 > best_metric:
                best_metric = valid_f1
                torch.save(model.state_dict(), os.path.join(args.save_path, "best_model.pth"))
                print("New best model saved.")
        else: # regression
            valid_loss, valid_mae = evaluate(model, valid_loader, criterion, args.task_type, device)
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Valid MAE: {valid_mae:.4f}")

            # Save the best model based on MAE
            if valid_mae < best_metric:
                best_metric = valid_mae
                torch.save(model.state_dict(), os.path.join(args.save_path, "best_model.pth"))
                print("New best model saved.")

    print("Training finished.")

    # --- Final Evaluation on Test Set ---
    # (Optional, but good practice)
    # test_dataset = ... test_loader = ...
    # load best_model.pth
    # test_results = evaluate(...)
    # print(f"Final test results: {test_results}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a simple RNN for audio classification or regression.")

    parser.add_argument('--hyperparams_file', type=str, required=True, help="Path to hyperparameters.json")
    parser.add_argument('--train_manifest', type=str, required=True, help="Path to the training manifest (train.csv)")
    parser.add_argument('--valid_manifest', type=str, required=True, help="Path to the validation manifest (valid.csv)")
    parser.add_argument('--task_type', type=str, choices=['classification', 'regression'], required=True, help="Type of task")
    parser.add_argument('--target_column', type=str, required=True, help="Name of the target column in the manifests")
    parser.add_argument('--save_path', type=str, default='RNN/saved_models', help="Directory to save the best model")

    args = parser.parse_args()

    # Create save directory if it doesn't exist
    os.makedirs(args.save_path, exist_ok=True)

    main(args)
