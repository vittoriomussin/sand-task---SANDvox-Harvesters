import os
import pandas as pd
import argparse
from sklearn.model_selection import GroupShuffleSplit

def prepare_data(metadata_file, data_folder, output_folder):
    """
    Scans a data folder for .wav files, merges them with metadata,
    and splits the data into train, validation, and test sets based on patient ID.

    Arguments:
        metadata_file (str): Path to the metadata CSV file.
        data_folder (str): Path to the root folder containing audio subfolders.
        output_folder (str): Path to the folder where the output CSVs will be saved.
    """
    # 1. Load metadata
    try:
        metadata_df = pd.read_csv(metadata_file)
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {metadata_file}")
        return

    print(f"Loaded metadata with {len(metadata_df)} entries.")

    # 2. Find all .wav files and extract IDs
    audio_files = []
    for root, _, files in os.walk(data_folder):
        for file in files:
            if file.endswith(".wav"):
                wav_path = os.path.join(root, file)
                # Extract ID from filename (e.g., "ID000_phonationA.wav" -> "ID000")
                patient_id = file.split('_')[0]
                audio_files.append({"ID": patient_id, "wav_path": wav_path})

    if not audio_files:
        print(f"Error: No .wav files found in {data_folder}")
        return

    audio_df = pd.DataFrame(audio_files)
    print(f"Found {len(audio_df)} audio files.")

    # 3. Merge audio files with metadata
    # Using a right merge to keep only audio files that have corresponding metadata
    full_df = pd.merge(metadata_df, audio_df, on="ID", how="right")

    # Check for missing metadata
    missing_metadata_count = full_df['Age'].isnull().sum()
    if missing_metadata_count > 0:
        print(f"Warning: {missing_metadata_count} audio files have no corresponding metadata and will be dropped.")
        full_df.dropna(subset=['Age'], inplace=True) # Assuming 'Age' is a mandatory column

    print(f"Created a full dataset with {len(full_df)} samples after merging.")

    # 4. Split data into train, validation, and test sets
    # Ensure all samples from a single patient (group) are in the same split.
    splitter = GroupShuffleSplit(test_size=0.3, n_splits=1, random_state=42)

    # First split: train vs. (validation + test)
    train_indices, temp_indices = next(splitter.split(full_df, groups=full_df['ID']))
    train_df = full_df.iloc[train_indices]
    temp_df = full_df.iloc[temp_indices]

    # Second split: validation vs. test
    # Adjust test_size for the second split (e.g., 0.5 means 50% of temp_df goes to test)
    temp_splitter = GroupShuffleSplit(test_size=0.5, n_splits=1, random_state=42)
    val_indices, test_indices = next(temp_splitter.split(temp_df, groups=temp_df['ID']))
    val_df = temp_df.iloc[val_indices]
    test_df = temp_df.iloc[test_indices]

    print(f"Data split as follows:")
    print(f"- Training set: {len(train_df)} samples ({train_df['ID'].nunique()} patients)")
    print(f"- Validation set: {len(val_df)} samples ({val_df['ID'].nunique()} patients)")
    print(f"- Test set: {len(test_df)} samples ({test_df['ID'].nunique()} patients)")

    # 5. Save the manifest files
    os.makedirs(output_folder, exist_ok=True)
    train_df.to_csv(os.path.join(output_folder, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_folder, "valid.csv"), index=False)
    test_df.to_csv(os.path.join(output_folder, "test.csv"), index=False)

    print(f"Manifest files saved successfully in {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare audio data manifests for training."
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        required=True,
        help="Path to the metadata CSV file (e.g., 'data/metadata.csv')."
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        required=True,
        help="Path to the root folder containing audio data."
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="manifests",
        help="Folder where the output CSV manifest files will be saved."
    )

    args = parser.parse_args()

    # Create the output folder inside the RNN directory
    output_path = os.path.join("RNN", args.output_folder)

    prepare_data(args.metadata_file, args.data_folder, output_path)
