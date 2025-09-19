[Link to the original form](https://docs.google.com/forms/d/e/1FAIpQLScZf58XZCZqFyKX8zwqslpYA3OD0OpQ8PJ9HOXX_nGUuT7U6w/viewform?usp=dialog)

# Approach to Tasks

**Section 1 of 5**

## Approach to Tasks

This form serves as a survey to understand the methodological approach to various tasks. It is a draft, so feel free to add content or discuss the choices.

*\*In an ideal world, every single option would be tested. However, this may not be feasible in terms of time/available computational resources. For this reason, there is also the possibility of selecting more than one option to indicate that we might want to explore both choices during the model selection phase.*

---

**Section 2 of 5**

## Output Representation: Training Task Decision

- [ ] **Option 1: Multiclass Classification**
  **Description:** Treat severity levels (e.g., Mild, Medium, Severe) as distinct categories. The model's goal will be to guess the exact category. The output is a one-hot vector (e.g., [0, 0, 1] for "Severe"), and the final layer uses a softmax activation function.
  **Why choose it:** Good for distinguishing between categories if the boundaries are clear.

- [ ] **Option 2: Discrete Regression**
  **Description:** Treat severity levels as ordered numerical values (e.g., 0, 1, 2, 3, 4). The model's goal is to predict a scalar value that reflects the severity. The final layer has a linear activation.
  **Why choose it:** Preferable for capturing the order and progression of the disease, penalizing more clinically significant errors more heavily.

---

**Section 3 of 5**

## Model Selection Strategy

- [ ] **Option 1: K-Fold Cross-Validation**
  **Description:** This is one of the most common and robust strategies. It consists of dividing the entire training dataset into K subsets (or "folds"). The model is trained K times, and each time a different fold is used as a validation set, while training on the other K-1 folds. The final performance is the average of the results obtained on all K folds.
  **Advantages:**
    - **Stability:** Provides a much more reliable estimate of the model's performance on unseen data compared to a single training/validation split.
    - **Data Utilization:** Allows the entire dataset to be used for both training and validation, reducing the risk of "wasting" valuable data.
  **Disadvantages:**
    - **Computational Cost:** Requires significantly more computation time, as the model is trained K times.

- [ ] **Option 2: Training-Validation-Test**
  **Description:** This strategy involves splitting the dataset into three parts:
    - **Training Set (e.g., 70%):** Used to train the model.
    - **Validation Set (e.g., 15%):** Used to optimize hyperparameters (e.g., learning rate, number of epochs) and choose the best model from the different trials.
    - **Test Set (e.g., 15%):** Used for the final evaluation of the chosen model's performance. It is crucial that this set is never used during training or hyperparameter optimization.
  **Advantages:**
    - **Simplicity:** It is simpler to implement and requires less computation time than cross-validation.
    - **Final Verification:** The "secret" test set provides an unbiased estimate of the model's performance in the real world.
  **Disadvantages:**
    - **Variance:** Performance can depend heavily on how the random data split occurred, especially if the dataset is small.

- [ ] **Other:**

---

**Section 4 of 5**

## Signal Analysis Methodologies

This section focuses on how we transform the raw audio signal into a representation that a machine learning model can process.


- [ ] **Option 1: The Signal as an Image (Spectrogram Analysis)**
  **1. Description**
  The central idea is to convert the one-dimensional audio signal (time-amplitude) into a two-dimensional time-frequency representation, i.e., a spectrogram. Typically, the Mel spectrogram is used, which adopts a non-linear frequency scale to mimic human auditory perception. Once this "image" is obtained, the audio classification problem is reformulated as an image classification problem, allowing us to use powerful Computer Vision models (e.g., CNNs, Vision Transformers).
  **2. Advantages**
    - **Leverages Transfer Learning:** We can use complex architectures (e.g., ResNet, EfficientNet, ViT) pre-trained on huge image datasets (like ImageNet), adapting them to our task with relatively fast fine-tuning.
    - **Automatic Feature Learning:** The model autonomously learns to identify relevant visual patterns in the spectrogram (e.g., shape of formants, presence of noise, interruptions, instability), eliminating the need for manual feature extraction.
    - **Preserves Temporal Information:** Unlike feature aggregation, the spectrogram maintains the temporal structure of the signal, allowing the model to analyze how frequencies evolve over time.
    - **Access to Data Augmentation Techniques:** Spectrogram-specific augmentation techniques (e.g., SpecAugment, which masks blocks of time and frequency) can be applied to make the model more robust.
  **3. Disadvantages**
    - **Computational Cost:** Spectrograms are large inputs. Training deep Computer Vision models requires significant computational resources (GPUs with a lot of VRAM).
    - **Loss of Phase Information:** The conversion to a spectrogram usually only considers the magnitude of the frequencies, discarding phase information, which, although often less critical, could contain useful signals.
    - **Lower Interpretability:** It is more difficult to directly correlate a "visual pattern" learned by a CNN to a specific phonetic-acoustic concept (e.g., jitter), making the model a "black box."


- [ ] **Option 2: Acoustic-Phonetic Feature Engineering**
  **1. Description**
  This is the "classic" approach, based on decades of research in phonetics and speech pathology. It consists of manually extracting a predefined set of numerical features from the audio signal, which are known to be correlated with voice quality and dysarthria. These features (e.g., jitter, shimmer, MFCCs, formants) are then aggregated (e.g., via mean and standard deviation) to create a single fixed-length vector for each subject, which is then used to train traditional machine learning models.
  **2. Advantages**
    - **Maximum Interpretability:** Each feature has a precise physical and clinical meaning. We can analyze the importance of features (e.g., with SHAP) to understand why the model makes a certain decision, providing valuable insights for the final paper.
    - **Computational Efficiency:** The input is a very small and compact vector. Training models like XGBoost or Random Forest is extremely fast and requires few resources.
    - **Robustness on Limited Data:** With a good set of features, this approach can achieve excellent performance even with small datasets, as domain knowledge is already "injected" into the process.
    - **Scientific Foundation:** The features are based on biomarkers validated by scientific literature.
  **3. Disadvantages**
    - **Requires Domain Knowledge:** Selecting the optimal feature set is an art and requires deep expertise in the field of voice signal analysis.
    - **Loss of Dynamic Information:** Aggregating features through statistics (mean, std) over entire audio files completely destroys information about the temporal evolution of characteristics within a single vocal utterance. (* see next section)
    - **Potentially "Fragile" Process:** The extraction of some features (e.g., F0, formants) can be sensitive to noise or fail on highly compromised voices, requiring careful pre-processing and cleaning.


- [ ] **Option 3: End-to-End with Pre-trained Speech Models**
  **1. Description**
  This approach leverages large Transformer models (e.g., Wav2Vec2, HuBERT, WavLM) that have been pre-trained on thousands of hours of unlabeled speech. These models learn to convert the raw waveform into rich and contextualized latent representations. For our task, we load a pre-trained model and "fine-tune" it on our specific dataset, adding only a small classification layer on top.
  **2. Advantages**
    - **State-of-the-Art Performance:** Currently, this approach tends to achieve the best performance in a wide range of speech-related tasks, including paralinguistics.
    - **Robust Representations:** Pre-training on very heterogeneous data makes the learned features robust to variations in speaker, background noise, and recording conditions.
    - **No Manual Feature Engineering:** The model learns the relevant features directly from the data, preserving all dynamic and contextual information.
    - **Flexibility:** The model can be adapted to different tasks with few modifications.
  **3. Disadvantages**
    - **Extremely High Computational Cost:** Fine-tuning these models is very demanding in terms of GPU and time.
    - **Minimal Interpretability:** The latent representations are high-dimensional vectors in an abstract space. It is almost impossible to understand their direct acoustic meaning.
    - **Risk of Overfitting:** Although pre-training helps, on a small dataset like the one in the challenge, there is still a risk that the model will overfit, especially on the less represented classes.
- [ ] **Other:**

---

**Section 5 of 5**

## Model Architectures

This section focuses on what type of model to use to process the representations obtained from the previous Section, grouping the architectures based on their ability to handle static or dynamic inputs.


- [ ] **Option 1: Models for Static Input**
  These models are designed to operate on inputs that do not have an explicit temporal dimension. The ideal input for them is a fixed-length vector, where the entire information of a subject is compressed into a single static "profile."
  **1. Description**
  This category includes both simple neural networks (MLPs) and more traditional machine learning models.
    - **Classic Machine Learning Models (XGBoost, SVM, Random Forest):** Algorithms that excel in analyzing tabular data. They learn to find complex patterns and relationships between engineered features to make a prediction.
    - **Simple Neural Networks (Multi-Layer Perceptron - MLP):** These are fully-connected neural networks composed of a series of linear layers and activation functions. They also take the flat vector as input and learn a non-linear mapping to the output classes.
  **2. Advantages**
    - **Extreme Efficiency:** They are very fast to train and require minimal computational resources.
    - **High Performance on Tabular Data:** When the features are well-defined and informative, these models are hard to beat and often represent a very strong baseline.
    - **Interpretability (especially for non-neural models):** Tools like SHAP for XGBoost or the analysis of feature importance in a Random Forest allow us to understand which acoustic characteristics are most decisive for classification.
    - **Robustness on Small Datasets:** They require less data to converge compared to deep architectures, reducing the risk of overfitting.
  **3. Disadvantages**
    - **Total Dependence on Feature Engineering:** Their performance is solely determined by the quality of the features provided as input. They cannot extract new information from raw data.
    - **Blindness to the Temporal Dimension:** Lacking memory or sequential structure, they cannot in any way model the dynamics and temporal dependencies of the vocal signal. All dynamic information must be pre-encoded in the static features (e.g., F0_std).


- [ ] **Option 2: Recurrent Models for Dynamic Input**
  These architectures were created specifically to process data sequences, where order is fundamental. The ideal input is a sequence of frame-vectors (e.g., a matrix (number_of_frames, number_of_features_per_frame)), which preserves the temporal evolution of the signal.
  **1. Description**
  **Recurrent Neural Networks (RNN, LSTM, GRU):** Their distinctive feature is an internal state (memory) that is passed from one time step to the next. At each frame of the sequence, the network updates its memory based on the current input and the previous state, allowing it to "remember" the past to contextualize the present. More advanced variants like LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) use "gate" mechanisms to better manage the flow of information and learn long-range dependencies.
  **2. Advantages**
    - **Natural Time Modeling:** They are the quintessential architecture for capturing evolution, rhythm, and temporal dependencies in a sequence.
    - **Handling of Variable-Length Inputs:** They can natively process sequences of different lengths, which is perfect for audio files of different durations.
    - **Dynamic Feature Learning:** They learn to recognize dynamic patterns (e.g., the transition from a vowel to a consonant, instability over time) that are impossible to capture with a static input.
  **3. Disadvantages**
    - **Sequential and Slow Training:** The calculation of the state at time t requires that the state at time t-1 has already been calculated. This sequential nature prevents efficient parallelization along the time dimension, making training slower than other architectures.
    - **Difficulty with Very Long Dependencies:** Although LSTM and GRU are effective, they can still struggle to correlate events that occur hundreds or thousands of time steps apart.
    - **Potentially Less Powerful than Transformers:** For many tasks, they have been surpassed in performance by Transformer-based models.


- [ ] **Option 3: Transformer-based Models**
  Transformers represent a different paradigm for data processing, applicable to both dynamic inputs (sequences) and image-like inputs (spectrograms). They have revolutionized the field thanks to their self-attention mechanism.
  **1. Description**
  Unlike RNNs that process data in order, Transformers process the entire sequence (or the entire partitioned image) in parallel. The self-attention mechanism allows each element of the input to "look at" and "weigh" the importance of all other elements to build its own contextualized representation. This architecture is the foundation of:
    - **Vision Transformers (ViT):** Applied to spectrograms (image-like input).
    - **End-to-End Models (Wav2Vec2, HuBERT):** Applied directly to the raw waveform or a sequence of features.
  **2. Advantages**
    - **Capture of Global Dependencies:** Self-attention has no difficulty modeling relationships between very distant elements in the input, overcoming the main limitation of RNNs.
    - **Maximum Parallelization:** Their non-sequential nature makes them extremely efficient to train on modern hardware (GPUs/TPUs).
    - **State-of-the-Art Performance:** They are the basis of the best-performing models available today for almost all domains (text, audio, images).
    - **Huge Potential for Transfer Learning:** Their scalability has allowed the creation of foundation models pre-trained on vast amounts of data, which can be adapted to our task with effective fine-tuning.
  **3. Disadvantages**
    - **Greedy for Data and Resources:** To be trained from scratch, they require huge datasets and considerable computing power. For this reason, their use in our context is almost exclusively limited to fine-tuning pre-existing models.
    - **Quadratic Complexity:** The computational and memory cost grows with the square of the sequence length, which can be a problem for very long audio signals without the use of more efficient variants of the architecture.
    - **Lower Interpretability:** The attention weights can give some clues, but in general, they are very complex and difficult to interpret models.
- [ ] **Other:**