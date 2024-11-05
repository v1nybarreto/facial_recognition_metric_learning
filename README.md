
# Facial Recognition Task

## Project Structure
The goal of this project is to develop a facial recognition model using **Metric Learning** to handle both masked and unmasked faces, overcoming traditional model limitations.

## Requirements
- Python 3.7+
- TensorFlow
- NumPy
- OpenCV
- MediaPipe
- Matplotlib
- SciKit-Learn

Install requirements using:
```bash
pip install -r requirements.txt
```

## Steps
1. **Train the Neural Network**: 
   - Train a neural network on a dataset of celebrity faces, leveraging metric learning to create feature vectors.
   
2. **Descriptor Database Creation**:
   - Generate a database of facial feature vectors based on the trained model.
   
3. **Add New Person**:
   - Add a new person's unmasked image to the database and update the feature vector collection.
   
4. **Masked Face Recognition**:
   - Use a masked version of the person's image to test the recognition model by comparing it to existing descriptors.

## Usage
To execute the project:
1. Train the model and save it:
   ```python
   model = build_model(input_shape=(224, 224, 3), num_classes=len(class_names))
   history = train_model(model, train_ds, val_ds, epochs=20)
   save_model(model)
   ```
2. Extract and save descriptors:
   ```python
   extract_and_save_descriptors(model, base_dir)
   ```
3. Add a new person to the database:
   ```python
   add_person_to_database(model, 'path/to/image.png', "Person Name")
   ```
4. Perform facial recognition with a masked image:
   ```python
   recognize_person(model, 'path/to/masked_image.png')
   ```

## Results
The model is expected to recognize faces even when masked, with similarity scores for verification.

## Directory Structure
```plaintext
Facial Recognition/
├── data/               # Training dataset
├── mask/               # Mask image for augmentation
├── metric_learning/    # PDF instructions
├── person/
│   ├── person_masked/  # Masked photo of the person
│   └── person_no_mask/ # Unmasked photo of the person
├── source/             # Main code
└── README.md           # Project documentation
```
