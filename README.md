# Supply-Chain-Sustainability-Classifier

**Analytical Model for Text Data Analysis**

This repository contains the code and resources for an analytical model designed to analyze text data. The model goes through several steps, including data extraction, preprocessing, feature extraction, and model training, to perform various natural language processing (NLP) tasks.

### Files and Directories

- `my_trained_model.h5`: Trained machine learning model saved in HDF5 format.
  
- `label_encoder.pkl`: Label encoder used for encoding target labels during model training.

- `trainmodel.ipynb`: Jupyter Notebook for training the machine learning model.

- `aicp_final-version.ipynb`: Jupyter Notebook for the entire pipeline, from data extraction to model evaluation.

- `download_files.ipynb`: Jupyter Notebook for automatically downloading files from a list of URLs.

- `cleaned_texts.csv`: Preprocessed text data after cleaning and preprocessing steps.

- `bert_features_matrix.npy`: BERT-encoded features extracted from preprocessed text data.

- `reduced_bert_features.npy`: BERT-encoded features reduced to lower dimensions using PCA.

### Usage

1. **Training the Model**: Run the code in `trainmodel.ipynb` to train the machine learning model using preprocessed text data.

2. **Generating Predictions**: Use `aicp_final-version.ipynb` to run the entire pipeline, including data preprocessing, feature extraction, model training, and prediction generation.

3. **Downloading Files**: Use `download_files.ipynb` to automatically download files from a list of URLs provided in an Excel file.

### Dependencies

Ensure you have the following dependencies installed to run the code:

- Python 3.x
- NumPy
- pandas
- transformers
- torch
- scikit-learn
- Keras

You can install the required Python packages using pip:

```
pip install numpy pandas transformers torch scikit-learn keras
```

### Contributors

- Alan Liang (AICP Team Schulich)
- Sihan Liu (AICP Team Schulich)
- Cullen Jiang (AICP Team Schulich)
- N'guessan, Komenan (AICP Team Schulich)

### License

This project is licensed under the MIT License. Feel free to modify and distribute the code for your own purposes.
