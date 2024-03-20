# Supply-Chain-Sustainability-Classifier

**Analytical Model for Text Data Analysis**

This repository contains the code and resources for an analytical model designed to analyze text data. The model goes through several steps, including data extraction, preprocessing, feature extraction, and model training, to perform various natural language processing (NLP) tasks.

### Files and Directories

- `my_trained_model.h5`: This file contains the trained machine learning model saved in the Hierarchical Data Format (HDF5) format. It can be loaded and used for making predictions on new text data.
  
- `label_encoder.pkl`: This file contains the label encoder used to encode the target labels during model training. It is necessary for decoding the predicted labels back to their original form.

- `trainmodel.ipynb`: This Jupyter Notebook file contains the code used to train the machine learning model. It includes steps for data preprocessing, feature extraction, model definition, training, and evaluation.

- `aicp_final-version.ipynb`: This Jupyter Notebook file contains the main code for running the entire pipeline, from data extraction to model evaluation. It includes all the necessary steps to preprocess the data, extract features, and make predictions using the trained model.

- `cleaned_texts.csv`: This CSV file contains the preprocessed text data after cleaning and preprocessing steps. It serves as the input for feature extraction and model training.

- `bert_features_matrix.npy`: This NumPy array file contains the BERT-encoded featuhttps://docs.github.com/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntaxres extracted from the preprocessed text data. It is used as input for training the machine learning model.

- `reduced_bert_features.npy`: This NumPy array file contains the BERT-encoded features reduced to lower dimensions using Principal Component Analysis (PCA). It can be used as an alternative input for training the machine learning model.

### Usage

1. **Training the Model**: To train the machine learning model, run the code in the `trainmodel.ipynb` notebook. Ensure that the `cleaned_texts.csv` file containing preprocessed text data is available in the directory.

2. **Generating Predictions**: Use the `aicp_final-version.ipynb` notebook to run the entire pipeline, including data preprocessing, feature extraction, model training, and prediction generation. Make sure to have the trained model (`my_trained_model.h5`) and label encoder (`label_encoder.pkl`) files available.

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

- Alan Liang(AICP Team Shculich)
- Sihan Liu(AICP Team Shculich)
- Cullen Jiang(AICP Team Shculich)
- N'guessan, Komenan(AICP Team Shculich)
  

### License

This project is free to modify and distribute the code for your own purposes.
