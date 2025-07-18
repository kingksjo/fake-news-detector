# Fake News Classifier using Natural Language Processing


This repository contains a machine learning project that classifies news articles as **Real** or **Fake**. The project leverages NLP techniques to analyze article content.

## Table of Contents
1. [Project Goal](#project-goal)
2. [Key Features](#key-features)
3. [Dataset](#dataset)
4. [Technologies Used](#technologies-used)
6. [How to Use](#how-to-use)
7. [Model Performance](#model-performance)
8. [Important Limitations](#important-limitations)
9. [Future Work](#future-work)

## Project Goal
The primary objective of this project is to build and evaluate a robust text classification model that can distinguish between real and fake news. We explore the predictive power of different parts of an article by training two distinct models: one using only headlines and an enhanced version using both headlines and the full article text.

## Key Features
- **Data Cleaning & EDA:** Includes handling of missing values and identification and removal of a critical data leak in the original dataset.
- **Advanced Text Pre-processing:** Utilizes **SpaCy** for efficient lemmatization, stop word removal, and punctuation handling.
- **Feature Engineering:** Implements **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert text into meaningful numerical vectors.
- **Dual-Model Approach:**
    - `Model v1`: A baseline classifier trained only on article headlines.
    - `Model v2`: An enhanced classifier trained on a combined feature set of headlines and full article text.
- **Interactive Prediction:** Provides easy-to-use functions to test the trained models on new, unseen articles.

## Dataset
The dataset used for this project is the "Real and Fake News" dataset available on [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset). It consists of two files, `True.csv` and `Fake.csv`, containing articles from various sources.

> **Note:** During EDA, the `subject` column was identified as a source of data leakage and was dropped to ensure the model learns from linguistic patterns rather than metadata shortcuts.

## Technologies Used
- **Core Libraries:**
    - `pandas`: For data manipulation and analysis.
    - `scikit-learn`: For TF-IDF vectorization, model training, and evaluation.
    - `spacy`: For high-performance NLP pre-processing.
    - `matplotlib` & `seaborn`: For data visualization.
    - `tqdm`: For progress bars during long-running operations.
- **Environment:** Jupyter Notebook / JupyterLab


5.  **Download the dataset:**
    Download the `True.csv` and `Fake.csv` files from the [Kaggle link](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) and place them in the root directory of the project.

## How to Use
1.  **Launch Jupyter:**
    ```bash
    jupyter lab
    ```
2.  **Open the notebook:**
    Open the `fk-news-classify.ipynb` file and run the cells sequentially.

3.  **Test with the interactive prediction functions:**
    At the end of the notebook, you can use the provided functions to test new articles.

    *Example for Model v2:*
    ```python
    title = "Your article title here"
    text = "The full body of the article text here..."

    predict_full_article(title, text)
    ```

## Model Performance

A comparison of the two models on the held-out test set highlights the significant value of analyzing the full article text.

| Model | Features Used | Test Accuracy |
| :--- | :--- | :--- |
| Model v1 | Headlines Only | `94.25%` |
| **Model v2** | **Headlines + Full Text** | `99.21%` |

The enhanced `v2` model reduced the total classification errors by **over 85%** compared to the baseline model.

## Important Limitations
> **This model is a style-checker, not a fact-checker.**

The model determines if an article is *written* in a style consistent with real journalism or typical misinformation from the training data. It cannot verify the factual accuracy of the claims within an article. Its high performance is partly due to the distinct stylistic differences in the sources from the dataset. It should be used as an assistive tool to flag suspicious content, not as a final arbiter of truth.

## Future Work
- **Add Metadata Features:** Enhance the model by incorporating features like sentiment scores, readability scores, and named entity counts.
- **Experiment with Different Models:** Test more complex algorithms like `LightGBM` or `XGBoost` to see if they can capture more nuanced patterns.
- **Deep Learning Approach:** Implement a more advanced model using deep learning architectures like LSTMs or Transformers (e.g., BERT) for potentially higher accuracy and better semantic understanding.