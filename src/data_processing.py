import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

# Use absolute path for better recognition
BASE_DIR = Path(__file__).resolve().parent.parent  # Assuming data_processing.py is inside the src folder
DATA_DIR = BASE_DIR / "data"

def get_filepaths_labels(image_dir):
    filepaths = list(image_dir.glob(r'**/*.JPG')) + list(image_dir.glob(r'**/*.jpg')) + list(image_dir.glob(r'**/*.png')) + list(image_dir.glob(r'**/*.png'))
    labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))
    return filepaths, labels

def create_dataframe(filepaths, labels):
    filepaths = pd.Series(filepaths, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')
    return pd.concat([filepaths, labels], axis=1)

def split_train_test_data(image_df):
    return train_test_split(image_df, test_size=0.2, shuffle=True, random_state=42)

def display_class_distribution_barplot(class_names, class_dis):
    plt.figure(figsize=(10, 5))
    sns.barplot(x=class_names, y=class_dis)
    plt.grid()
    plt.axhline(np.mean(class_dis), color='k', linestyle='--', label="Mean Images")
    plt.legend()
    plt.show()
