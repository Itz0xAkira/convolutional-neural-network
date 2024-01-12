# Convolutional Neural Network (CNN) Project

## Overview

The **Convolutional Neural Network (CNN) Project** is a comprehensive example showcasing the implementation of a convolutional neural network for image classification tasks. CNNs have proven to be highly effective in image-related tasks, making them a popular choice for computer vision applications.

### Key Features

1. **Transfer Learning:** The project leverages the power of transfer learning by utilizing the ResNet152V2 architecture, a pre-trained deep neural network, to enhance the model's performance on image classification tasks.

2. **Data Processing:** Efficient data processing techniques are implemented in the `data_processing.py` module to handle image loading, preprocessing, and dataset splitting.

3. **Model Building:** The `model_building.py` module contains functions for creating and compiling a CNN model based on the ResNet152V2 architecture. It also includes utilities for setting up callbacks and evaluating the model.

4. **Visualization:** The `visualization.py` module offers visualization tools to display class distribution, sample images, training/validation accuracy and loss curves, confusion matrix, and more.

5. **Flexible Structure:** The project follows a modular structure, making it easy to understand, extend, and customize. The main script, `main.py`, serves as the entry point for running the entire project.

### How to Use

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/cnn-project.git
   cd cnn-project

2. **Install Dependencies:**
   ```bash
    pip install -r requirements.txt

3. **Run the Project:**
   ```bash
    python main.py

### Dependencies

The project relies on popular deep learning libraries such as TensorFlow, NumPy, Pandas, and visualization tools like Seaborn and Plotly. These dependencies are listed in the requirements.txt file.

[**The IQ-OTH/NCCD Lung Cancer Dataset on Kaggle**](https://www.kaggle.com/datasets/hamdallak/the-iqothnccd-lung-cancer-dataset)

**Dataset Information**
- **Dataset Source:** Iraq-Oncology Teaching Hospital/National Center for Cancer Diseases (IQ-OTH/NCCD)
- **Data Collection Period:** Three months in fall 2019
- **Data Content:**
  - 1190 CT scan images
  - Represents 110 cases
  - Includes lung cancer patients in different stages and healthy subjects
- **Case Distribution:**
  - 40 cases diagnosed as malignant
  - 15 cases diagnosed as benign
  - 55 cases classified as normal
- **Image Format:** DICOM
- **Scanner Used:** Siemens SOMATOM
- **CT Protocol:**
  - 120 kV
  - 1 mm slice thickness
  - Window width: 350 to 1200 HU
  - Window center: 50 to 600
  - Breath hold at full inspiration
- **De-identification:** All images de-identified before analysis
- **Consent:** Written consent waived; approved by institutional review board
- **Case Variation:**
  - Gender, age, educational attainment, area of residence, and living status
  - Occupations include employees of Iraqi ministries, farmers, and gainers
- **Geographic Origin:** Mainly from middle region of Iraq (Baghdad, Wasit, Diyala, Salahuddin, Babylon provinces).
