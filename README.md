### `README.md` for **Health Prediction using Neural Networks**  

```md
# Health Prediction using Neural Networks

![GitHub repo](https://img.shields.io/github/stars/sriharinmn/health-prediction?style=social)
![GitHub forks](https://img.shields.io/github/forks/sriharinmn/health-prediction?style=social)
![GitHub license](https://img.shields.io/github/license/sriharinmn/health-prediction)

## Overview

This project is a **Health Prediction System** that utilizes **Neural Networks (ANN)** to predict the likelihood of **Diabetes, Heart Disease, and Parkinson's Disease**. It uses **Streamlit** for visualization and provides an intuitive user interface for easy interaction.

## Features

✅ **Predicts multiple diseases** (Diabetes, Heart Disease, Parkinson's)  
✅ **Uses Artificial Neural Networks (ANN)** for classification  
✅ **Interactive UI built with Streamlit**  
✅ **Preprocessing with Scikit-learn** for handling missing values and feature scaling  
✅ **Optimized using Adam optimizer and ReLU activation function**  

## System Design

The workflow consists of:  

1. **Data Collection & Preprocessing**  
   - Medical datasets are collected and preprocessed using **StandardScaler**  
   - Missing values handled and data split into train/test sets  

2. **Model Training & Evaluation**  
   - Individual ANN models are trained for each disease  
   - Performance is evaluated using **accuracy, precision, recall, and F1-score**  

3. **User Interaction via Streamlit**  
   - Users can input symptoms and get a prediction from the trained models  

## Tech Stack

- **Python 3.x**
- **TensorFlow/Keras** (for neural networks)
- **Scikit-learn** (for preprocessing and evaluation)
- **Pandas & NumPy** (for data handling)
- **Matplotlib & Seaborn** (for visualization)
- **Streamlit** (for web interface)

## Installation

Clone the repository:

```sh
git clone https://github.com/sriharinmn/health-prediction.git
cd health-prediction
```

Install dependencies:

```sh
pip install -r requirements.txt
```

Run the Streamlit app:

```sh
streamlit run app.py
```

## Dataset

- The datasets used for training the models can be found at:  
  - [Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  
  - [Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)  
  - [Parkinson’s Dataset](https://www.kaggle.com/datasets/nidaguler/parkinsons-disease-dataset)  

## Screenshots

| Input Form | Prediction Output |
|------------|------------------|
| ![Input](assets/input.png) | ![Output](assets/output.png) |

## Contributing

Feel free to contribute by submitting a **Pull Request**.  
For major changes, please open an **issue** first.

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

### 📢 **Star this repo** ⭐ and follow for updates! 🚀
```
