# 🎵 Spotify Hit Predictor & ML Studio

[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classifiers-1DB954?style=for-the-badge&logo=spotify)](https://github.com)
[![Python Version](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)](https://python.org)
[![JavaScript](https://img.shields.io/badge/JavaScript-ES6+-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)](https://javascript.info)
[![HTML5 / CSS3](https://img.shields.io/badge/UI-Spotify%20Dark-1ED760?style=for-the-badge&logo=html5)](https://w3.org)

An advanced Machine Learning project analyzing the **Top Hits Spotify Dataset (2000–2019)** to classify and predict whether a song will become a **Popular Hit** (Popularity Score > 70) utilizing various classic Machine Learning classification models. 

This repository features both a baseline **Python Jupyter Pipeline** and a **premium, real-time client-side interactive Web Studio** built from scratch in Vanilla JavaScript, HTML, and CSS.

---

## 📸 Interactive Web Studio

The project includes an immersive client-side web application styled with Spotify's signature dark aesthetic, glassmorphism card layouts, and custom-styled range sliders.

It allows you to **train models in real-time** on the dataset inside the browser and adjust:
*   🎯 **Popularity Target Threshold** (defining what counts as a "Hit" from popularity score 50 to 90)
*   ✂️ **Train-Test Set Split Ratio**
*   ⚙️ **Model-specific Hyperparameters** (Max Depth, K-Neighbors, Learning Rate, Epochs)
*   🎛️ **Audio Characteristics Slider Deck** (to dynamically predict hit potential and calculate real-time probability)

---

## 📁 Repository Architecture

The project directory consists of the following key components:

| Filename | Role / Purpose |
| :--- | :--- |
| 📊 **`Spotify_Data.csv`** | Dataset containing 2,000 top Spotify tracks with numeric audio features. |
| 🐍 **`generate_spotify_notebook.py`** | Python generation utility to assemble the Jupyter Notebook cell structure. |
| 📓 **`spotify_ml_project.ipynb`** | Fully executed Jupyter Notebook containing python training, stats, and Matplotlib plots. |
| 🌐 **`spotify_ml_project.html`** | Exported HTML report of the Jupyter Notebook execution. |
| 🧠 **`ml_models.js`** | Core client-side JS library implementing CSV parsing, MinMax scaling, splitting, and ML models. |
| 🎨 **`spotify_ml_studio.html`** | Premium, responsive visual web dashboard for interactive ML training and song predictions. |

---

## 🎛️ Audio Features Analyzed

The models predict song hit potential based on the following structural audio characteristics:

*   **Danceability:** How suitable a track is for dancing (0.0 to 1.0).
*   **Energy:** Perceptual measure of intensity and activity (0.0 to 1.0).
*   **Acousticness:** Confidence measure of whether the track is acoustic (0.0 to 1.0).
*   **Valence:** Musical positiveness conveyed by a track (0.0 to 1.0).
*   **Liveness:** Detects the presence of an audience in the recording (0.0 to 1.0).
*   **Instrumentalness:** Predicts whether a track contains no vocals (0.0 to 1.0).
*   **Speechiness:** Detects the presence of spoken words in a track (0.0 to 1.0).
*   **Loudness:** Overall loudness of a track in decibels (dB, usually -60 to 0).
*   **Tempo:** Overall estimated tempo of a track in beats per minute (BPM).
*   **Explicit Content:** Binary flag indicating explicit lyrics (True/False).
*   **Key & Mode:** Estimated scale key (0 to 11) and melodic mode (Major=1, Minor=0).

---

## 📊 Baseline Model Performance

Trained on 2,000 Spotify tracks with a baseline threshold of `Popularity > 70` (20% test split):

| Classifier Model | Test Accuracy | Confusion Matrix (TN, FP, FN, TP) | Features / Advantages |
| :--- | :---: | :---: | :--- |
| **Logistic Regression** | **65.25%** | `[[255, 6], [133, 6]]` | High baseline stability, excellent scaling, probability calibration. |
| **Gaussian Naive Bayes** | **63.75%** | `[[229, 32], [113, 26]]` | Independent feature distribution analysis, fast training. |
| **K-Nearest Neighbors (K=5)** | **63.25%** | `[[220, 41], [106, 33]]` | Distance-based catalog matching in normalized feature space. |
| **Decision Tree (Max Depth=5)** | **61.50%** | `[[188, 73], [81, 58]]` | Excellent transparency, identifies categorical Gini division boundaries. |

---

## 🚀 Getting Started & How to Run

### 1. Running the Interactive Web Studio (Recommended)

You can launch and explore the real-time ML studio dashboard in any browser using three methods:

#### Method A: Using the Local Workspace Web Server
If you've started the local python web server, simply navigate your browser to:
👉 **[http://localhost:8000/spotify_ml_studio.html](http://localhost:8000/spotify_ml_studio.html)**

#### Method B: Starting a New Server
Open a terminal in the project directory and execute:
```powershell
python -m http.server 8000
```
Then navigate to `http://localhost:8000/spotify_ml_studio.html` in your browser.

#### Method C: Drag & Drop (Offline / Local File Protocol)
1. Open the [spotify_ml_studio.html](spotify_ml_studio.html) file directly in your browser (Double-click the file).
2. Since modern browsers block local resource fetch by default (`file://` protocol CORS block), you will see an upload overlay.
3. Simply drag **`Spotify_Data.csv`** from your explorer and drop it onto the web page.
4. The dashboard will instantly parse the CSV, scale the features, train your models, and initialize the predictor!

### 2. Running the Python Jupyter Notebook
Ensure you have the required packages installed:
```powershell
pip install pandas numpy scikit-learn matplotlib jupyter nbconvert
```
To run the notebook interactively, open a terminal and run:
```powershell
jupyter notebook spotify_ml_project.ipynb
```
To re-run the entire notebook programmatically from the command line and save the executed results:
```powershell
jupyter nbconvert --to notebook --execute spotify_ml_project.ipynb --inplace
```

---

## 🎓 Academic Credit
This project is part of:
**Predicting Song Popularity: A Machine Learning Study on Spotify Hits**  
*Authored by:* **John Michael Angelo C. Binonggo**
