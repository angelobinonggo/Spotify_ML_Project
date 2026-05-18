import json

filename = "spotify_ml_project.ipynb"

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# SPOTIFY DATA VISUALIZATION AND HIT PREDICTION\n",
            "\n",
            "Spotify transformed music listening forever when it launched in Sweden in 2008. Today, Spotify is the most popular global audio streaming service with millions of users. They are the largest driver of revenue to the music business today.\n",
            "\n",
            "This project analyzes the Top Hits Spotify Dataset (2000-2019) to predict whether a song will become a **Popular Hit** (Popularity Score > 70) utilizing various Machine Learning classification models.\n",
            "\n",
            "### TABLE OF CONTENTS\n",
            "1. IMPORTING LIBRARIES AND LOADING DATA\n",
            "2. DATA INFORMATION\n",
            "3. EXPLORATORY DATA ANALYSIS\n",
            "4. DATA PREPROCESSING AND FEATURE EXTRACTION\n",
            "5. TRAINING THE MACHINE LEARNING MODELS\n",
            "6. CONCLUSION\n",
            "7. END"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 1. IMPORTING LIBRARIES AND LOADING DATA\n",
            "We begin by importing the required tools (Pandas, Matplotlib) and loading our `Spotify_Data.csv` dataset. We also apply a deeply immersive Spotify Dark Mode aesthetic globally to our visualization components."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import warnings\n",
            "warnings.filterwarnings('ignore')\n",
            "\n",
            "# Apply Custom Spotify Dark Aesthetic Globally\n",
            "plt.style.use('dark_background')\n",
            "plt.rcParams.update({\n",
            "    # Deep Black & Dark Charcoal Backgrounds\n",
            "    'axes.facecolor': '#121212', \n",
            "    'figure.facecolor': '#121212',\n",
            "    'savefig.facecolor': '#121212',\n",
            "    # Medium Gray borders and gridlines\n",
            "    'axes.edgecolor': '#535353',\n",
            "    'grid.color': '#535353',\n",
            "    # Light Gray textual context\n",
            "    'text.color': '#B3B3B3',\n",
            "    'axes.labelcolor': '#B3B3B3',\n",
            "    'xtick.color': '#B3B3B3',\n",
            "    'ytick.color': '#B3B3B3'\n",
            "})\n",
            "\n",
            "df = pd.read_csv('Spotify_Data.csv')\n",
            "df.head()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 2. DATA INFORMATION\n",
            "Here we identify that there are 2000 records of tracking data measuring features such as Acousticness, Danceability, Energy, Valence, Tempo, and overall Popularity."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(df.info())\n",
            "print(\"\\nDataset Shape:\", df.shape)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 3. EXPLORATORY DATA ANALYSIS\n",
            "Let's explore how Danceability and Explicit Language map to overall Song Popularity using incredibly styled visualizations."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Explicit Content Distribution\n",
            "plt.figure(figsize=(7, 5))\n",
            "df['explicit'].value_counts().plot(kind='bar', color=['#1DB954', '#212121'], edgecolor='#535353')\n",
            "plt.title(\"Songs with Explicit Content\", fontweight='bold', color='#1DB954')\n",
            "plt.ylabel(\"Count\")\n",
            "plt.xticks(ticks=[0, 1], labels=[\"Clean (False)\", \"Explicit (True)\"], rotation=0)\n",
            "plt.show()\n",
            "\n",
            "# Danceability Histogram\n",
            "plt.figure(figsize=(7, 5))\n",
            "plt.hist(df['danceability'], bins=20, color='#1DB954', edgecolor='#121212')\n",
            "plt.title(\"Song Danceability Distribution\", fontweight='bold', color='#1DB954')\n",
            "plt.xlabel(\"Danceability Score\")\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 4. DATA PREPROCESSING AND FEATURE EXTRACTION\n",
            "Prior to model training, we prepare the dataset by removing string identifiers (like Artist Name and Song Name). Crucially, we format the Target Variable `y` as a binary flag `is_popular` (1 for Popularity > 70, else 0)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Drop textual details\n",
            "df = df.drop(['artist', 'song'], axis=1)\n",
            "\n",
            "# Encode categorical (genres and bools)\n",
            "from sklearn.preprocessing import LabelEncoder\n",
            "le = LabelEncoder()\n",
            "for col in df.select_dtypes(include=['object', 'string', 'bool']).columns:\n",
            "    df[col] = le.fit_transform(df[col])\n",
            "\n",
            "# Create Target Variable\n",
            "df['is_popular'] = df['popularity'].apply(lambda x: 1 if x > 70 else 0)\n",
            "df = df.drop('popularity', axis=1)\n",
            "\n",
            "# Define X and Y\n",
            "X = df.drop('is_popular', axis=1)\n",
            "y = df['is_popular']\n",
            "\n",
            "# Train Test Split\n",
            "from sklearn.model_selection import train_test_split\n",
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
            "print(\"Training Set Size:\", len(X_train))\n",
            "print(\"Testing Set Size:\", len(X_test))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 5. TRAINING THE MACHINE LEARNING MODELS\n",
            "We train four Machine Learning classifiers on our Spotify Features (X) to accurately predict whether a track hits global popularity statuses (y)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from sklearn.tree import DecisionTreeClassifier\n",
            "from sklearn.naive_bayes import GaussianNB\n",
            "from sklearn.neighbors import KNeighborsClassifier\n",
            "from sklearn.linear_model import LogisticRegression\n",
            "from sklearn.metrics import accuracy_score\n",
            "\n",
            "# 1. Decision Tree\n",
            "dt = DecisionTreeClassifier(random_state=42)\n",
            "dt.fit(X_train, y_train)\n",
            "dt_acc = accuracy_score(y_test, dt.predict(X_test))\n",
            "\n",
            "# 2. Naive Bayes\n",
            "nb = GaussianNB()\n",
            "nb.fit(X_train, y_train)\n",
            "nb_acc = accuracy_score(y_test, nb.predict(X_test))\n",
            "\n",
            "# 3. KNN\n",
            "knn = KNeighborsClassifier(n_neighbors=5)\n",
            "knn.fit(X_train, y_train)\n",
            "knn_acc = accuracy_score(y_test, knn.predict(X_test))\n",
            "\n",
            "# 4. Logistic Regression\n",
            "lr = LogisticRegression(max_iter=2000, random_state=42)\n",
            "lr.fit(X_train, y_train)\n",
            "lr_acc = accuracy_score(y_test, lr.predict(X_test))\n",
            "\n",
            "print(\"Decision Tree Accuracy:\", round(dt_acc, 3))\n",
            "print(\"Naive Bayes Accuracy:\", round(nb_acc, 3))\n",
            "print(\"KNN Accuracy:\", round(knn_acc, 3))\n",
            "print(\"Logistic Regression Accuracy:\", round(lr_acc, 3))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "#### Model Accuracy Comparison Visualization"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "models = [\"Decision Tree\", \"Naive Bayes\", \"KNN\", \"Logistic Regression\"]\n",
            "scores = [dt_acc, nb_acc, knn_acc, lr_acc]\n",
            "\n",
            "plt.figure(figsize=(10, 6))\n",
            "colors = ['#1DB954', '#212121', '#535353', '#1DB954']\n",
            "edges = ['#121212', '#1DB954', '#121212', '#121212']\n",
            "bars = plt.bar(models, scores, color=colors, edgecolor=edges, linewidth=2)\n",
            "\n",
            "for bar in bars:\n",
            "    yval = bar.get_height()\n",
            "    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom', fontweight='bold', color='#B3B3B3')\n",
            "\n",
            "plt.title(\"Spotify ML Model Performance\", fontsize=14, fontweight='bold', color='#1DB954')\n",
            "plt.ylabel(\"Accuracy Score\")\n",
            "plt.ylim(0, 1.1)\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 6. CONCLUSION\n",
            "Based on the analysis of 2000 Spotify tracks, the algorithmic classification was able to successfully identify structural patterns mapping song attributes (Danceability, Tempo, Acousticness) to overarching Global Popularity targets.\n",
            "\n",
            "### 7. END"
        ]
    }
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.8"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(filename, 'w') as f:
    json.dump(notebook, f, indent=1)
