# Web Services Clustering using Unsubervised Learning Algorithms 🧠✨

This project is a professional and interactive web application that performs **clustering of Web Services** based on their textual descriptions using **unsupervised machine learning**. The system is built with **Flask**, and it allows users to extract semantic features using **TF-IDF** — with the option to enhance results using **WordNet synonyms** for improved clustering accuracy.

## 📌 Project Overview

With the massive growth of available web services, it becomes essential to classify and group them for easier management and retrieval. This tool enables users to:

- Upload a CSV file describing web services.
- Choose between **normal TF-IDF** or **TF-IDF with WordNet expansion**.
- Cluster services per domain using different algorithms.
- Explore and compare clustering results through visual outputs and scores.

## 🔍 Features

- 📂 Upload custom CSV file with service data
- 🧠 Choose between:
  - ✅ **TF-IDF only**
  - ✅ **TF-IDF with WordNet** (synonym expansion via NLTK WordNet)
- 📊 Apply unsupervised clustering algorithms:
  - **KMeans**
  - **DBSCAN**
  - **Hierarchical (Agglomerative) Clustering**
- 🧼 Built-in data cleaning (removes empty rows/columns and extra spaces)
- 📈 Show statistics: number of clusters, silhouette score, frequent terms, etc.

## 🏗️ Tech Stack

- **Backend**: Python, Flask
- **Frontend**: HTML, Tailwind CSS
- **ML & NLP**: Scikit-learn, Pandas, NLTK (WordNet), NumPy
- **Visualization**: Matplotlib, Plotly (optional for bar charts or tables)


## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/web-services-clustering.git
cd web-services-clustering

2. Create Virtual Environment & Install Dependencies
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

You may also need to download WordNet data:
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

3. Run the Application
python app.py
Visit: http://127.0.0.1:5000

🧠 Clustering Algorithms
KMeans: Clusters based on centroids.

DBSCAN: Density-based clustering, handles noise.

Hierarchical (Agglomerative): Builds nested clusters using linkage criteria.

Each algorithm gives:

Cluster assignments

Silhouette Score

Cluster statistics

🌐 WordNet Integration
If enabled, the WordNet-enhanced TF-IDF expands tokens using synonyms from the WordNet lexical database, improving semantic similarity. This step may increase processing time but often improves cluster quality.



👩‍💻 Author
Samiha Smail
Computer Network Engineering & Telecommunications
GitHub • LinkedIn
Email: samihasmail33@gmail.com









