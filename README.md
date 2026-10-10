# 🛍️ Customer Segmentation Using K-Means Clustering

## 📌 Project Overview

Customer Segmentation is a Machine Learning project that groups customers based on similarities in their annual income and spending behavior using the **K-Means Clustering algorithm**.

The project helps identify different customer segments, understand purchasing patterns, and support businesses in developing targeted marketing strategies.

An interactive **Streamlit web application** allows users to upload customer data, visualize clusters and centroids, analyze customer groups, and download the segmentation results.

## 🎯 Objectives

- Group customers based on annual income and spending score.
- Apply unsupervised machine learning using K-Means clustering.
- Use the Elbow Method to help determine the optimal number of clusters.
- Visualize customer segments and their centroids.
- Analyze customer groups using cluster summaries.
- Build an interactive dashboard using Streamlit.

## 🛠️ Technologies Used

- **Python** – Programming language
- **Pandas** – Data manipulation and analysis
- **NumPy** – Numerical computations
- **Scikit-learn** – K-Means clustering and feature scaling
- **Matplotlib** – Data visualization
- **Seaborn** – Statistical visualization
- **Streamlit** – Interactive web application

## 📂 Dataset

This project uses the Mall Customers dataset, which contains customer information such as:

- Customer ID
- Gender
- Age
- Annual Income (k$)
- Spending Score (1–100)

The main features used for clustering are **Annual Income** and **Spending Score**.

Dataset source: [Mall Customer Segmentation Data – Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

## 🧠 Machine Learning Algorithm

### K-Means Clustering

K-Means is an unsupervised machine learning algorithm that divides data into K clusters based on similarity.

The algorithm works by:
1. Initializing K cluster centroids.
2. Assigning each customer to the nearest centroid.
3. Recalculating centroids based on the mean position of assigned customers.
4. Repeating the process until the clusters stabilize.

### Elbow Method

The Elbow Method helps select a suitable number of clusters by examining the within-cluster sum of squares (WCSS) as K increases.

### Feature Scaling

StandardScaler is used to scale the selected features so that differences in their numerical ranges do not dominate the distance calculations.

## ✨ Features

- Upload customer datasets in CSV format.
- Select numerical features for clustering.
- Adjust the number of clusters interactively.
- Visualize customer segments using scatter plots.
- Display cluster centroids.
- Analyze the Elbow Method graph.
- View customer counts and average feature values for each cluster.
- Download segmented customer data as a CSV file.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/ekthavijayesh/Customer-Segmentation-KMeans.git
```

### 2. Navigate to the project directory

```bash
cd Customer-Segmentation-KMeans
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 6. Run the application

```bash
python -m streamlit run app.py
```

The application will open in your browser. Upload the Mall Customers CSV file to start the analysis.

## 📁 Project Structure

```text
Customer-Segmentation-KMeans/
│
├── app.py
├── train_model.py
├── requirements.txt
├── Mall_Customers.csv
├── .gitignore
└── README.md
```

The dataset and trained model files may be included depending on how you configure your project.

## 📊 Expected Results

The application produces:
- Color-coded customer clusters.
- Centroid markers showing the center of each cluster.
- An Elbow Method plot to support cluster selection.
- A summary of customer counts and average feature values for each segment.
- A downloadable CSV file containing customer cluster assignments.

The resulting clusters can help businesses identify customer groups with different spending patterns.

## 💼 Applications

- Customer relationship management
- Targeted marketing campaigns
- Personalized promotions
- Retail customer behavior analysis
- Customer profiling and business analytics

## 🚀 Future Enhancements

- Deploy the application online.
- Add more customer features for segmentation.
- Improve cluster evaluation using silhouette scores.
- Add interactive filtering and richer visualizations.
- Provide descriptive labels for each customer segment.

## 👩‍💻 Author

**Ektha**

MCA Student | Machine Learning Enthusiast

## 📜 License

This project is intended for educational and learning purposes. Check the dataset's licensing terms before redistributing it.
