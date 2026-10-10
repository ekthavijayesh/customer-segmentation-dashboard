# 📊 Customer Segmentation Dashboard Using K-Means Clustering

A machine learning web application that groups customers into different segments based on their numerical characteristics using **K-Means Clustering**. The interactive dashboard allows users to upload customer datasets, explore customer groups, visualize clusters, analyze results, and download segmented data.

## 🌟 Project Overview

Customer segmentation is a machine learning technique used to divide customers into groups with similar characteristics. It helps businesses understand customer behavior, identify spending patterns, and develop targeted marketing strategies.

This project uses the **K-Means clustering algorithm**, an unsupervised machine learning technique, to identify customer segments based on two user-selected numerical features.

The application is built with Python and Streamlit and features a clean, responsive green-and-white dashboard.

## ✨ Features

- **CSV Dataset Upload:** Upload customer data directly through the dashboard.
- **Dataset Overview:** View the total number of customers, total features, and numerical features.
- **Feature Selection:** Choose two numerical features for clustering.
- **Adjustable Clusters:** Select the number of clusters (K) from 2 to 10, subject to dataset size.
- **K-Means Clustering:** Automatically group customers based on their selected features.
- **Cluster Visualization:** View customer segments in a scatter plot, with centroids highlighted.
- **Elbow Method:** Explore inertia values for different K values to help choose the number of clusters.
- **Cluster Analysis:** View the customer count, percentage, and average feature values for each cluster.
- **Centroid Analysis:** Examine the centroid coordinates of the identified clusters.
- **Customer Assignment:** View individual customer records and their assigned cluster.
- **Customer Search:** Search the assignment table using customer values.
- **CSV Download:** Download the segmented dataset, including the generated cluster labels.
- **Green-and-White UI:** A clean interface with readable tables, green accents, and white backgrounds.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Streamlit | Interactive web dashboard |
| Pandas | Data processing and analysis |
| NumPy | Numerical operations |
| Scikit-learn | Standardization and K-Means clustering |
| Matplotlib | Data visualization |
| Seaborn | Cluster visualization |

## 🧠 Machine Learning Algorithm

### K-Means Clustering

K-Means is an unsupervised machine learning algorithm that divides data points into K clusters based on their similarity.

The application follows these steps:

1. Load the customer dataset from a CSV file.
2. Identify the numerical columns.
3. Allow the user to select two features.
4. Handle missing and infinite values in the selected features.
5. Standardize the selected features using `StandardScaler`.
6. Train a K-Means model with the selected number of clusters.
7. Assign each valid customer to a cluster.
8. Visualize the clusters and their centroids.
9. Calculate cluster statistics and allow the results to be downloaded.

### Elbow Method

The Elbow Method helps estimate a suitable number of clusters by plotting the K-Means inertia against different values of K. The point where the reduction in inertia begins to slow down can provide a useful estimate of an appropriate cluster count.

The Elbow Method is a guide rather than a guarantee of the optimal number of clusters.

## 📂 Project Structure

```text
customersegmentation/
│
├── app.py
├── train_model.py
├── customer_segmentation_model.joblib
├── Mall_Customers.csv
├── requirements.txt
├── README.md
└── .gitignore
```

**File descriptions:**

- `app.py` — Runs the interactive Streamlit dashboard and performs clustering.
- `train_model.py` — Training script, if used in your project workflow.
- `customer_segmentation_model.joblib` — Saved model artifact, if required by your workflow.
- `Mall_Customers.csv` — Example customer dataset.
- `requirements.txt` — Lists the Python dependencies.
- `README.md` — Project documentation.
- `.gitignore` — Specifies files and folders Git should ignore.

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.10 or a compatible Python version
- pip
- Git (optional, for cloning or uploading the project)
- Visual Studio Code or another Python editor

### 1. Clone the Repository

Replace `YOUR-USERNAME` with your GitHub username.

```bash
git clone https://github.com/YOUR-USERNAME/Customer-Segmentation-KMeans.git
```

Navigate to the project directory:

```bash
cd Customer-Segmentation-KMeans
```

### 2. Create a Virtual Environment

**Windows:**

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If needed, install the packages individually:

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn
```

### 4. Run the Application

```bash
python -m streamlit run app.py
```

Streamlit will display a local URL, usually:

```text
http://localhost:8501
```

Open the URL in your browser to use the dashboard.

## 📄 Dataset Format

The application accepts CSV files containing at least two numerical columns and at least two customer records.

An example dataset may contain:

| CustomerID | Gender | Age | Annual Income (k$) | Spending Score (1-100) |
|---:|---|---:|---:|---:|
| 1 | Male | 19 | 15 | 39 |
| 2 | Female | 21 | 16 | 81 |
| 3 | Female | 20 | 17 | 6 |
| 4 | Male | 23 | 18 | 77 |
| 5 | Female | 31 | 19 | 40 |

These rows are illustrative examples. Your results depend on the dataset you upload and the features you select.

The dashboard defaults to `Annual Income (k$)` and `Spending Score (1-100)` when those columns are available.

## 📈 Results and Interpretation

The dashboard provides:

- A scatter plot showing customers grouped into clusters.
- Centroid markers representing the centers of the clusters.
- An Elbow Method plot to help assess different cluster counts.
- A cluster summary showing customer counts, percentages, and average feature values.
- A customer assignment table containing the cluster label for each valid record.
- A downloadable CSV containing the original data and the added `Cluster` column.

**Note:** Cluster labels such as 0, 1, and 2 are identifiers, not rankings. The algorithm does not automatically assign business meanings such as "high-value customers" to the clusters. Those interpretations require examining the actual results.

## 🚀 Future Enhancements

- Add automatic suggestions for the number of clusters.
- Support clustering with more than two selected features.
- Add customer segment descriptions based on cluster characteristics.
- Include additional clustering algorithms such as DBSCAN and hierarchical clustering.
- Add more interactive visualizations.
- Deploy the application online using Streamlit Community Cloud.

## 🎯 Applications

Customer segmentation can support:

- Targeted marketing campaigns
- Customer behavior analysis
- Spending pattern identification
- Customer relationship management
- Business strategy and decision-making

## 👩‍💻 Author

**Ektha Vijayesh**

MCA Student | Machine Learning Learner

## 📜 License

This project is intended for educational and portfolio purposes. Add a license file if you want to specify the terms under which others may use, modify, or distribute your code.
