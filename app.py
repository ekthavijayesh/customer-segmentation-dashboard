
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# GREEN AND WHITE THEME
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background-color: #F7FAF7;
    color: #1F2937;
}

[data-testid="stHeader"] {
    background-color: #FFFFFF;
}

[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #DDE8DD;
}

[data-testid="stSidebar"] * {
    color: #243528;
}

h1, h2, h3, h4 {
    color: #087A35 !important;
}

p, label, .stMarkdown {
    color: #26352A;
}

.main-title {
    color: #087A35;
    font-size: 34px;
    font-weight: 750;
}

.subtitle {
    color: #647468;
    font-size: 16px;
    margin-bottom: 22px;
}

[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #DCE8DD;
    border-radius: 12px;
    padding: 18px;
}

[data-testid="stMetricLabel"] {
    color: #46594A !important;
}

[data-testid="stMetricValue"] {
    color: #087A35 !important;
}

[data-testid="stFileUploader"] {
    background: #F0F8F0;
    border: 2px dashed #81B98D;
    border-radius: 12px;
    padding: 12px;
}

[data-testid="stFileUploader"] section {
    background: #FFFFFF !important;
}

[data-testid="stFileUploader"] section * {
    color: #243528 !important;
}

[data-testid="stFileUploader"] button {
    background-color: #087A35 !important;
    color: white !important;
    border: none !important;
}

.stButton > button,
.stDownloadButton > button {
    background-color: #087A35;
    color: #FFFFFF !important;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background-color: #056328;
    color: #FFFFFF !important;
}

[data-baseweb="select"] > div {
    background-color: #FFFFFF;
    color: #26352A;
    border-color: #C8D8CA;
}

[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    background-color: #FFFFFF;
    color: #26352A;
}

[data-testid="stDataFrame"],
[data-testid="stTable"] {
    background-color: #FFFFFF !important;
    border: 1px solid #DCE8DD !important;
    border-radius: 10px !important;
}

hr {
    border-color: #DCE8DD;
}

[data-testid="stAlert"] {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LIGHT TABLE STYLING
# --------------------------------------------------
def style_table(data):
    return (
        data.style
        .set_properties(**{
            "background-color": "#FFFFFF",
            "color": "#26352A",
            "border-color": "#DCE8DD",
            "text-align": "center"
        })
        .set_table_styles([
            {
                "selector": "thead th",
                "props": [
                    ("background-color", "#EAF5EB"),
                    ("color", "#087A35"),
                    ("font-weight", "bold"),
                    ("border-color", "#D1E4D3")
                ]
            },
            {
                "selector": "tbody td",
                "props": [
                    ("color", "#26352A"),
                    ("border-color", "#DCE8DD")
                ]
            },
            {
                "selector": "tbody tr:nth-child(even)",
                "props": [
                    ("background-color", "#F5FAF5")
                ]
            }
        ])
    )


# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown(
    '<div class="main-title">Customer Segmentation Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Discover customer groups using '
    'K-Means machine learning.</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("Customer Segmentation")
st.sidebar.markdown("### Upload Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload your customer CSV file",
    type=["csv"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About This Project")
st.sidebar.write(
    "This application uses K-Means clustering to group customers "
    "based on selected numerical features."
)

st.sidebar.markdown("### Technology Stack")
st.sidebar.write(
    "- Python\n"
    "- Streamlit\n"
    "- Pandas\n"
    "- Scikit-learn\n"
    "- Matplotlib\n"
    "- Seaborn"
)

st.sidebar.info(
    "Upload your dataset to explore customer segments and "
    "download the results."
)


# --------------------------------------------------
# DATA UPLOAD
# --------------------------------------------------
if uploaded_file is None:
    st.info("Upload a customer CSV file using the sidebar to get started.")

    st.subheader("Expected Dataset Format")

    sample_data = pd.DataFrame({
        "CustomerID": [1, 2, 3, 4, 5],
        "Gender": ["Male", "Female", "Female", "Male", "Female"],
        "Age": [19, 21, 20, 23, 31],
        "Annual Income (k$)": [15, 16, 17, 18, 19],
        "Spending Score (1-100)": [39, 81, 6, 77, 40]
    })

    st.dataframe(
        style_table(sample_data),
        use_container_width=True,
        hide_index=True
    )

    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as error:
    st.error(f"Could not read the CSV file: {error}")
    st.stop()

if df.empty:
    st.error("The uploaded CSV file is empty.")
    st.stop()

if len(df) < 2:
    st.error("At least two customer records are required.")
    st.stop()


# --------------------------------------------------
# IDENTIFY NUMERIC COLUMNS
# --------------------------------------------------
for column in df.columns:
    if df[column].dtype == "object":
        converted = pd.to_numeric(df[column], errors="coerce")

        if (
            converted.notna().sum() > 0
            and converted.notna().sum() == df[column].notna().sum()
        ):
            df[column] = converted

numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

if len(numeric_columns) < 2:
    st.error("Your dataset must have at least two numeric columns.")
    st.stop()


# --------------------------------------------------
# DATASET OVERVIEW
# --------------------------------------------------
st.subheader("Dataset Overview")

m1, m2, m3 = st.columns(3)

with m1:
    st.metric("Total Customers", f"{len(df):,}")

with m2:
    st.metric("Total Features", len(df.columns))

with m3:
    st.metric("Numeric Features", len(numeric_columns))

with st.expander("Preview Uploaded Dataset"):
    st.dataframe(
        style_table(df.head(10)),
        use_container_width=True,
        hide_index=True
    )


# --------------------------------------------------
# CLUSTER CONFIGURATION
# --------------------------------------------------
st.markdown("---")
st.subheader("Clustering Configuration")

default_first = (
    "Annual Income (k$)"
    if "Annual Income (k$)" in numeric_columns
    else numeric_columns[0]
)

default_second = (
    "Spending Score (1-100)"
    if "Spending Score (1-100)" in numeric_columns
    else next(
        column for column in numeric_columns
        if column != default_first
    )
)

c1, c2 = st.columns(2)

with c1:
    feature_1 = st.selectbox(
        "Select First Feature",
        numeric_columns,
        index=numeric_columns.index(default_first)
    )

with c2:
    available_second = [
        column for column in numeric_columns
        if column != feature_1
    ]

    second_index = (
        available_second.index(default_second)
        if default_second in available_second
        else 0
    )

    feature_2 = st.selectbox(
        "Select Second Feature",
        available_second,
        index=second_index
    )

cluster_count = st.slider(
    "Number of Clusters (K)",
    min_value=2,
    max_value=min(10, len(df)),
    value=min(5, len(df), 10)
)


# --------------------------------------------------
# PREPARE DATA AND TRAIN MODEL
# --------------------------------------------------
model_data = df[[feature_1, feature_2]].copy()
model_data = model_data.replace([np.inf, -np.inf], np.nan)

valid_mask = model_data.notna().all(axis=1)
valid_data = model_data.loc[valid_mask].copy()

if len(valid_data) < cluster_count:
    st.error(
        f"At least {cluster_count} complete records are required "
        "for the selected features."
    )
    st.stop()

scaler = StandardScaler()
scaled_data = scaler.fit_transform(valid_data)

model = KMeans(
    n_clusters=cluster_count,
    random_state=42,
    n_init=10
)

labels = model.fit_predict(scaled_data)

result_df = df.copy()
result_df["Cluster"] = pd.Series(
    pd.NA,
    index=result_df.index,
    dtype="Int64"
)

result_df.loc[valid_data.index, "Cluster"] = labels

centroids = scaler.inverse_transform(model.cluster_centers_)


# --------------------------------------------------
# CLUSTER METRICS
# --------------------------------------------------
st.markdown("---")
st.subheader("Clustering Results")

r1, r2, r3 = st.columns(3)

with r1:
    st.metric("Clusters Created", cluster_count)

with r2:
    st.metric("Customers Clustered", f"{len(valid_data):,}")

with r3:
    st.metric("Model Inertia", f"{model.inertia_:,.2f}")


# --------------------------------------------------
# CLUSTER VISUALIZATION
# --------------------------------------------------
st.markdown("---")
st.subheader("Customer Cluster Visualization")

chart1, chart2 = st.columns([1.2, 1])

with chart1:
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.scatterplot(
        x=valid_data[feature_1],
        y=valid_data[feature_2],
        hue=labels,
        palette="Set2",
        s=75,
        alpha=0.85,
        ax=ax
    )

    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="X",
        s=260,
        color="#087A35",
        edgecolor="white",
        linewidth=1.5,
        label="Centroids"
    )

    ax.set_xlabel(feature_1)
    ax.set_ylabel(feature_2)
    ax.set_title("Customer Segments")
    ax.set_facecolor("#FFFFFF")
    fig.patch.set_facecolor("#FFFFFF")
    ax.grid(alpha=0.2)
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with chart2:
    st.markdown("#### Elbow Method")

    max_k = min(10, len(valid_data))
    k_values = list(range(1, max_k + 1))
    inertias = []

    for k in k_values:
        elbow_model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )
        elbow_model.fit(scaled_data)
        inertias.append(elbow_model.inertia_)

    elbow_fig, elbow_ax = plt.subplots(figsize=(7, 5))

    elbow_ax.plot(
        k_values,
        inertias,
        marker="o",
        color="#16803D",
        linewidth=2
    )

    elbow_ax.set_xlabel("Number of Clusters (K)")
    elbow_ax.set_ylabel("Inertia")
    elbow_ax.set_title("Choosing the Number of Clusters")
    elbow_ax.set_xticks(k_values)
    elbow_ax.set_facecolor("#FFFFFF")
    elbow_fig.patch.set_facecolor("#FFFFFF")
    elbow_ax.grid(alpha=0.25)
    elbow_fig.tight_layout()

    st.pyplot(elbow_fig, use_container_width=True)
    plt.close(elbow_fig)


# --------------------------------------------------
# CLUSTER ANALYSIS
# --------------------------------------------------
st.markdown("---")
st.subheader("Cluster Analysis")

clustered = result_df.dropna(subset=["Cluster"]).copy()
clustered["Cluster"] = clustered["Cluster"].astype(int)

summary = (
    clustered.groupby("Cluster")
    .agg(
        **{
            "Number of Customers": ("Cluster", "size"),
            feature_1: (feature_1, "mean"),
            feature_2: (feature_2, "mean")
        }
    )
    .reset_index()
)

summary["Customers (%)"] = (
    summary["Number of Customers"] / len(clustered) * 100
).round(2)

summary[feature_1] = summary[feature_1].round(2)
summary[feature_2] = summary[feature_2].round(2)

summary = summary[
    [
        "Cluster",
        "Number of Customers",
        "Customers (%)",
        feature_1,
        feature_2
    ]
]

st.dataframe(
    style_table(summary),
    use_container_width=True,
    hide_index=True
)

with st.expander("View Cluster Centroid Values"):
    centroid_df = pd.DataFrame(
        centroids,
        columns=[feature_1, feature_2]
    )
    centroid_df.insert(
        0, "Cluster", range(cluster_count)
    )

    st.dataframe(
        style_table(centroid_df.round(2)),
        use_container_width=True,
        hide_index=True
    )


# --------------------------------------------------
# CUSTOMER ASSIGNMENT
# --------------------------------------------------
st.markdown("---")
st.subheader("Customer Assignment")

st.write(
    "View the cluster assigned to each customer based on the "
    "selected numerical features."
)

search = st.text_input(
    "Search Customers",
    placeholder="Search by customer ID, gender, or another value..."
)

assignment_df = result_df.copy()

if search.strip():
    mask = assignment_df.astype(str).apply(
        lambda column: column.str.contains(
            search.strip(),
            case=False,
            na=False
        )
    ).any(axis=1)

    assignment_df = assignment_df.loc[mask]

st.caption(f"Showing {len(assignment_df):,} customer records")

st.dataframe(
    style_table(assignment_df),
    use_container_width=True,
    hide_index=True,
    height=400
)


# --------------------------------------------------
# DOWNLOAD RESULTS
# --------------------------------------------------
st.markdown("---")
st.subheader("Download Results")

st.write(
    "Download the dataset with the cluster labels added."
)

csv = result_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Segmented Data (CSV)",
    data=csv,
    file_name="customer_segments.csv",
    mime="text/csv"
)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")

st.markdown(
    """
    <div style="
        text-align:center;
        color:#52745A;
        padding:12px;
        background-color:#EFF7EF;
        border-radius:8px;
    ">
        Customer Segmentation Dashboard |
        Powered by Python, Streamlit and Scikit-learn
    </div>
    """,
    unsafe_allow_html=True
)
