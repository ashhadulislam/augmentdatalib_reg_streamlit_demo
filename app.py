import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import os
import pickle
import matplotlib.pyplot as plt
from knnor_reg import data_augment

def normalize_data(df):
    """Normalize all columns except the last one."""
    df_normalized = df.copy()
    for col in df.columns[:-1]: 
        df_normalized[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df_normalized

# Load and preprocess datasets from the 'data' directory
files = os.listdir("data")
datasets = {f: normalize_data(pd.read_csv(f"data/{f}")) for f in files}

# Dropdown menu to select dataset
st.sidebar.title("Select Dataset")
dataset_name = st.sidebar.selectbox("Choose a dataset", list(datasets.keys()))

# Load the selected dataset
df = datasets[dataset_name]

# Import the KNNOR_Reg module

knnor_reg = data_augment.KNNOR_Reg()

# Sidebar options for KNNOR parameters
st.sidebar.title("KNNOR Parameters")
num_nbrs = st.sidebar.slider("Number of Neighbors", min_value=3, max_value=10, value=3)

# proportion_of_minority = st.sidebar.selectbox("Proportion of Minority", [0.5, 0.7, 0.9])ss
proportion_of_minority = st.sidebar.slider("Proportion of Minority", min_value=0.3, max_value=0.9, value=0.3)
k_bins = st.sidebar.slider("Number of bins", min_value=5, max_value=10, value=5)


# Prepare the data for augmentation
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

hist = plt.hist(y, bins=k_bins)
plt.close()
freqs, vals = hist[0], hist[1]
min_freq = int(np.min(freqs))
max_freq=int(np.max(freqs))

threshold_freq = st.sidebar.slider("Threshold frequency", min_value=min_freq, max_value=max_freq, value=min_freq)

    

# Check if augmented data pickle file exists
pickle_file = f"pickles/augmented_{dataset_name}_{num_nbrs}_{k_bins}_{threshold_freq}_{proportion_of_minority}.pkl"

if os.path.exists(pickle_file):
    with open(pickle_file, "rb") as f:
        X_new, y_new = pickle.load(f)
else:
    X_new, y_new = knnor_reg.fit_resample(X, y, target_freq=threshold_freq)
    with open(pickle_file, "wb") as f:
        pickle.dump((X_new, y_new), f)

print(X.shape,X_new.shape)
# Replace the original dataframe with the augmented data
df = pd.DataFrame(X_new, columns=df.columns[:-1])
df[df.columns[-1]] = y_new

# Sidebar options for columns and bins
st.sidebar.title("Select Columns and Bins")
columns = df.columns.tolist()
x_axis = st.sidebar.selectbox("Select X-axis", columns)
y_axis = st.sidebar.selectbox("Select Y-axis", columns, index=1)


# Create the histogram using Matplotlib
last_column = df.columns[-1]
hist, bins = np.histogram(df[last_column], bins=k_bins)



# Create colors for bins
colors = px.colors.qualitative.Vivid
bin_colors = [colors[i % len(colors)] for i in range(len(bins)-1)]

# Create Plotly Histogram using Matplotlib data
hist_fig = go.Figure()
for i in range(len(bins)-1):
    hist_fig.add_trace(go.Bar(
        x=[(bins[i] + bins[i+1]) / 2],
        y=[hist[i]],
        width=[(bins[i+1] - bins[i]) * 0.9],
        marker_color=bin_colors[i],
        name=f'Bin {i}'
    ))

hist_fig.update_layout(
    title=f'Histogram of {last_column}',
    xaxis_title=last_column,
    yaxis_title='Frequency'
)

# Create Scatter Plot
scatter_fig = px.scatter(
    df,
    x=x_axis,
    y=y_axis,
    color=pd.cut(df[last_column], bins=k_bins).astype(str),
    color_continuous_scale='Viridis'
)
scatter_fig.update_layout(
    title=f'Scatter plot of {x_axis} vs {y_axis}',
    xaxis_title=x_axis,
    yaxis_title=y_axis
)

# Display plots side by side
st.plotly_chart(hist_fig)
st.plotly_chart(scatter_fig)
