import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

def generate_plots(dataset):
    plots = {}

    # Scatter Plot for first two columns if numeric
    if dataset.select_dtypes(include='number').shape[1] >= 2:
        numeric_columns = dataset.select_dtypes(include='number').columns[:2]
        fig = px.scatter(dataset, x=numeric_columns[0], y=numeric_columns[1])
        plots['scatter_plot'] = fig.to_html(full_html=False)

    # Correlation heatmap for all numeric columns
    correlation_matrix = dataset.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', cbar=True)
    heatmap = save_matplotlib_to_base64()
    plots['correlation_heatmap'] = heatmap

    return plots

# Helper function to save Matplotlib figures as base64 strings
def save_matplotlib_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f'data:image/png;base64,{img_base64}'
