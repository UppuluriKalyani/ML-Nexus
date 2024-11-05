from flask import Flask, render_template, request, redirect, url_for, flash
from datasets.dataset_manager import DatasetManager
from evaluations.evaluator import ModelEvaluator
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for session management

dataset_manager = DatasetManager()
evaluator = ModelEvaluator()

@app.route('/')
def index():
    datasets = dataset_manager.get_standard_datasets()  # Get available datasets
    return render_template('index.html', datasets=datasets)

@app.route('/upload', methods=['POST'])
def upload_dataset():
    if request.method == 'POST':
        file = request.files['dataset']
        if file and dataset_manager.allowed_file(file.filename):
            dataset_name = dataset_manager.save_uploaded_dataset(file)
            flash(f'Dataset {dataset_name} uploaded successfully!', 'success')
        else:
            flash('Invalid file type. Please upload a valid dataset.', 'danger')
    return redirect(url_for('index'))

@app.route('/evaluate', methods=['POST'])
def evaluate_model():
    model_name = request.form['model_name']
    dataset_name = request.form['dataset_name']
    results = evaluator.evaluate(model_name, dataset_name)
    if results:
        return render_template('results.html', results=results)
    else:
        flash('Evaluation failed. Please check the model and dataset.', 'danger')
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
