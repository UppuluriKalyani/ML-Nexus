from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
from visualizations import plots

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle dataset uploads and generate visualizations
@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if a file is included in the request
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    # Save the uploaded file
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Read dataset and generate visualizations
        try:
            dataset = pd.read_csv(filepath)
        except Exception as e:
            return f"Error reading file: {e}"

        plots_data = plots.generate_plots(dataset)

        # Pass generated plots to the template
        return render_template('index.html', plots=plots_data)

if __name__ == '__main__':
    app.run(debug=True)
