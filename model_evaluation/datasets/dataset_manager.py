import os
import pandas as pd

class DatasetManager:
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'json'}  # Define allowed file extensions

    def __init__(self):
        self.standard_datasets = ['MNIST', 'CIFAR-10', 'IMDB']  # Placeholder for actual paths

    def allowed_file(self, filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in self.ALLOWED_EXTENSIONS

    def get_standard_datasets(self):
        return self.standard_datasets

    def save_uploaded_dataset(self, file):
        dataset_path = os.path.join('datasets', file.filename)
        file.save(dataset_path)
        return file.filename
