from src.data_loader import load_data
from src.model import create_model
from src.utils import save_training_plots, save_metrics
import os
from src.config import *

def main():
    # Load and preprocess data
    print("Loading data...")
    x_train, x_test, y_train, y_test = load_data()
    
    # Create and train model
    print("Creating and training model...")
    model = create_model()
    history = model.fit(
        x_train, y_train,
        epochs=25,
        validation_data=(x_test, y_test),
        verbose=1
    )
    
    # Save results
    print("Saving results...")
    save_training_plots(history)
    save_metrics(model, x_test, y_test)
    
    # Save model
    model.save(os.path.join('results', 'model.h5'))
    print("Training complete! Results saved in 'results' directory.")

if __name__ == "__main__":
    main()
