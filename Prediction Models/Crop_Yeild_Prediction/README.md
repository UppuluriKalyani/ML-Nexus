# Crop Yield Prediction

This project is a machine learning-based web application designed to predict crop yield using various environmental and agricultural factors. The app provides a user-friendly interface for data input, and based on that data, it predicts crop yield, assisting farmers, researchers, and agricultural advisors in decision-making.

## Key Project Points

- **Objective**: To provide an accessible tool for predicting crop yield based on a variety of environmental and soil parameters.
- **User Interface**: A simple, intuitive interface for inputting data and viewing crop yield predictions.
- **Deployment with Flask**: The app is deployed using Flask as a backend to handle requests and render predictions in a web-based format.
- **Template Management**: HTML files (like `index.html`) are organized within the `templates/` folder, while static assets (CSS, JavaScript) reside in the `static/` directory.

Here's the updated **Project Structure** section with `dtr.pkl` included at the end:

---

## Project Structure

```
project-directory/
├── app.py                    # Main application file
├── templates/
│   └── index.html            # HTML template for the web interface
├── static/
│   └── (Optional CSS/JS files for styling)
├── preprocessor.pkl          # Preprocessing pipeline for input data
└── dtr.pkl                   # Decision Tree Regressor model file for predictions
```


## Requirements

- Python 3.x
- Flask
- scikit-learn
- Jinja2

Install dependencies with:
```bash
pip install -r requirements.txt
```

## How to Run

1. Navigate to the project root directory in your terminal.
2. Start the Flask application:
   ```bash
   python app.py
   ```

## Models Used

This project incorporates two machine learning models for crop yield prediction:

1.Linear Regression (lr): A standard linear model that finds the linear relationship between features and the target variable.
2.Lasso Regression (lss): A regularized linear model that uses L1 regularization, useful for feature selection by reducing coefficients of less important features to zero.
3.Ridge Regression (Rid): A linear model that uses L2 regularization to penalize large coefficients, helping to reduce overfitting.
4.Decision Tree Regressor (Dtr): A non-linear model that uses tree-like structures to capture complex patterns in the data and make predictions.

## Troubleshooting

### Template Not Found
- **Error**: `TemplateNotFound` for `index.html`.
  - **Solution**: Ensure `index.html` is in the `templates/` folder, as Flask requires HTML templates to be in this directory.

### ImportError with Watchdog
- **Error**: Issues with `watchdog.events` or reloader functions in development.
  - **Solution**: Update `watchdog` using `pip install --upgrade watchdog`. Alternatively, set `debug=False` to avoid the reloader.

### Development Mode Warning
- **Warning**: `This is a development server. Do not use it in a production deployment.`
  - **Solution**: For production, use a WSGI server, such as `gunicorn`, with `gunicorn app:app`.

### Missing Model Files
- **Error**: Missing model files can prevent predictions from running.
  - **Solution**: Ensure the model file is in the specified directory (e.g., `models/crop_yield_model.pkl`) and that `app.py` correctly references its path.

### Environment Setup
- **Tip**: Use `requirements.txt` to streamline dependency installation with `pip install -r requirements.txt`.

### Debugging Tips
- **Tip**: The Flask debugger provides detailed logs, which can help identify missing files or imports. It also reloads on file changes, highlighting potential errors.

