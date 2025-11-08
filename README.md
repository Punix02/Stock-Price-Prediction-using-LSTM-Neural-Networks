# Stock-Price-Prediction-using-LSTM-Neural-Networks
This project focuses on predicting stock market prices using advanced Recurrent Neural Network (RNN) architectures, specifically Long Short-Term Memory (LSTM) networks. It demonstrates both offline forecasting using historical data and real-time adaptive prediction using live stock data fetched from Yahoo Finance.
🚀 Project Overview

The stock market is highly volatile and influenced by several unpredictable factors.
Traditional models like ARIMA or linear regression fail to capture complex, non-linear temporal patterns in financial data.
This project leverages deep learning techniques — LSTM models — to learn sequential dependencies and improve forecast accuracy.

Key features:

Predicts future stock prices based on historical OHLC data
Includes data normalization (Min-Max Scaling) for stable training
Evaluates model using RMSE and visualizes training/testing results
Integrates real-time data fetching using the Yahoo Finance API
Continuously updates live predictions and visual dashboards
Supports automatic CSV logging of predictions

🧠 Model Workflow

Data Collection: Historical stock data from reliable sources (e.g., Yahoo Finance, Kaggle).
Data Preprocessing: Cleaning, handling missing values, and applying Min-Max normalization.
Model Training: Building and training an LSTM network using TensorFlow/Keras.
Evaluation: Calculating RMSE and visualizing predicted vs. actual prices.
Real-Time Prediction: Fetching live data every minute and predicting the next price dynamically.
Visualization: Real-time Matplotlib dashboard showing actual and predicted stock trends.

⚙️ Technologies Used

Python 3.10+
TensorFlow / Keras – Deep Learning Framework
Scikit-learn – Data Preprocessing & Metrics
NumPy / Pandas – Data Handling
Matplotlib – Visualization
yFinance API – Live Stock Data
Joblib – Saving and loading MinMaxScaler
CSV Logging + Live Dashboard

💾 Installation & Setup

Clone the repository

git clone https://github.com/yourusername/stock-price-prediction-lstm.git
cd stock-price-prediction-lstm

Install dependencies

pip install -r requirements.txt
(or manually install)
pip install tensorflow scikit-learn pandas numpy matplotlib yfinance joblib

Train the model (optional)
Run the training script (Stocks predictions using LSTM neural networks.py)

Save the model:
model.save("stock_model_lstm.h5")

Save the scaler:
joblib.dump(scaler, "scaler.save")

Run the real-time predictor

python realtime_predictor.py

📊 Example Output

Without Normalization: Unstable training, high RMSE, poor prediction alignment.
With Normalization: Smooth convergence, lower RMSE, accurate trend tracking.
Real-Time Mode: Continuously updated prediction chart with CSV logging.
🔍 Performance Metric
