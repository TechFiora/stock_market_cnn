from src.data_preprocessing import preprocess_data
from src.cnn_model import build_model
from src.visualization import plot_results
import yfinance as yf

# Fetch stock data
data = yf.download("AAPL", start="2015-01-01", end="2024-01-01")
X, y, scaler = preprocess_data(data)

# Build the CNN model and train
model = build_model(input_shape=(X.shape[1], 1))
model.fit(X, y, epochs=10, batch_size=32)

# Make predictions
predictions = model.predict(X)

# Visualize the results
plot_results(y, predictions)
