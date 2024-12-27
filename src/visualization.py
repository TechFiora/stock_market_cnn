import matplotlib.pyplot as plt


def plot_results(y_test, predictions):
    plt.plot(y_test, color="blue", label="Real Stock Price")
    plt.plot(predictions, color="red", label="Predicted Stock Price")
    plt.title("Stock Price Prediction using CNN")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.show()
