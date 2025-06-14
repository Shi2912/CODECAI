import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_data(file_name):
    """Load and clean weather data from CSV."""
    try:
        file_path = os.path.join(os.path.dirname(__file__), file_name)
        print(f"📂 Loading file: {file_path}")
        df = pd.read_csv(file_path)

        print("🔍 Original columns:", df.columns.tolist())

        # Rename columns properly
        df.rename(columns={"Formatted Date": "Date", "Temperature (C)": "Temp"}, inplace=True)

        df['Date'] = pd.to_datetime(df['Date'], utc=True, errors='coerce')
        df = df.sort_values('Date')
        df.set_index('Date', inplace=True)

        df['Temp'] = pd.to_numeric(df['Temp'], errors='coerce')
        df['Temp'] = df['Temp'].interpolate()
        df = df.dropna(subset=['Temp'])

        return df

    except Exception as e:
        print("❌ Error loading file:", e)
        exit()


def add_features(df):
    """Add 'Days' column for model training."""
    df['Days'] = (df.index - df.index.min()).days
    return df


def train_model(df):
    """Train Linear Regression model."""
    X = df[['Days']]
    y = df['Temp']

    model = LinearRegression()
    model.fit(X, y)
    df['Predicted_Temp'] = model.predict(X)

    return model, df


def forecast_future(model, df, future_days=30):
    """Forecast temperature for future days."""
    last_day = df['Days'].max()
    future = pd.DataFrame({'Days': np.arange(last_day + 1, last_day + 1 + future_days)})
    future['Predicted_Temp'] = model.predict(future)
    future['Date'] = df.index.min() + pd.to_timedelta(future['Days'], unit='D')
    future.set_index('Date', inplace=True)
    return future


def plot_results(df, future):
    """Plot actual and predicted temperature trends."""
    all_preds = pd.concat([df[['Temp', 'Predicted_Temp']], future[['Predicted_Temp']]], axis=0)

    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['Temp'], label='Actual Temperature', color='blue')
    plt.plot(all_preds.index, all_preds['Predicted_Temp'], label='Predicted Temperature', color='red', linestyle='--')
    plt.axvline(x=df.index.max(), color='gray', linestyle='--', alpha=0.5)
    plt.title('Weather Temperature Prediction')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def evaluate_model(y_true, y_pred):
    """Evaluate model performance."""
    print("\n--- 📊 Model Evaluation ---")
    print("Mean Absolute Error:", round(mean_absolute_error(y_true, y_pred), 2))
    print("Root Mean Squared Error:", round(np.sqrt(mean_squared_error(y_true, y_pred)), 2))
    print("R² Score:", round(r2_score(y_true, y_pred), 2))


def ask_rain_prediction():
    """Simple rule-based rain prediction."""
    print("\n--- ☁️ Rain Prediction ---")
    try:
        temp = float(input("Enter today's temperature (°C): "))
        humidity = float(input("Enter humidity (%): "))
        wind = float(input("Enter wind speed (km/h): "))

        if humidity > 80 and temp < 25 and wind < 20:
            print("Prediction: 🌧️ It might rain.")
        else:
            print("Prediction: ☀️ It probably won't rain.")
    except ValueError:
        print("⚠️ Invalid input. Please enter numbers only.")


def export_to_csv(df, future, filename="temperature_predictions.csv"):
    """Export prediction data to CSV."""
    combined = pd.concat([df[['Temp', 'Predicted_Temp']], future[['Predicted_Temp']]], axis=0)
    combined.to_csv(filename)
    print(f"\n📁 Predictions exported to: {filename}")


def main():
    print("====== Weather Data Analysis and Prediction ======")

    df = load_data('weatherHistory.csv')

    if df.empty:
        print("❌ No usable data found after cleaning. Please check your CSV file.")
        return

    df = add_features(df)
    model, df = train_model(df)
    future = forecast_future(model, df, future_days=30)
    plot_results(df, future)
    evaluate_model(df['Temp'], df['Predicted_Temp'])

    if input("\nExport predictions to CSV? (y/n): ").lower() == 'y':
        export_to_csv(df, future)

    ask_rain_prediction()


if __name__ == "__main__":
    main()
