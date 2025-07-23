# Importing necessary libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_data(file_name):
    """Load and clean weather data from CSV file."""
    try:
        # Get full file path
        file_path = os.path.join(os.path.dirname(__file__), file_name)
        print(f"📂 Loading file: {file_path}")
        df = pd.read_csv(file_path)

        print("🔍 Original columns:", df.columns.tolist())

        # Rename columns for consistency
        df.rename(columns={"Formatted Date": "Date", "Temperature (C)": "Temp"}, inplace=True)

        # Convert Date column to datetime format and sort
        df['Date'] = pd.to_datetime(df['Date'], utc=True, errors='coerce')
        df = df.sort_values('Date')
        df.set_index('Date', inplace=True)

        # Ensure temperature values are numeric and handle missing values by interpolation
        df['Temp'] = pd.to_numeric(df['Temp'], errors='coerce')
        df['Temp'] = df['Temp'].interpolate()
        df = df.dropna(subset=['Temp'])

        return df

    except Exception as e:
        print("❌ Error loading file:", e)
        exit()


def add_features(df):
    """Add numeric 'Days' feature for model input based on date."""
    df['Days'] = (df.index - df.index.min()).days
    return df


def train_model(df):
    """Train a Linear Regression model to predict temperature."""
    X = df[['Days']]  # Independent variable (day number)
    y = df['Temp']    # Dependent variable (temperature)

    model = LinearRegression()
    model.fit(X, y)  # Fit the model on historical data

    # Add predicted values to the DataFrame
    df['Predicted_Temp'] = model.predict(X)

    return model, df


def forecast_future(model, df, future_days=30):
    """Forecast temperature for the next 'future_days' using the trained model."""
    last_day = df['Days'].max()

    # Create a DataFrame for future dates
    future = pd.DataFrame({'Days': np.arange(last_day + 1, last_day + 1 + future_days)})

    # Predict future temperatures
    future['Predicted_Temp'] = model.predict(future)

    # Convert day numbers back to dates
    future['Date'] = df.index.min() + pd.to_timedelta(future['Days'], unit='D')
    future.set_index('Date', inplace=True)

    return future


def plot_results(df, future):
    """Plot actual vs predicted temperature trends, including future forecast."""
    # Combine historical and future predictions
    all_preds = pd.concat([df[['Temp', 'Predicted_Temp']], future[['Predicted_Temp']]], axis=0)

    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['Temp'], label='Actual Temperature', color='blue')
    plt.plot(all_preds.index, all_preds['Predicted_Temp'], label='Predicted Temperature', color='red', linestyle='--')
    plt.axvline(x=df.index.max(), color='gray', linestyle='--', alpha=0.5)  # Mark boundary between real and forecast
    plt.title('Weather Temperature Prediction')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def evaluate_model(y_true, y_pred):
    """Print evaluation metrics for the regression model."""
    print("\n--- 📊 Model Evaluation ---")
    print("Mean Absolute Error:", round(mean_absolute_error(y_true, y_pred), 2))
    print("Root Mean Squared Error:", round(np.sqrt(mean_squared_error(y_true, y_pred)), 2))
    print("R² Score:", round(r2_score(y_true, y_pred), 2))


def ask_rain_prediction():
    """A basic rule-based rain prediction based on user input."""
    print("\n--- ☁️ Rain Prediction ---")
    try:
        # Get user inputs
        temp = float(input("Enter today's temperature (°C): "))
        humidity = float(input("Enter humidity (%): "))
        wind = float(input("Enter wind speed (km/h): "))

        # Basic conditional rule for rain likelihood
        if humidity > 80 and temp < 25 and wind < 20:
            print("Prediction: 🌧️ It might rain.")
        else:
            print("Prediction: ☀️ It probably won't rain.")
    except ValueError:
        print("⚠️ Invalid input. Please enter numbers only.")


def export_to_csv(df, future, filename="temperature_predictions.csv"):
    """Export both historical and forecasted predictions to a CSV file."""
    combined = pd.concat([df[['Temp', 'Predicted_Temp']], future[['Predicted_Temp']]], axis=0)
    combined.to_csv(filename)
    print(f"\n📁 Predictions exported to: {filename}")


def main():
    """Main driver function for the weather prediction workflow."""
    print("====== Weather Data Analysis and Prediction ======")

    # Step 1: Load and clean data
    df = load_data('weatherHistory.csv')

    if df.empty:
        print("❌ No usable data found after cleaning. Please check your CSV file.")
        return

    # Step 2: Feature engineering
    df = add_features(df)

    # Step 3: Model training
    model, df = train_model(df)

    # Step 4: Forecasting
    future = forecast_future(model, df, future_days=30)

    # Step 5: Visualization
    plot_results(df, future)

    # Step 6: Evaluation
    evaluate_model(df['Temp'], df['Predicted_Temp'])

    # Step 7: Optional CSV export
    if input("\nExport predictions to CSV? (y/n): ").lower() == 'y':
        export_to_csv(df, future)

    # Step 8: Ask for simple rain prediction
    ask_rain_prediction()


# Entry point of the program
if __name__ == "__main__":
    main()
