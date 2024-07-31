from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def predict_features(df, province, input_year, input_pr="", input_tr="", input_population=""):
    
    if df.empty:
        raise ValueError("The input DataFrame is empty. Please provide valid data.")
 
    # Drop the 'Province' column as it is not relevant for the model
    df_cleaned = df.drop(columns=['Province'])

    # Initialize the scaler
    scaler = MinMaxScaler()

    # Normalize the data (excluding the 'Year' column)
    df_scaled = df_cleaned.copy()
    df_scaled.iloc[:, 1:] = scaler.fit_transform(df_cleaned.iloc[:, 1:])

    # Convert the DataFrame to a numpy array
    data_array = df_scaled.values

    # Function to create sequences and targets
    def create_sequences(data, seq_length):
        sequences = []
        targets = []
        for i in range(len(data) - seq_length):
            sequences.append(data[i:i+seq_length, 1:])  # Exclude the 'Year' column
            targets.append(data[i+seq_length, 1:])  # All features are targets
        return np.array(sequences), np.array(targets)

    # Define the sequence length
    SEQ_LENGTH = 3

    # Create sequences and targets
    sequences, targets = create_sequences(data_array, SEQ_LENGTH)

    # Split the data into training and test sets
    train_size = int(len(sequences) * 0.8)
    train_sequences, test_sequences = sequences[:train_size], sequences[train_size:]
    train_targets, test_targets = targets[:train_size], targets[train_size:]

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(SEQ_LENGTH, train_sequences.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(train_targets.shape[1]))  # Output layer matches the number of features

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Train the model with early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(train_sequences, train_targets, epochs=200, batch_size=16, validation_data=(test_sequences, test_targets), callbacks=[early_stopping], verbose=1)

    # Evaluate the model
    test_predictions = model.predict(test_sequences)

    # Calculate R² score and MAPE for the population predictions
    r2 = r2_score(test_targets[:, 0], test_predictions[:, 0])
    mape = mean_absolute_percentage_error(test_targets[:, 0], test_predictions[:, 0])
    mae = np.mean(np.abs(test_targets[:, 0] - test_predictions[:, 0]))

    print(f'R² score for population prediction: {r2:.2f}')
    print(f'MAPE for population prediction: {mape:.2%}')
    print(f'MAE for population prediction: {mae:.2f}')

    # Function to predict future values iteratively using the LSTM model
    def predict_future_values_lstm(model, initial_data, years_to_predict, seq_length):
        predictions = []
        current_sequence = initial_data[-seq_length:, 1:]  # Exclude the 'Year' column
        current_sequence = current_sequence.reshape((1, seq_length, current_sequence.shape[1]))

        for year in range(years_to_predict):
            # Predict the next set of features
            next_values = model.predict(current_sequence)[0]
            predictions.append(next_values)

            # Update the current sequence by removing the oldest entry and adding the new prediction
            next_sequence = np.append(current_sequence[0, 1:, :], next_values.reshape(1, -1), axis=0)
            current_sequence = next_sequence.reshape((1, seq_length, current_sequence.shape[2]))

        return np.array(predictions)

    # Use the last known data points as the initial input
    initial_data_lstm = df_scaled.values  # Use all scaled data for initial input
    calc_year = input_year - 2022
    years_to_predict = max(0, calc_year)  # Ensure years_to_predict is not negative

    if years_to_predict == 0:
        return pd.DataFrame(columns=df_cleaned.columns)

    # Predict future values
    future_values_lstm = predict_future_values_lstm(model, initial_data_lstm, years_to_predict, SEQ_LENGTH)

    # Prepare the data for inverse transformation
    future_values_full = np.zeros((years_to_predict, df_scaled.shape[1] - 1))  # Exclude the 'Year' column
    future_values_full[:, :] = future_values_lstm

    # Inverse transform to get the values in the original scale
    future_values_original_scale = scaler.inverse_transform(future_values_full)

    # Convert the predictions to a DataFrame with appropriate column labels
    predicted_columns = df_cleaned.columns[1:]  # Exclude the 'Year' column
    predicted_df = pd.DataFrame(future_values_original_scale, columns=predicted_columns)

    # Add the years for the predictions
    predicted_df.insert(0, 'Year', range(2022, 2022 + years_to_predict))

    # Predict Beds and Physicians based on the predicted population using Linear Regression
    beds_model = LinearRegression()
    physicians_model = LinearRegression()

    # Train the regression models on historical data
    beds_model.fit(df_cleaned[['Population']], df_cleaned['Beds'])
    physicians_model.fit(df_cleaned[['Population']], df_cleaned['Physician'])

    # Predict Beds and Physicians based on the predicted population
    predicted_df['Beds'] = beds_model.predict(predicted_df[['Population']])
    predicted_df['Physician'] = physicians_model.predict(predicted_df[['Population']])

    # Format the DataFrame to display whole numbers
    predicted_df = predicted_df.round().astype(int)
    predicted_df['Province'] = province
    
    return predicted_df
