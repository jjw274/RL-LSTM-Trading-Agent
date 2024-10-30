import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

class LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM_Model, self).__init__()
        
        # LSTM layer(s)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer to map LSTM outputs to desired output size
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden_state):
        # Forward pass through the LSTM
        out, hidden_state = self.lstm(x, hidden_state)
        
        # Apply the fully connected layer to each time step's output
        out = self.fc(out)  # Shape: [batch_size, sequence_length, output_size]
        
        return out, hidden_state
    
    def init_hidden(self, batch_size):
        # Initialize hidden state and cell state with zeros
        hidden_state = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
                        torch.zeros(num_layers, batch_size, hidden_size).to(device))
        return hidden_state
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_path = 'AAPL_Quotes_Data.csv'
data = pd.read_csv(file_path)

# Convert timestamp to datetime and extract the date to group data by days
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Sort the data by timestamp just to ensure proper order
data = data.sort_values(by='timestamp')

# Extract the date part to identify full trading days
data['date'] = data['timestamp'].dt.date

# Count the number of rows per day to confirm that we have full trading days (390 minutes per day)
data_day_counts = data['date'].value_counts().sort_index()

# Filter out incomplete days (those with fewer than 390 rows)
full_day_data = data.groupby('date').filter(lambda x: len(x) == 390)

# Drop the 'date' column as we don't need it for modeling
full_day_data = full_day_data.drop(columns=['date'])

# Selecting columns for normalization (including 'shares_sold' now, excluding 'timestamp')
columns_to_normalize = full_day_data.columns.difference(['timestamp'])

# Initialize the scaler
scaler = MinMaxScaler()

# Apply the scaler to the selected columns
full_day_data[columns_to_normalize] = scaler.fit_transform(full_day_data[columns_to_normalize])

# Now, we will organize the data into sequences of 390 time steps (1 full day per sequence)
def create_day_sequences(df):
    sequences = []
    grouped = df.groupby(df['timestamp'].dt.date)  # Group by each trading day
    for _, group in grouped:
        sequences.append(group.drop(columns='timestamp').values)  # Drop timestamp for modeling
    return sequences

# Create the sequences
sequences = create_day_sequences(full_day_data)

# Function to compute spread (ask_price_1 - bid_price_1)
def compute_spread(df):
    return df['ask_price_1'] - df['bid_price_1']

# Function to compute volatility (rolling standard deviation of the mid-price)
def compute_volatility(df, window=10):
    mid_price = (df['bid_price_1'] + df['ask_price_1']) / 2
    return mid_price.rolling(window=window).std().fillna(0)

# Function to compute slippage (difference between execution price and mid-price)
def compute_slippage(df):
    mid_price = (df['bid_price_1'] + df['ask_price_1']) / 2
    execution_price = df['bid_price_1']  # Assuming we are selling, so filled at bid price
    return execution_price - mid_price

# Function to compute liquidity (sum of bid and ask sizes)
def compute_liquidity(df):
    bid_liquidity = df[['bid_size_1', 'bid_size_2', 'bid_size_3', 'bid_size_4', 'bid_size_5']].sum(axis=1)
    ask_liquidity = df[['ask_size_1', 'ask_size_2', 'ask_size_3', 'ask_size_4', 'ask_size_5']].sum(axis=1)
    return bid_liquidity + ask_liquidity

# Process each sequence to create inputs and outputs
def prepare_training_data(sequences):
    X = []
    y = []

    for sequence in sequences:
        df = pd.DataFrame(sequence, columns=columns_to_normalize)

        # Calculate target metrics for each sequence
        spread = compute_spread(df)
        volatility = compute_volatility(df)
        slippage = compute_slippage(df)
        liquidity = compute_liquidity(df)

        # Stack the metrics into the output target
        targets = np.column_stack([slippage, spread, volatility, liquidity])

        # Use all but the last time step as inputs, and the next time step's target as output
        X.append(sequence[:-1])   # Input sequence (first 389 time steps)
        y.append(targets[1:])     # Target sequence (next 389 time steps)

    return np.array(X), np.array(y)

# Prepare the data
X, y = prepare_training_data(sequences)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameters
input_size = 20   # Number of input features
hidden_size = 64  # Number of hidden units in LSTM
num_layers = 2    # Number of LSTM layers
output_size = 4   # Number of target metrics (slippage, spread, volatility, liquidity)
num_epochs = 50
learning_rate = 0.001
batch_size = 1    # Sequence per batch for time series

# Initialize the model, loss function, and optimizer
model = LSTM_Model(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.MSELoss()  # Mean Squared Error Loss for regression tasks
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Convert data to PyTorch tensors
X_train_torch = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_torch = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_torch = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_torch = torch.tensor(y_test, dtype=torch.float32).to(device)

# Training loop
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Initialize hidden state for each batch
    hidden_state = model.init_hidden(X_train_torch.size(0))  # Initialize hidden state with batch size

    # Forward pass
    outputs, hidden_state = model(X_train_torch, hidden_state)

    # Calculate the loss
    loss = criterion(outputs, y_train_torch)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# After training, evaluate the model on the test set
model.eval()
with torch.no_grad():
    # Initialize hidden state for the test set
    hidden_state = model.init_hidden(X_test_torch.size(0))

    # Forward pass through the model
    test_outputs, hidden_state = model(X_test_torch, hidden_state)

    # Calculate test loss
    test_loss = criterion(test_outputs, y_test_torch)

print(f'Test Loss: {test_loss.item():.4f}')

#torch.save(model, "lstm.pth")
torch.save(model.state_dict(), "lstm_state_dict.pth")