import gym
from gym import spaces
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from stable_baselines3 import SAC
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.sac.policies import MlpPolicy
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import json
from Benchmark import Benchmark
import matplotlib.pyplot as plt

file_path = 'AAPL_Quotes_Data.csv'
data = pd.read_csv(file_path)

# Convert timestamp to datetime
data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
# Drop any rows where the timestamp conversion failed
data = data.dropna(subset=['timestamp'])

# Sort by timestamp to ensure chronological order
data = data.sort_values(by='timestamp')

# Extract and filter only full trading days
data['date'] = data['timestamp'].dt.date
data_day_counts = data['date'].value_counts().sort_index()
full_day_data = data.groupby('date').filter(lambda x: len(x) == 390)

# Drop unnecessary columns
full_day_data = full_day_data.drop(columns=['date'])

# Select columns for normalization
columns_to_normalize = full_day_data.columns.difference(['timestamp'])

# Apply MinMaxScaler to normalize data columns
scaler = MinMaxScaler()
full_day_data[columns_to_normalize] = scaler.fit_transform(full_day_data[columns_to_normalize].fillna(0))

# Verify there are no missing or infinite values
full_day_data = full_day_data.replace([np.inf, -np.inf], 0).dropna()

def create_day_sequences(df):
    sequences = []
    grouped = df.groupby(df['timestamp'].dt.date)  # Group by each trading day
    for _, group in grouped:
        sequences.append(group.drop(columns='timestamp').values)  # Drop timestamp for modeling
    return sequences

# Assuming full_day_data is prepared as described in previous steps
sequences = create_day_sequences(full_day_data)

print("Prepared data shape:", full_day_data.shape)
print("Prepared data sample:\n", full_day_data.head())

# Hyperparameters
GAMMA = 0.99
TAU = 0.005
LR = 3e-4
INITIAL_TEMPERATURE = 5.0
TARGET_ENTROPY = -2
REPLAY_BUFFER_SIZE = int(1e6)
BATCH_SIZE = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import gym
from gym import spaces
import numpy as np
import torch
import pandas as pd

class TradingEnv(gym.Env):
    def __init__(self, sequences, lstm, initial_inventory=1000):
        super(TradingEnv, self).__init__()
        self.sequences = sequences  # List of pre-processed day sequences
        self.lstm_model = lstm
        self.initial_inventory = initial_inventory
        self.inventory = self.initial_inventory
        self.day_index = 0  # Index to track the current day in sequences
        self.day_data = self.sequences[self.day_index]
        self.current_step = 0
        self.max_steps = 390  # Maximum steps in a trading day, assuming each day is 390 rows
        self.hidden_state = init_hidden(self.lstm_model)
        self.window_size = 5
        self.history_shape = (self.window_size, sequences[0].shape[1])
        self.history = np.zeros(self.history_shape, dtype=np.float32)
        self.share_frac = 20

        # Define action and observation space
        self.action_space = spaces.Box(low=0, high=initial_inventory / self.share_frac, shape=(1,), dtype=np.float32)
        initial_state = self._get_observation()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=initial_state.shape, dtype=np.float32)
        
        # Rolling history setup for LSTM-based state observations
        self.window_size = 5
        self.history_shape = (self.window_size, sequences[0].shape[1])
        self.history = np.zeros(self.history_shape, dtype=np.float32)

    def reset(self):
        # Reset inventory and step count for a new day
        self.inventory = self.initial_inventory
        self.current_step = 0
        self.hidden_state = init_hidden(self.lstm_model)

        # Load the data for the current day from sequences
        self.day_data = self.sequences[self.day_index]
        self.day_index = (self.day_index + 1) % len(self.sequences)  # Rotate to the next day for the following episode
        
        # Initialize the rolling window with the first state data
        self.history.fill(0)
        self.history[-1] = self.day_data[0]

        # Initial state preparation
        state = self._get_observation().astype(np.float32)
        
        return state

    def _get_state(self):
        # Construct the base state representation from current data and inventory
        data_obs = self.day_data[self.current_step]
        time_remaining = self.max_steps - self.current_step
        return np.array([self.inventory, *data_obs, time_remaining], dtype=np.float32)

    def _get_observation(self):
        # Update history with the latest step’s data
        if self.current_step < self.max_steps:
            self.history = np.roll(self.history, shift=-1, axis=0)
            self.history[-1] = self.day_data[self.current_step]
            
        self.history = np.clip(self.history, -1e6, 1e6)

        # Process history through LSTM model with hidden state
        with torch.no_grad():
            lstm_input = torch.tensor(self.history).unsqueeze(0).float()  # Add batch dimension
            lstm_output, self.hidden_state = self.lstm_model(lstm_input, self.hidden_state)
            lstm_output = lstm_output[:, -1, :]  # Take the last output for sequence processing

        # Flatten the history to include it directly in the observation
        flattened_history = self.history.flatten()

        # Other components of the observation
        time_remaining = self.max_steps - self.current_step
        state = np.array([self.inventory, time_remaining], dtype=np.float32)

        # Concatenate state, flattened history, and LSTM output
        observation = np.concatenate((state, flattened_history, lstm_output.flatten().numpy()), axis=0)
        observation = np.clip(observation, -1e6, 1e6)
        
        return observation

    def step(self, action):
        done = False

        # Clamp action to be within allowed range, ensuring controlled inventory depletion
        action = np.clip(action[0], 0, min(self.inventory, self.initial_inventory / self.share_frac))
        shares_sold = action

        # Execute trade at the bid price
        executed_price = self.day_data[self.current_step][0]  # Assuming bid_price_1 is the first column
        self.inventory -= shares_sold
        reward = self.compute_reward(shares_sold, executed_price)
        reward = float(reward)

        # Move to next time step
        self.current_step += 1

        # Check if the episode is done
        if self.current_step >= self.max_steps or self.inventory <= 0:
            done = True

        # Get the next state
        state = self._get_observation()
        return state, reward, done, {}

    def vwap(self, idx, shares):
        # Retrieve bid prices and sizes for VWAP calculation
        bid_prices = np.array(self.day_data[self.current_step, :5])  # First 5 columns assumed to be bid prices
        bid_sizes = np.array(self.day_data[self.current_step, 5:10])  # Next 5 columns assumed to be bid sizes

        # Calculate cumulative sum for VWAP calculation
        cumsum = 0
        for i, size in enumerate(bid_sizes):
            cumsum += size
            
            if np.any(cumsum >= shares):
                break

        # Compute VWAP for the portion of bids that meet the share count
        denominator = np.sum(bid_sizes[:i + 1])
        if denominator == 0:
            return 0  
        
        return np.sum(bid_prices[:i + 1] * bid_sizes[:i + 1]) / denominator

    def compute_reward(self, shares_sold, executed_price):
        # Calculate VWAP and derive slippage and transaction costs
        actual_price = float(self.vwap(self.current_step, shares_sold))
        slippage = (executed_price - actual_price) * shares_sold
        slippage = np.clip(slippage, -1e6, 1e6)

        # Market impact as a function of shares sold (example scaling factor for impact)
        alpha = 4.439584265535017e-06
        market_impact = alpha * np.sqrt(shares_sold)
        market_impact = np.clip(market_impact, -1e6, 1e6)
        
        # Total transaction cost
        transaction_cost = slippage + market_impact
        
        reward = -float(transaction_cost)
        reward = np.clip(reward, -1e6, 1e6)
        
        return reward

def init_hidden(lstm_model, batch_size=1):
    """Initialize hidden and cell states based on a loaded LSTM model's parameters."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_size = lstm_model.lstm.hidden_size
    num_layers = lstm_model.lstm.num_layers
    hidden_state = (
        torch.zeros(num_layers, batch_size, hidden_size).to(device),
        torch.zeros(num_layers, batch_size, hidden_size).to(device)
    )
    return hidden_state
        
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
    
lstm = torch.load("lstm.pth")
lstm.eval()

# Initialize Environment and Model Parameters
env = TradingEnv(sequences=sequences, lstm=lstm, initial_inventory=1000)
        
class EpisodeSummaryCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super(EpisodeSummaryCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.episode_reward = 0.0
        self.episode_steps = 0

    def _on_step(self):
        # Accumulate rewards and step counts as floats
        reward = float(self.locals.get("rewards", 0))
        self.episode_reward += reward
        self.episode_steps += 1

        # Check if the episode has ended
        if self.locals.get("dones", [False])[0]:
            # Calculate percentage of training completed
            training_progress = (self.num_timesteps / self.total_timesteps) * 100

            # Print episode summary
            print(f"End of Episode - Total Reward: {float(self.episode_reward):.2f}, "f"Steps: {self.episode_steps}, Training Progress: {training_progress:.2f}%")

            # Reset episode statistics
            self.episode_reward = 0.0
            self.episode_steps = 0

        return True

    def _on_training_end(self):
        print("Training complete.")

# Total timesteps for training
total_timesteps = int(1e5)

model = SAC(
    MlpPolicy,
    env,
    gamma=GAMMA,
    tau=TAU,
    learning_rate=LR,
    buffer_size=REPLAY_BUFFER_SIZE,
    batch_size=BATCH_SIZE,
    train_freq=1,
    gradient_steps=-1,
    ent_coef='auto',  # Entropy coefficient tuning
    target_entropy=TARGET_ENTROPY,
    device=device,
    tensorboard_log="./sac_trading_tensorboard/"
)

model.load("sac_trading_model")
    
def evaluate_model_over_days_1(model, env, benchmark, initial_inventory=1000, preferred_timeframe=390):
    """
    Evaluates the trained model over multiple days and computes comparative statistics
    against TWAP and VWAP benchmark strategies.
    """
    daily_results = []
    
    # Iterate through each day in the data sequences
    for day_index, day_data in enumerate(env.sequences):
        env.day_index = day_index
        obs = env.reset()
        done = False
        trade_schedule = []
        cumulative_reward = 0
        steps = 0

        # Run model over the episode (one trading day)
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            cumulative_reward += reward
            steps += 1
            trade_event = {"Timestamp": env.day_data[env.current_step - 1][0], "shares": action[0]}
            trade_schedule.append(trade_event)

        trades_df = pd.DataFrame(trade_schedule)
        
        # Transaction Cost Analysis
        slippage, market_impact = benchmark.simulate_strategy(trades_df, day_data, preferred_timeframe)
        model_total_transaction_cost = sum(slippage) + sum(market_impact)

        # TWAP Transaction Cost
        twap_trades = benchmark.get_twap_trades(day_data, initial_inventory, preferred_timeframe)
        twap_slippage, twap_market_impact = benchmark.simulate_strategy(twap_trades, day_data, preferred_timeframe)
        twap_total_transaction_cost = sum(twap_slippage) + sum(twap_market_impact)

        # VWAP Transaction Cost
        vwap_trades = benchmark.get_vwap_trades(day_data, initial_inventory, preferred_timeframe)
        vwap_slippage, vwap_market_impact = benchmark.simulate_strategy(vwap_trades, day_data, preferred_timeframe)
        vwap_total_transaction_cost = sum(vwap_slippage) + sum(vwap_market_impact)

        final_inventory = env.inventory
        inventory_depletion_rate = (initial_inventory - final_inventory) / initial_inventory
        inventory_remaining_percentage = (final_inventory / initial_inventory) * 100
        trades_df['Timestamp'] = pd.to_datetime(trades_df['Timestamp'], errors='coerce')
        trades_df['Trade Interval'] = trades_df['Timestamp'].diff().dt.total_seconds()
        avg_trade_spacing = trades_df['Trade Interval'].mean() if not trades_df['Trade Interval'].isnull().all() else 0
        trade_count = len(trades_df)

        # Execution Price Comparison
        model_avg_price = trades_df['shares'].dot(
            trades_df['Timestamp'].apply(lambda ts: day_data.loc[day_data['timestamp'] == ts, 'close'].values[0])
        ) / trades_df['shares'].sum()
        twap_avg_price = twap_trades['shares'].dot(twap_trades['price']) / twap_trades['shares'].sum()
        vwap_avg_price = vwap_trades['shares'].dot(vwap_trades['price']) / vwap_trades['shares'].sum()

        daily_results.append({
            "Day Index": day_index,
            "Total Reward": cumulative_reward,
            "Total Steps": steps,
            "Model Total Transaction Cost": model_total_transaction_cost,
            "TWAP Total Transaction Cost": twap_total_transaction_cost,
            "VWAP Total Transaction Cost": vwap_total_transaction_cost,
            "Inventory Remaining Percentage": inventory_remaining_percentage,
            "Inventory Depletion Rate": inventory_depletion_rate,
            "Average Trade Spacing (seconds)": avg_trade_spacing,
            "Trade Count": trade_count,
            "Model Average Execution Price": model_avg_price,
            "TWAP Average Execution Price": twap_avg_price,
            "VWAP Average Execution Price": vwap_avg_price,
        })

    results_df = pd.DataFrame(daily_results)
    summary_stats = results_df.mean().to_frame(name="Mean").join(results_df.std().to_frame(name="StdDev"))
    print("Summary Statistics Across All Days:")
    print(summary_stats)

    return results_df, summary_stats

def evaluate_model_over_days(model, env, benchmark, initial_inventory=1000, preferred_timeframe=390):
    daily_results = []

    # Iterate through each day in the data sequences
    for day_index, day_data in enumerate(env.sequences):
        env.day_index = day_index
        obs = env.reset()
        done = False
        trade_schedule = []
        cumulative_reward = 0
        steps = 0

        # Run model over the episode (one trading day)
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            cumulative_reward += reward
            steps += 1
            trade_event = {"Timestamp": env.day_data[env.current_step - 1][0], "shares": action[0]}
            trade_schedule.append(trade_event)

        trades_df = pd.DataFrame(trade_schedule)

        # Transaction Cost Analysis
        slippage, market_impact = benchmark.simulate_strategy(trades_df, day_data, preferred_timeframe)
        model_total_transaction_cost = sum(slippage) + sum(market_impact)

        # TWAP Transaction Cost
        twap_trades = benchmark.get_twap_trades(day_data, initial_inventory, preferred_timeframe)
        twap_slippage, twap_market_impact = benchmark.simulate_strategy(twap_trades, day_data, preferred_timeframe)
        twap_total_transaction_cost = sum(twap_slippage) + sum(twap_market_impact)

        # VWAP Transaction Cost
        vwap_trades = benchmark.get_vwap_trades(day_data, initial_inventory, preferred_timeframe)
        vwap_slippage, vwap_market_impact = benchmark.simulate_strategy(vwap_trades, day_data, preferred_timeframe)
        vwap_total_transaction_cost = sum(vwap_slippage) + sum(vwap_market_impact)

        # Calculate average execution price using bid prices as proxy
        model_avg_price = trades_df.apply(
            lambda row: row['shares'] * day_data[env.current_step - 1][0],  # assuming bid_price_1 is the first column
            axis=1
        ).sum() / trades_df['shares'].sum()

        # TWAP and VWAP average prices using derived prices in twap_trades and vwap_trades
        twap_avg_price = twap_trades['shares'].dot(twap_trades['price']) / twap_trades['shares'].sum()
        vwap_avg_price = vwap_trades['shares'].dot(vwap_trades['price']) / vwap_trades['shares'].sum()

        final_inventory = env.inventory
        inventory_depletion_rate = (initial_inventory - final_inventory) / initial_inventory
        inventory_remaining_percentage = (final_inventory / initial_inventory) * 100
        trades_df['Timestamp'] = pd.to_datetime(trades_df['Timestamp'], errors='coerce')
        trades_df['Trade Interval'] = trades_df['Timestamp'].diff().dt.total_seconds()
        avg_trade_spacing = trades_df['Trade Interval'].mean() if not trades_df['Trade Interval'].isnull().all() else 0
        trade_count = len(trades_df)

        daily_results.append({
            "Day Index": day_index,
            "Total Reward": cumulative_reward,
            "Total Steps": steps,
            "Model Total Transaction Cost": model_total_transaction_cost,
            "TWAP Total Transaction Cost": twap_total_transaction_cost,
            "VWAP Total Transaction Cost": vwap_total_transaction_cost,
            "Inventory Remaining Percentage": inventory_remaining_percentage,
            "Inventory Depletion Rate": inventory_depletion_rate,
            "Average Trade Spacing (seconds)": avg_trade_spacing,
            "Trade Count": trade_count,
            "Model Average Execution Price": model_avg_price,
            "TWAP Average Execution Price": twap_avg_price,
            "VWAP Average Execution Price": vwap_avg_price,
        })

    results_df = pd.DataFrame(daily_results)
    summary_stats = results_df.mean().to_frame(name="Mean").join(results_df.std().to_frame(name="StdDev"))
    print("Summary Statistics Across All Days:")
    print(summary_stats)

    return results_df, summary_stats

# Run the evaluation
benchmark_data = pd.read_csv('test_set.csv')
benchmark = Benchmark(data=benchmark_data)

daily_results, summary_stats = evaluate_model_over_days(model, env, benchmark)
daily_results.to_csv("daily_results.csv", index=False)
summary_stats.to_csv("summary_statistics.csv")

# Plotting function
def plot_transaction_costs(daily_results):
    plt.figure(figsize=(12, 6))
    plt.plot(daily_results["Day Index"], daily_results["Model Total Transaction Cost"], label="Model", marker='o')
    plt.plot(daily_results["Day Index"], daily_results["TWAP Total Transaction Cost"], label="TWAP", marker='s')
    plt.plot(daily_results["Day Index"], daily_results["VWAP Total Transaction Cost"], label="VWAP", marker='^')
    plt.xlabel("Day Index")
    plt.ylabel("Total Transaction Cost")
    plt.title("Transaction Costs per Day for Model, TWAP, and VWAP Strategies")
    plt.legend()
    plt.grid(True)
    plt.show()

# Call the function to plot
plot_transaction_costs(daily_results)
