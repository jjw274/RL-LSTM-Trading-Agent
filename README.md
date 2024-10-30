# RL-LSTM-Trading-Agent
A reinforcement learning-based trading agent using Soft Actor-Critic (SAC) with an LSTM model for intraday trade scheduling and transaction cost minimization. 

## Features
- **Soft Actor-Critic (SAC)**: A state-of-the-art reinforcement learning algorithm designed for continuous action spaces.
- **LSTM for Temporal Modeling**: An LSTM network captures sequential dependencies in high-frequency trading data.
- **Custom Trading Environment**: Simulates market conditions, transaction costs, and inventory depletion dynamics.
- **Benchmark Comparisons**: Evaluates performance against TWAP and VWAP, providing insights into cost efficiency and execution quality.
- **Detailed Evaluation and Visualization**: Generates daily performance statistics and plots for comprehensive analysis.

## Repository Structure
```
├── data/                           # Contains sample market data files
├── models/                         # Contains saved models (SAC, LSTM)
├── results/                        # Generated evaluation results and plots
├── src/                            # Source code for the trading agent and environment
│   ├── sac.py                      # SAC training and agent setup
│   ├── lstm.py                     # LSTM model for sequence modeling
│   ├── trading_env.py              # Custom Gym trading environment
│   ├── eval.py                     # Evaluation script for model performance
│   └── benchmark.py                # TWAP and VWAP benchmark calculations
└── README.md                       # Project documentation
```

## Sample Results

Results include transaction cost comparisons across different strategies:
- **Model Total Transaction Cost**: Lower than TWAP and VWAP, showing cost efficiency.
- **Average Execution Price**: Comparable to traditional benchmarks, indicating effective trade timing.

![Transaction Costs Comparison](results/comparison.png)

## License
This project is licensed under the MIT License.

## Acknowledgments
This project was inspired by research on reinforcement learning for finance, including works by Gordon Ritter, Marcos Lopez de Prado, and Giuseppe Paleologo. This was prepared for a Blockhouse research trial.
