# Multi-Armed Bandit for Marketing Channel Budget Optimization

This repository contains implementations of a multi-armed bandit algorithm to optimize marketing budget allocation across various channels, including both traditional and new age digital marketing channels.

![Channel Optimization Dashboard](dashboard_preview.png)

## 🚀 Features

- **Epsilon-greedy multi-armed bandit algorithm** for balancing exploration and exploitation
- **Dynamic budget allocation** that adapts to channel performance
- **Simulated ROI modeling** with different mean returns and volatility for each channel
- **Support for both traditional and emerging channels**
- **Implementations in both React (interactive dashboard) and Python (data analysis)**
- **Comprehensive visualizations** to track performance over time

## 📊 Channel Types

The simulation includes various marketing channels with different ROI characteristics:

**Traditional Channels:**
- Search
- Social Media
- Display
- Email

**New Age Channels:**
- Influencer Marketing
- Podcast Advertising
- AR/VR Experiences
- Metaverse Marketing

## 🛠️ Installation

### React Dashboard

```bash
# Clone the repository
git clone https://github.com/viznuv/Channel-budgeting-random-armed-bandit.git
cd Channel-budgeting-random-armed-bandit

# Install dependencies
npm install

# Start the development server
npm start
```

### Python Implementation

```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt

# Run the simulation
python channel_optimization.py
```

## 📚 How It Works

The multi-armed bandit algorithm works on the principle of balancing exploration (trying different channels to learn about their performance) and exploitation (allocating more budget to channels that are known to perform well).

### Key Components:

1. **Epsilon-Greedy Strategy**:
   - With probability ε, allocate budget randomly (exploration)
   - With probability 1-ε, allocate based on historical performance (exploitation)

2. **Channel ROI Simulation**:
   - Each channel has a different mean ROI and standard deviation
   - Returns are simulated using a normal distribution

3. **Budget Allocation**:
   - Budget is proportionally allocated based on historical performance during exploitation
   - Equal allocation during exploration phases

4. **Performance Tracking**:
   - Algorithm tracks cumulative returns, average ROI, and allocation history
   - Visualization of performance metrics over time

## 🔍 Usage

### React Dashboard

The interactive dashboard allows you to:

- Adjust the exploration rate (ε) to control the exploration-exploitation tradeoff
- Set the total marketing budget
- Choose the number of simulation iterations
- View real-time updates of budget allocation and performance
- Analyze the cumulative returns and ROI over time

### Python Implementation

The Python script provides:

```python
# Run a basic simulation with default parameters
python channel_optimization.py

# Customize parameters
python channel_optimization.py --budget 200000 --epsilon 0.15 --iterations 100
```

## 📈 Example Results

After running the simulation:

1. **Channel Allocation**: The algorithm typically converges to allocate more budget to high-performing channels while still maintaining some exploration.

2. **ROI Optimization**: The overall ROI tends to improve over time as the algorithm learns which channels perform best.

3. **Risk Management**: By maintaining some level of exploration, the algorithm helps discover emerging opportunities in new channels.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
