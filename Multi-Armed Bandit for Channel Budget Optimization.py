import React, { useState, useEffect } from 'react';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ComposedChart, Area } from 'recharts';
import _ from 'lodash';

// Channel data structure with parameters for ROI distribution
const CHANNELS = [
  { name: 'Search', meanROI: 2.8, stdDev: 0.6, isNewAge: false, color: '#1f77b4' },
  { name: 'Social Media', meanROI: 2.5, stdDev: 0.8, isNewAge: false, color: '#ff7f0e' },
  { name: 'Display', meanROI: 1.8, stdDev: 0.4, isNewAge: false, color: '#2ca02c' },
  { name: 'Email', meanROI: 3.2, stdDev: 0.5, isNewAge: false, color: '#d62728' },
  { name: 'Influencer', meanROI: 2.7, stdDev: 1.2, isNewAge: true, color: '#9467bd' },
  { name: 'Podcast', meanROI: 2.4, stdDev: 0.9, isNewAge: true, color: '#8c564b' },
  { name: 'AR/VR', meanROI: 2.2, stdDev: 1.5, isNewAge: true, color: '#e377c2' },
  { name: 'Metaverse', meanROI: 1.9, stdDev: 1.8, isNewAge: true, color: '#7f7f7f' },
];

// Function to simulate ROI for a channel based on its parameters
const simulateChannelROI = (channel) => {
  // Box-Muller transform for normal distribution
  const u1 = Math.random();
  const u2 = Math.random();
  const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  
  // Generate ROI from normal distribution with channel parameters
  return channel.meanROI + channel.stdDev * z0;
};

const MultiArmedBanditOptimizer = () => {
  // State for simulation
  const [epsilon, setEpsilon] = useState(0.2);
  const [totalBudget, setTotalBudget] = useState(100000);
  const [iterations, setIterations] = useState(50);
  const [runningSimulation, setRunningSimulation] = useState(false);
  const [currentIteration, setCurrentIteration] = useState(0);
  
  // State for results
  const [channelStats, setChannelStats] = useState([]);
  const [allocationHistory, setAllocationHistory] = useState([]);
  const [rewardHistory, setRewardHistory] = useState([]);
  const [cumulativeReward, setCumulativeReward] = useState(0);
  
  // Initialize channel statistics
  useEffect(() => {
    const initialStats = CHANNELS.map(channel => ({
      ...channel,
      totalReward: 0,
      totalAllocations: 0,
      averageReward: 0,
      currentAllocation: totalBudget / CHANNELS.length,
    }));
    
    setChannelStats(initialStats);
    setAllocationHistory([]);
    setRewardHistory([]);
    setCumulativeReward(0);
    setCurrentIteration(0);
  }, [totalBudget, epsilon]);
  
  // Epsilon-greedy allocation strategy
  const epsilonGreedyAllocation = () => {
    const newStats = [...channelStats];
    let newAllocation = {};
    let iterationReward = 0;
    
    // Initialize allocation record for this iteration
    CHANNELS.forEach(channel => {
      newAllocation[channel.name] = 0;
    });
    
    // Decide between exploration and exploitation
    if (Math.random() < epsilon || currentIteration === 0) {
      // Exploration: random allocation
      const randomAllocation = _.mapValues(newAllocation, () => totalBudget / CHANNELS.length);
      newAllocation = randomAllocation;
    } else {
      // Exploitation: allocate based on best performing channels
      const totalAvgReward = newStats.reduce((sum, ch) => sum + Math.max(0.1, ch.averageReward), 0);
      
      newAllocation = _.mapValues(newAllocation, (_, channelName) => {
        const channel = newStats.find(ch => ch.name === channelName);
        const channelWeight = Math.max(0.1, channel.averageReward) / totalAvgReward;
        return channelWeight * totalBudget;
      });
    }
    
    // Simulate returns for each channel based on allocation
    newStats.forEach((channel, index) => {
      const allocation = newAllocation[channel.name];
      const roi = simulateChannelROI(channel);
      const reward = allocation * roi;
      
      // Update channel statistics
      newStats[index] = {
        ...channel,
        totalReward: channel.totalReward + reward,
        totalAllocations: channel.totalAllocations + allocation,
        averageReward: channel.totalAllocations === 0 ? 0 : 
          (channel.totalReward + reward) / (channel.totalAllocations + allocation),
        currentAllocation: allocation,
      };
      
      iterationReward += reward;
    });
    
    // Update state with new statistics
    setChannelStats(newStats);
    
    // Add to history
    const allocHistoryEntry = {
      iteration: currentIteration + 1,
      ..._.mapValues(newAllocation, v => v)
    };
    
    setAllocationHistory(prev => [...prev, allocHistoryEntry]);
    
    const rewardEntry = {
      iteration: currentIteration + 1,
      reward: iterationReward,
      cumulativeReward: cumulativeReward + iterationReward
    };
    
    setRewardHistory(prev => [...prev, rewardEntry]);
    setCumulativeReward(prev => prev + iterationReward);
    
    // Increment iteration counter
    setCurrentIteration(prev => prev + 1);
    
    return newStats;
  };
  
  // Run the simulation
  useEffect(() => {
    if (runningSimulation && currentIteration < iterations) {
      const timer = setTimeout(() => {
        epsilonGreedyAllocation();
      }, 300); // Slow down for visualization
      
      return () => clearTimeout(timer);
    } else if (currentIteration >= iterations) {
      setRunningSimulation(false);
    }
  }, [runningSimulation, currentIteration, iterations]);
  
  // Generate allocation data for charts
  const generateAllocationData = () => {
    return channelStats.map(channel => ({
      name: channel.name,
      allocation: channel.currentAllocation,
      averageROI: channel.averageReward / (channel.currentAllocation || 1),
      isNewAge: channel.isNewAge,
      color: channel.color
    }));
  };
  
  // Handle simulation start
  const startSimulation = () => {
    // Reset state
    const initialStats = CHANNELS.map(channel => ({
      ...channel,
      totalReward: 0,
      totalAllocations: 0,
      averageReward: 0,
      currentAllocation: totalBudget / CHANNELS.length,
    }));
    
    setChannelStats(initialStats);
    setAllocationHistory([]);
    setRewardHistory([]);
    setCumulativeReward(0);
    setCurrentIteration(0);
    setRunningSimulation(true);
  };
  
  // Format for currency display
  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      maximumFractionDigits: 0,
    }).format(value);
  };
  
  // Calculate ROI 
  const calculateTotalROI = () => {
    if (cumulativeReward === 0) return 0;
    return cumulativeReward / (totalBudget * currentIteration);
  };
  
  return (
    <div className="p-4 bg-gray-50 min-h-screen">
      <div className="max-w-6xl mx-auto bg-white p-6 rounded-lg shadow-md">
        <h1 className="text-2xl font-bold mb-6 text-center">Multi-Armed Bandit for Channel Budget Optimization</h1>
        
        {/* Controls */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div className="p-4 border rounded-lg">
            <label className="block text-sm font-medium text-gray-700 mb-1">Exploration Rate (ε)</label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={epsilon}
              onChange={(e) => setEpsilon(parseFloat(e.target.value))}
              className="w-full"
            />
            <div className="text-center mt-1">{epsilon.toFixed(2)}</div>
          </div>
          
          <div className="p-4 border rounded-lg">
            <label className="block text-sm font-medium text-gray-700 mb-1">Total Budget</label>
            <input
              type="range"
              min="10000"
              max="1000000"
              step="10000"
              value={totalBudget}
              onChange={(e) => setTotalBudget(parseInt(e.target.value))}
              className="w-full"
            />
            <div className="text-center mt-1">{formatCurrency(totalBudget)}</div>
          </div>
          
          <div className="p-4 border rounded-lg">
            <label className="block text-sm font-medium text-gray-700 mb-1">Iterations</label>
            <input
              type="range"
              min="10"
              max="100"
              step="5"
              value={iterations}
              onChange={(e) => setIterations(parseInt(e.target.value))}
              className="w-full"
            />
            <div className="text-center mt-1">{iterations}</div>
          </div>
        </div>
        
        <div className="flex justify-center mb-6">
          <button
            onClick={startSimulation}
            disabled={runningSimulation}
            className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-blue-300"
          >
            {runningSimulation ? 'Simulation Running...' : 'Start Simulation'}
          </button>
        </div>
        
        {/* Progress and Results */}
        <div className="mb-6 p-4 border rounded-lg">
          <div className="grid grid-cols-3 gap-4">
            <div className="text-center">
              <p className="text-sm text-gray-500">Iteration</p>
              <p className="text-xl font-bold">{currentIteration} / {iterations}</p>
            </div>
            <div className="text-center">
              <p className="text-sm text-gray-500">Cumulative Return</p>
              <p className="text-xl font-bold">{formatCurrency(cumulativeReward)}</p>
            </div>
            <div className="text-center">
              <p className="text-sm text-gray-500">Average ROI</p>
              <p className="text-xl font-bold">{(calculateTotalROI() * 100).toFixed(2)}%</p>
            </div>
          </div>
        </div>
        
        {/* Allocation Chart */}
        <div className="mb-8">
          <h2 className="text-lg font-semibold mb-4">Current Budget Allocation</h2>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={generateAllocationData()} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" tickFormatter={formatCurrency} />
                <YAxis dataKey="name" type="category" width={100} />
                <Tooltip 
                  formatter={(value) => [formatCurrency(value), "Allocation"]}
                  labelFormatter={(label) => `Channel: ${label}`}
                />
                <Legend />
                <Bar dataKey="allocation" name="Budget Allocation" fill="#8884d8">
                  {generateAllocationData().map((entry, index) => (
                    <rect key={`rect-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        
        {/* ROI Chart */}
        <div className="mb-8">
          <h2 className="text-lg font-semibold mb-4">Channel Performance (ROI)</h2>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={generateAllocationData()} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" domain={[0, 5]} />
                <YAxis dataKey="name" type="category" width={100} />
                <Tooltip 
                  formatter={(value) => [`${value.toFixed(2)}x`, "ROI"]}
                  labelFormatter={(label) => `Channel: ${label}`}
                />
                <Legend />
                <Bar dataKey="averageROI" name="Average ROI" fill="#82ca9d">
                  {generateAllocationData().map((entry, index) => (
                    <rect key={`rect-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        
        {/* Allocation History Chart */}
        {allocationHistory.length > 0 && (
          <div className="mb-8">
            <h2 className="text-lg font-semibold mb-4">Budget Allocation Over Time</h2>
            <div className="h-96">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={allocationHistory}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="iteration" />
                  <YAxis tickFormatter={formatCurrency} />
                  <Tooltip formatter={(value) => [formatCurrency(value), ""]} />
                  <Legend />
                  {CHANNELS.map(channel => (
                    <Area 
                      key={channel.name}
                      type="monotone" 
                      dataKey={channel.name} 
                      stackId="1" 
                      fill={channel.color} 
                      stroke={channel.color}
                      name={channel.name}
                    />
                  ))}
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
        
        {/* Reward History Chart */}
        {rewardHistory.length > 0 && (
          <div className="mb-8">
            <h2 className="text-lg font-semibold mb-4">Cumulative Return Over Time</h2>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={rewardHistory}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="iteration" />
                  <YAxis tickFormatter={formatCurrency} />
                  <Tooltip formatter={(value) => [formatCurrency(value), ""]} />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="cumulativeReward" 
                    name="Cumulative Return" 
                    stroke="#8884d8" 
                    dot={false}
                    strokeWidth={2}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="reward" 
                    name="Return per Iteration" 
                    stroke="#82ca9d"
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
        
        <div className="p-4 bg-blue-50 rounded-lg mt-4">
          <h3 className="text-lg font-semibold mb-2">How This Works</h3>
          <ul className="list-disc pl-5 space-y-2">
            <li><strong>Epsilon-Greedy Algorithm:</strong> Balances exploration (trying all channels) and exploitation (focusing on high-performing channels).</li>
            <li><strong>Exploration Rate (ε):</strong> Higher values encourage more experimentation with different channels.</li>
            <li><strong>Channel ROI:</strong> Each channel has a different expected return and volatility.</li>
            <li><strong>New Age Channels:</strong> Channels like AR/VR, Metaverse, etc. typically have higher volatility but potential for high returns.</li>
            <li><strong>Budget Allocation:</strong> As the algorithm learns, it optimizes allocation toward better-performing channels.</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default MultiArmedBanditOptimizer;
