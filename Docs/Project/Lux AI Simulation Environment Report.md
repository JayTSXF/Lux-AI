# Lux AI Simulation Environment Report

Lux AI is a reinforcement learning competition that focuses on developing agents to tackle a diverse set of problems, including energy management, relic collection, and enemy neutralization. The competition takes place on a 24 × 24 grid, which changes with each time step of the environment. Agents must compete against opposing agents to achieve the highest score in a given match.

To tackle this problem, we must categorize each parameter in the environment into one of two groups: **state space** or **action space**. The state space describes the number of possible states (or the entropy) of the environment, whereas the action space describes the number of actions an agent can take within the environment.

Since asteroids, nebula fields, relic nodes, energy tiles, and agents are continuously moving with randomized parameters at each time step, the number of states in the environment is arbitrarily large. Therefore, we categorize the environment as a **continuous state space**. Agents, however, have only six possible movements: up, down, left, right, center, and sap. Thus, the action space for this competition is **discrete**, with a size of six.

---

## Features Within the Environment

### Spatial and Positional Features
Each unit operates on a 2D grid, where its own (x, y) position is a fundamental feature. These are typically encoded either as raw coordinates or one-hot position matrices. Additionally, the positions of teammates, enemies, and landmarks (e.g., relic nodes or energy nodes) are all included in the agent’s state, often masked by vision limitations.

### Vision Mask and Local Observability
Vision is implemented as a Chebyshev-distance-based mask centered on each unit. This yields a localized spatial awareness feature map, where the agent can observe tile types, enemy units, and relics within its sensor range. The output of this process is a binary or weighted matrix representing the agent’s sensor mask, combined across all team units to produce a global visibility map. Vision is an essential input for partial observability modeling, requiring the agent to learn to explore and maximize information gain.

### Terrain and Environmental Features
Every tile in the grid map is characterized by a tile type feature:

- `0`: Empty (normal traversable space)  
- `1`: Nebula (reduces sensor strength)  
- `2`: Asteroid (blocks both vision and movement)  

These categorical features are embedded into matrix form (e.g., 2D grid with values representing tile types) and are used in spatial convolutional layers or attention mechanisms in RL models. The tile type also affects downstream rewards and pathfinding heuristics.

### Energy-Related Features
Each unit has an internal energy state, represented as a scalar feature. Energy is consumed when performing actions such as movement or sapping. Units must manage this resource, with energy regeneration features tied to the presence and positions of energy nodes on the map. This introduces temporal dependencies, encouraging agents to balance aggression and conservation.

### Relic Node Configuration Features
Relic nodes are key scoring elements on the map. Their positions are observable, but the scoring logic involves a hidden 5x5 mask centered on each relic node. Only certain tiles in that region yield team points when occupied. This creates a layer of latent reward features that are not immediately visible and must be inferred through trial-and-error, which is ideal for exploration-based learning and uncertainty modeling.

### Action Encoding and Behavior Modeling
Each unit can take one of six discrete actions per turn, such as movement in cardinal directions, staying still, or executing a sap command. These are encoded as categorical action spaces and used in policy networks for output prediction. The effects of these actions depend on environmental context, particularly the proximity to enemies, relic nodes, and obstacles.

### Enemy State and Opponent Modeling
The positions and energy levels of enemy units are included as features, but only if within vision. This introduces partial state observability, where agents must learn to track enemy behavior using limited data. Historical patterns and prediction of unseen states can be incorporated to improve opponent modeling.

### Combined Team State and Multi-Agent Awareness
For teams with multiple units, shared observation features such as team sensor mask, team scores, and unit distribution are provided. These features enable cooperative behavior and allow models to learn coordinated strategies such as coverage, clustering, or role division. In structured models, this can lead to emergent behaviors like flanking or area denial.

### Temporal Dynamics and Map Shifts
At randomized intervals during each match, the map undergoes a horizontal shift, displacing tiles, relic nodes, and potentially enemy units. The interval `X` is a scalar value randomly determined at the start of the game. This mechanic introduces nonstationarity into the environment. It challenges agents to develop robust and adaptable strategies, as the spatial layout can change unexpectedly, requiring dynamic reassessment of unit positioning and objectives.

### Score and Reward Feedback
The agent receives scalar feedback through the reward signal, which is shaped by a combination of:

- Exploration bonuses (e.g., discovering new tiles)  
- Relic scoring (based on correct tile occupancy)  
- Sap-based combat outcomes  
- Positional penalties (e.g., being blocked or clustered)  
- End-of-turn point gains  

These elements form the reward shaping features, which heavily influence policy learning and value estimation.

---

## Environment Simulation and PPO Integration

### Data Collection using Rollouts
During training, the PPO algorithm relies on a process known as data collection through rollouts. To efficiently gather a diverse and representative dataset of experiences, the training script initiates multiple parallel instances of the environment. Each environment runs independently in its own process, allowing several agents to simultaneously interact with their environment copies. These interactions generate a continuous stream of transitions, which include the current state, the action taken, the resulting reward, the next state, and an indicator of whether the episode has terminated. This parallelization significantly speeds up experience collection and provides the learning algorithm with a more varied sampling of the environment's dynamics.

To support analysis and performance tracking, each environment is wrapped in a monitoring tool known as the monitor wrapper. This wrapper records key metrics throughout training and provides real-time feedback on agent performance. Some of the primary statistics captured include the mean episode reward, which reflects how well the agent is performing over time, and the entropy loss, which measures the randomness in the policy's action selection. Additionally, the monitor tracks value loss, which quantifies how accurately the value network is predicting expected returns, and the KL divergence, which measures the change between the old and new policy after updates. These metrics are essential for diagnosing learning behavior and ensuring that the agent is making stable, meaningful progress throughout training.

### Observation Transformation
In this setup, the environment returns observations as dictionaries. The custom wrapper converts dictionary-based observation and action spaces into flat arrays. If the model’s policy expects a particular observation format, this transformation ensures compatibility with the policy architecture. This step is crucial for integrating modular or custom neural networks into the learning pipeline.

### PPO Updates
The Proximal Policy Optimization (PPO) algorithm, as implemented in Stable Baselines3, begins by collecting data from its interactions with the environment. This data consists of transitions, typically stored as tuples containing the current state, the action taken, the reward received, the next state, and a boolean indicating whether the episode has ended. These transitions are gathered over a fixed rollout period, which is a set number of time steps during which the agent samples experiences by following its current policy. The quality and diversity of these samples are critical for learning, as they reflect how the current version of the policy behaves under real environment conditions.

After the rollout period, PPO processes the collected experiences to calculate two key components: discounted rewards and advantage estimates. Discounted rewards are computed using a discount factor (commonly denoted as gamma) to give more weight to immediate rewards while still accounting for future outcomes. The advantage estimate quantifies how much better or worse an action performed compared to the expected value of that state, providing directional feedback for policy updates. PPO uses Generalized Advantage Estimation (GAE) to calculate these advantages in a way that balances bias and variance. These calculations help guide the learning process, allowing the agent to adjust its behavior based on which actions led to favorable results.

To further enhance training stability and efficiency, PPO includes a dynamic entropy schedule to manage exploration. Entropy, in this context, refers to the randomness of the policy's action selection. At the beginning of training, the entropy coefficient is kept high to encourage the agent to explore a wide range of possible actions and discover effective strategies. Over time, as the policy improves, the entropy is gradually reduced, allowing the agent to exploit its knowledge and behave more deterministically. This adaptive scheduling ensures that the agent does not prematurely converge on suboptimal behaviors, striking a balance between trying new things and refining existing strategies.

Finally, using the processed rollout data, PPO performs gradient descent updates on two neural networks: The policy network, which determines the agent's action distribution and the value network, which estimates the expected future reward from each state. The policy is updated using a clipped surrogate objective, which is a core innovation of PPO.  Instead of allowing large, potentially destabilizing updates, PPO clips the policy  ratio, which means the change in probability between the old and new policy to stay within a safe range (e.g., 0.8 to 1.2). This prevents the policy from moving too far in a single update step and ensures monotonic improvement, resulting in a more stable and reliable training process.

