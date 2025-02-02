# Project Charter

## Description
The **Lux AI Season 3** competition challenges developers to create an AI agent to tackle a 1v1 multi-agent game in a partially observable environment. The game requires AI agents to adapt to constant environmental change to strategically balance resource management, decision-making under uncertainty, and long-term planning to get an edge over competing agents.

The core problems involve **reinforcement learning (RL)**, **imitation learning**, and **meta-learning**, which help create a model that can be generalized across different scenarios and opponents. This project is significant for various fields, such as **adapting to market changes in real-time in financial modeling**, **adapting to new medical conditions to assist in medical diagnosis**, and more.

## Scope
**This project will focus on:**
  - Developing an **AI agent** capable of competing in Lux AI
  - Implement **reinforcement learning strategies** to optimize agent performance
  - Train agent to **adapt to numerous known and unknown scenerios**
  - Submit the agent on **official leaderboard** for benchmarking
    
**What the project will deliver:**
  - A fully functional **AI agent** capable of competing in the Lux AI Season 3 competition.
  - Interactive **dashboard** for **performance analysis report**
  - **Final submission** for ranking

## Metrics

**Competition Performance:**

| **Metric**                 | **Description**                                      | **Goal** |
|----------------------------|------------------------------------------------------|----------|
| **Win Rate**               | Percentage of games won against random and top-ranked agents. | ≥ 60% against random agents from the submission pool. |
| **Leaderboard Ranking**    | Position on the official leaderboard. | Top 15% of submissions (stretch goal: top 10%). |
| **Skill Rating (μ, σ)** | Measures the agent’s estimated skill level (μ) and uncertainty (σ). | Increase μ while reducing σ over time for ranking stability. |
| **Validation Pass Rate**   | Percentage of submitted bots that pass the self-match validation. | ≥ 95% successful validation rate. |
| **Match Frequency (New Submissions)** | Number of matches played per new bot submission. | Ensure new bots receive enough episodes for quick ranking updates. |
| **Game Length (Turns Survived)** | Measures how long the agent remains competitive in a match. | Maximize survival and long-term sustainability. |

**Model Metrics:**

| **Metric**                 | **Description**                                      | **Goal** |
|----------------------------|------------------------------------------------------|----------|
| **Training Convergence** | Measures how well reinforcement learning (RL) models are learning over time. | Loss function should decrease consistently. |
| **Adaptability Score**     | Measures how well the AI learns new strategies against novel opponents. | Improve performance against unseen strategies. |
| **Computation Time per Move** | Measures AI decision-making speed per turn. | Keep inference time low to ensure fast response. |
| **Model Size & Complexity** | Assesses the computational cost of running the agent. | Optimize for efficiency without sacrificing performance. |
| **Resource Utilization Efficiency** | Measures how well the agent gathers and uses resources. | Minimize wasted resources and optimize usage. |
| **Expansion Rate**         | Tracks how quickly the agent expands. | Ensure strategic expansion without overextending. |
| **Opponent Adaptation Rate** | Measures how quickly the AI adjusts to new opponent strategies. | Improve learning rate over multiple matches. |


## Architecture
There are two primary data sources for this project: environment data and episode replay data. Both contain information on agent positions, environmental hazards, relics, and energy tiles. The environment data resides within the LuxAIGym framework, while episode replay data is stored in JSON files.

Initially, the model will be trained on environment data directly from LuxAIGym. In later stages, the project will leverage the JSON-based episode replay data by parsing it into 2D tensors, which then inform model training and fine-tuning. This offline replay data allows for additional experimentation and analysis beyond the live environment.

The final deliverable will be packaged as a zip file containing all required scripts, dependencies, and configuration files needed to run the model. Users can integrate this package seamlessly into their existing Python environment—whether on a local machine or a cloud-based server—ensuring easy testing.

## Plan
### 1.  Research & Design
Understand the Game Mechanics: Examine Lux AI's rules and objectives, including how agents interact with environment hazards, relics, and energy tiles.
**Data Exploration:**
Review environment data stored in LuxAIGym to understand its structure (e.g., map dimensions, unit attributes).
Inspect JSON-based replay data for details on agent movements, rewards, and actions across different episodes.
Algorithm Selection: Decide on the core reinforcement learning (RL) methods (e.g., PPO, SAC, offline RL, or imitation learning approaches) based on the complexity of the state/action space and available replay data.
Design the Agent Architecture:
Plan the neural network structure (e.g., CNN for grid-based features, MLP for flattened data).
Outline how different components (policy network, value function, replay buffer) fit together within the training pipeline.
### 2.  Development
**Environment Integration:**
Set up a robust interface with LuxAIGym to obtain live environment data (e.g., agent positions, tile information).
Ensure the framework can reset, step through, and render episodes for debugging and testing.

**Replay Data Handling:**
Implement scripts to parse JSON replays into 2D tensors, accommodating agent positions, hazards, relic locations, and energy distribution.
Develop data loaders that can optionally provide offline replay data for model pre-training or fine-tuning.

**Algorithm & Model Implementation:**
Write modular code for RL algorithms (on-policy/off-policy) or imitation learning pipelines (e.g., behavior cloning).
Integrate advanced features if needed (e.g., reward shaping, curriculum learning, or advanced exploration strategies).
### 3. Training & Testing
**Initial Training on Environment Data:***
Run baseline RL training directly within LuxAIGym to validate the agent’s ability to learn from the live environment.
Log intermediate performance (scores, win rates, etc.) and track convergence.

**Utilizing Episode Replay Data (Later Stages):**
Introduce offline training or imitation learning using the JSON replay dataset.
Convert these replays into 2D/3D tensors that mirror the input format used in live training.
Compare performance gains from combining live environment data with replay data.

**Iterative Refinement:**
Perform hyperparameter tuning (learning rate, batch size, reward shaping) to optimize results.
Evaluate different network architectures and data preprocessing techniques (e.g., normalization, embedding tile types).

**Robust Testing Regimen:**
Validate the agent’s performance across diverse scenarios (varying map sizes, hazard densities, and team configurations).
Use automated scripts to run multiple trials and statistically assess improvement over time.
### 4. Evaluation & Submission
**Performance Analysis:**
Conduct final evaluations to measure reliability, average score, and win rates across official or custom test scenarios.
Document any improvements gained by leveraging replay data versus environment-only training.

**Packaging & Deliverables:**
Package the final model—along with all necessary code, dependencies, and configurations—into a zip file.
Provide clear instructions for running model inside LuxAIGym (e.g., Python environment setup, usage examples).


## Personnel

## Communication
- **Slack** for real-time communication. 
- **Jira** for task and project management. 
- **GitHub** for version control and code collaboration. 
- **Regular meetings** to review progress and address challenges.




