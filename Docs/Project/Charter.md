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
**Understand the Game Mechanics:** Examine Lux AI's rules and objectives, including how agents interact with environment hazards, relics, and energy tiles.
**Data Exploration:**
Review environment data stored in LuxAIGym to understand its structure (e.g., map dimensions, unit attributes).
Inspect JSON-based replay data for details on agent movements, rewards, and actions across different episodes.
Algorithm Selection: Decide on the core reinforcement learning (RL) methods (e.g., PPO, SAC, offline RL, or imitation learning approaches) based on the complexity of the state/action space and available replay data.

**Design the Agent Architecture:**
Plan the neural network structure (e.g., CNN for grid-based features, MLP for flattened data).
Outline how different components (policy network, value function, replay buffer) fit together within the training pipeline.

### 2.  Development
**Environment Integration:**
Set up a robust interface with LuxAIGym to obtain live environment data (e.g., agent positions, tile information).
Ensure the framework can reset, step through, and render episodes for debugging and testing.

**Algorithm & Model Implementation:**
Write modular code for RL algorithms (on-policy/off-policy) or imitation learning pipelines (e.g., behavior cloning).
Integrate advanced features if needed (e.g., reward shaping, curriculum learning, or advanced exploration strategies).


### 3. **Training Phase 1 & Testing**

1. **Initial Training on Environment Data**  
   - Perform **baseline RL training** directly within the LuxAIGym environment.  
   - Measure and log **key performance indicators** (e.g., scores, win rates) to track learning progress.  
   - Use these initial results to **validate** the agent’s basic functionality and convergence behavior.

2. **Iterative Refinement**  
   - Conduct **hyperparameter tuning** (learning rate, batch size, reward shaping) to optimize training efficiency.  
   - Experiment with **alternative model architectures** (e.g., CNN vs. MLP) and data preprocessing techniques (normalization, tile-type embeddings).  
   - Maintain a **change log** of each experiment and systematically compare outcomes.

3. **Robust Testing Regimen**  
   - Evaluate the agent’s adaptability to **diverse scenarios**, such as varied map sizes, hazard densities, and team configurations.  
   - Use **automated scripts** to run multiple trials and gather statistically meaningful performance data (e.g., average rewards, standard deviations, confidence intervals).  
   - Document **improvements over time** to inform future training decisions.

### 4. **Training Phase 2 & Testing**

1. **Replay Data Handling**  
   - Implement scripts to **parse JSON replays** into structured formats (e.g., 2D tensors) capturing agent positions, hazards, relic locations, and energy tiles.  
   - Develop **data loaders** capable of batching and feeding this replay data into training pipelines for either **pre-training** or **fine-tuning**.

2. **Utilizing Episode Replay Data**  
   - Introduce **imitation learning** techniques (e.g., behavior cloning) to leverage human or expert-play data for faster convergence and improved decision-making.  
   - **Convert** replay episodes into 2D/3D tensors consistent with the input formats used in live training, ensuring smooth integration.  
   - **Compare performance** of agents trained solely on environment data against those incorporating both environment and replay data.

3. **Reintegration & Retraining in LuxAIGym**  
   - **Retrain** or fine-tune the agent directly in the LuxAIGym environment after incorporating insights or weights from the replay-based models.  
   - Conduct another round of **performance evaluations** to verify improvements when returning to a live training context.  
   - Finalize hyperparameters and data-processing strategies in preparation for **final evaluation** and submission.

### 4. Evaluation & Submission
**Performance Analysis:**
Conduct final evaluations to measure reliability, average score, and win rates across official or custom test scenarios.
Document any improvements gained by leveraging replay data versus environment-only training.

**Packaging & Deliverables:**
Package the final model—along with all necessary code, dependencies, and configurations—into a zip file.
Provide clear instructions for running model inside LuxAIGym (e.g., Python environment setup, usage examples).
### Timeline

| **Sprint 1 - ends 2/17**                                                                                                                                                                                                                           | **Sprint 2 - ends 3/25**                                                                                                                                                                                                                                     | **Sprint 3 - ends 4/28**                                                                                                                                                                                                                           |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Research & Design** <br><br>- **Understand the Game Mechanics**: Examine Lux AI's rules/objectives and how agents interact with environment hazards, relics, and energy tiles. <br>- **Data Exploration**: Review LuxAIGym data (map dimensions, unit attributes) and inspect JSON replays (agent movements, rewards, actions). <br>- **Algorithm Selection**: Choose RL methods (PPO, SAC, offline RL, or imitation learning). <br>- **Design the Agent Architecture**: Plan the neural network structure (CNN vs. MLP) and outline policy/value/replay buffer components. |**Data Manipulation** <br><br>  - **Replay Data Handling**: Parse JSON replays into 2D tensors (positions, hazards, relics). Build loaders for pre-training/fine-tuning.<br><br><br><br><br><br><br><br><br><br><br><br><br><br>| **Evaluation & Submission** <br><br>- **Performance Analysis**: Conduct final evaluations (average scores, reliability, win rates) on official or custom scenarios. Document improvements from replay-based approaches. <br>- **Packaging & Deliverables**: Provide a zip file (code, dependencies, configs) with usage instructions for LuxAIGym, then submit for official ranking.<br><br><br><br><br><br><br> |
| **Development** <br><br>- **Environment Integration**: Set up LuxAIGym interface (reset, step, render) for debugging/testing. <br>- **Algorithm & Model Implementation**: Write modular RL/imitation code; integrate advanced features (reward shaping, curriculum learning). | **Training Phase 2**<br><br> - **Utilizing Episode Replay Data**: Apply imitation learning (behavior cloning), convert replays to consistent inputs, compare environment-only vs. combined data. <br><br><br>                                                                                                                                                                                                                                                  |                                                                                                                                                                                                                                         |
| **Training Phase 1 & Testing** <br><br> - **Initial Training on Environment Data**: Baseline RL training in LuxAIGym; log scores and convergence. <br> - **Iterative Refinement**: Tune hyperparameters, test architectures (CNN vs. MLP), maintain change logs. <br> - **Robust Testing Regimen**: Validate on diverse maps, run multiple trials, document improvements. |**Testing** <br><br> - **Reintegration & Retraining**: Fine-tune the agent with replay insights, re-check performance, finalize hyperparams/data pipelines.<br><br><br><br><br><br><br><br><br>                                                                                                                                                                                                                                                  |                                                                                                                                                                                                                                         |





## Personnel
- Quang Nguyen
- Brandon Hugger
- Jie Huang
- Jihyeon Jeong

## Communication
- **Slack** for real-time communication. 
- **Jira** for task and project management. 
- **GitHub** for version control and code collaboration. 
- **Regular meetings** to review progress and address challenges.




