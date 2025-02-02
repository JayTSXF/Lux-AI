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
There are two primary data sources for this project: environment data and episode replay data. Both include information about agent positions, environmental hazards, relics, and energy tiles. The environment data is hosted within the LuxAIGym environment itself, while episode replay data is stored in JSON files. These JSON files can be parsed and processed by our scripts to train the model but will only be utilized in the later stages of the project. The final deliverable will be a zip file containing all necessary functions and configurations to run the model within LuxAIGym. Users can integrate this zip file directly with their existing Python environment 

## Plan
1. **Research & Design:** Study game mechanics and design the AI agent. 
2. **Development:** Implement algorithms and reinforcement learning strategies. 
3. **Training & Testing :** Train and refine the agent’s performance across different scenarios. 
4. **Evaluation & Submission:** Finalize performance analysis, prepare submission, and compete. 

## Personnel

## Communication
- **Slack** for real-time communication. 
- **Jira** for task and project management. 
- **GitHub** for version control and code collaboration. 
- **Regular meetings** to review progress and address challenges.




