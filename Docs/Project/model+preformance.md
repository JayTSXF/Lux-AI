# Model + Performance Report

## Environment Overview

The Lux AI Season 3 challenge unfolds on a forty‑square grid in which two independent teams of agents vie for control and resources over a fixed number of turns. At every step a procedurally generated map presents varied terrain features such as impassable asteroids, shifting nebulae, energy nodes that refill unit batteries, and hidden relic nodes that reward strategic positioning. Each agent sees only what falls within its sensor range, so fog of war limits global awareness and forces local decision making. Observations consist of six separate layers encoding terrain elements, resource fields, and both friendly and enemy unit locations, alongside a fixed array of per‑unit state vectors that record each unit’s normalized coordinates, energy level, and alive status. Actions require choosing a move direction or allocating energy to sap an opponent within a defined radius, so each turn involves selecting one of five movement options or one of many sap target offsets for every active unit.

## Training Methodology
To train effective policies in this complex multi‑agent environment we employ Proximal Policy Optimization (PPO), which learns directly from the latest interactions without storing past data in a replay memory. We run many self‑play matches in parallel to gather thousands of state‑action‑reward trajectories each update. Generalized advantage estimation smooths reward signals across time by combining discounted returns with value predictions so that policy and value networks receive stable training targets. A clipped objective prevents policy updates that would shift action probabilities too far in one step, while an entropy term encourages continued exploration of novel maneuvers. We typically collect two thousand environment steps per update then perform multiple gradient iterations using small mini‑batches to maximize sample efficiency. Learning‑rate scheduling, adaptive clipping parameters, and balanced loss weighting help maintain steady progress even as opponents evolve their strategies.

## Feature Engineering
Before feeding observations into our neural networks we transform the raw game state into a uniform feature vector for each player. We first flatten the six float‑valued map layers—each scaled between zero and one—so that terrain and resource intensities appear on the same numerical scale. Next we append a fixed‑length array of unit features where each entry holds the unit’s normalized x coordinate, normalized y coordinate, normalized remaining energy, and a binary alive indicator. To reward exploration we include a scalar count of newly revealed tiles for that turn and to guide strategic focus we append the current score difference between teams. All components are concatenated in a consistent order so that the policy network always receives the same input structure. This disciplined feature engineering ensures that spatial patterns and unit‑level details are both available to the network at every decision point.

---

## Training Metrics Overview

**Performance Report: PPO Training Latest Run**  
**Training Duration:** 5 Million Steps (~8.6 hrs)

| Metric                  | Start           | Midpoint (≈2.5 M) | End (5 M)   | Trend                                                       |
|-------------------------|-----------------|-------------------|-------------|-------------------------------------------------------------|
| **Value Loss**          | ≈ 0 (warm‑up)    | ≈ 390             | ≈ 450       | Upward drift with spikes (critic underfitting)              |
| **Policy Gradient Loss**| –0.033          | –0.031            | –0.030      | Moderate ascent toward zero (fine‑tuning phase)             |
| **Total Loss**          | ≈ 0–50          | ≈ 200–300         | ≈ 80–150    | Oscillatory (reflects alternating updates)                  |
| **Explained Variance**  | 0.75            | 0.89              | 0.88        | Rapid rise then stable high predictive power                |
| **Clip Fraction**       | 0.14            | 0.09–0.10         | 0.14        | U‑shaped curve (initial large then moderate, late uptick)   |
| **Approx. KL Divergence**| 0.012          | 0.010             | 0.012       | Maintained near 0.01 (trust‑region honored)                 |
| **Episode Reward Mean** | –100            | –80               | –75         | Gradual improvement despite variance                        |

---

### Value Loss Analysis
![1](https://github.com/user-attachments/assets/e52cc121-2c60-4211-b95d-d7ad53474e8b)

Over the course of training, the value‑function loss steadily climbs from near zero to peaks around 450 by the 5 M‑step mark. Early on (before 500 k steps), the critic learns from scratch and struggles to predict returns accurately. Between 500 k and 1.5 M steps, oscillations in the 300–380 range reflect the critic adjusting to the changing policy. From 1.5 M to 4 M steps, the trend is upward with spikes, suggesting the policy explores new behaviors that temporarily increase error until the critic adapts. In the final million steps, value loss reaches its highest levels, likely because the policy generates more extreme returns that the critic must fit.

### Policy Gradient Loss Analysis
![2](https://github.com/user-attachments/assets/8661fd7b-c0b2-4d54-8ae3-70c97f6f0376)

The policy gradient loss starts around –0.033 and moves upward toward –0.030 by 5 M steps. Initially, it deepens (more negative) until about 700 k steps, indicating stronger policy updates. After that, it plateaus and oscillates modestly, showing that updates are smoothing out as the agent converges. In the last phase, the loss moves slightly toward zero (less negative), suggesting diminishing improvement steps typical of nearing convergence.

### Total Loss Analysis
![3](https://github.com/user-attachments/assets/4126f470-e174-4e2c-b1ca-492c8b9c6fd9)

The total loss aggregates policy, value, and entropy losses. Early spikes near 150 at 1 M steps reflect large initial errors. From 1 M to 2 M steps, it dips toward zero as networks learn. Post‑2 M steps, loss climbs above 300 around 3 M, indicating alternating exploration and critic readjustment phases. It then oscillates before settling near 100 at 5 M steps. This pattern shows robustness, but smoothing via loss weighting could improve stability.

### Explained Variance Analysis
![4](https://github.com/user-attachments/assets/a1d15a86-64a4-4007-9b23-4146a4bbc402)

Explained variance measures how well the critic predicts returns. Starting at 0.75 around 500 k steps, it climbs to 0.89 by 1 M steps, then remains stable around 0.88–0.90. A slight dip toward 0.87 near 5 M steps suggests minor challenges in fitting late‑stage policy nuances.

### Clip Fraction Analysis
![5](https://github.com/user-attachments/assets/078a73c2-8cc1-42a0-82c0-ca862489c8d5)

The clip fraction tracks how often policy updates exceed the clipping threshold. Early in training it is high (~0.14), falling to ~0.09 by 1 M steps as updates become more conservative. Between 1 M and 3.5 M steps it stays around 0.09–0.10, then rises again to ~0.14 in the final phase, indicating late‑stage policy refinements that occasionally push the boundary.

### Approximate KL Divergence Analysis
![6](https://github.com/user-attachments/assets/993f95d4-84c7-4355-931b-639502752631)

KL divergence remained near the PPO target (≈ 0.01) throughout (0.009–0.012), confirming that policy changes respected the trust‑region constraint without over‑stepping.

### Episode Reward Mean Analysis
![7](https://github.com/user-attachments/assets/59239d33-2424-4584-b603-33cee2c3a26b)

The average episode reward starts around –100 in early rollouts, reflecting random behavior. It climbs to –90 to –80 between 500 k and 1.5 M steps, reaches peaks near –70 around 2.5 M steps, and settles around –75 to –80 at 5 M steps. This non‑monotonic improvement reflects exploration and environment stochasticity but demonstrates substantive learning.

---

## Conclusion
These metrics suggest that while the critic occasionally lagged behind evolving policies, it maintained high explained variance and reliable value estimates. Policy updates were neither too timid nor too aggressive, and the system stayed stable. The steady rise in episode returns implies that extending training beyond 5 M steps and refining reward shaping or curriculum learning could yield further gains.

## Game Performance
![8](https://github.com/user-attachments/assets/6fd9f07a-a6c7-497e-99ca-56f49f9cc128)

In head‑to‑head evaluations against the established third‑place baseline our PPO agent won two out of three matches, demonstrating improved strategic coordination and resource management. In the first match our model efficiently captured relic nodes under fog of war and sustained unit energy levels. In the second it displayed superior defense by sapping opponent units at critical moments. These victories reflect both sharper action selection and more accurate value estimation, as shown by rising episode returns and reduced policy gradient volatility. Outperforming a top‑tier solution confirms the robustness and strategic depth of our approach, motivating larger‑scale evaluations and fine‑tuning in future work.
