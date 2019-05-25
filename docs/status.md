---
layout:  default
title:    Status
---

## Project summary:

In this project, our goal is to let agent protect villagers and kill zombies. We do this by using deep reinforcement learning,
which is a combination of deep learning and reinforcement learning. By randomly choosing actions at the beginning, agent will 
learn from the algorithm and start to choose the best action or move it should do to accomplish our goal. 

## Approach: 

Following is the whole process diagram of our algorithm:


<i>Algorithm:<em>

To be specific, we use DQN(Deep Q-Network) as our main algorithm. 
The key point of this algorithm is to fit the original q-value function by using a neural network, which is: 

                                               <i>Q(s,a,w)~q(s,a)<em>
