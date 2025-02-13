---
layout:  default
title:    Proposal
---

## Summary of the Project

The main idea of the project is to kill random spawning zombies in order to minimize the loss of villagers. By performing enforcement learning on the agent, the agent is able to learn/identify the nearest threat to a villager and eliminate the threat as quick as possible.  

The goal of the project is to let the agent protects as many villagers as possible by performing actions such as moving to and attacking zombies. 

Input information: The map, where the agent and zombies can walk around. The objects on the map and their positions, such as agent, villagers and zombies. The algorithm will guide the agent to perform desired tasks, for example attacking, moving, etc. 

Output information: Time survived so far, zombies killed so far, and score received so far. 
  Three possible outcome: 
    1. Agent was killed by zombies 
    2. All villagers were killed by zombies 
    3. Both agent and villagers survived in limited time frame (ex. 2 mins)

## AI/ML Algorithms 

Reinforcement learning for updating agent’s strategy.

## Evaluation Plan

The metric would be how many villagers can the agent save in two minutes(the time will vary since we haven’t done any test)

The baseline would be just kill the zombie without considering the greatest profit and few strategy, for example, just protect one villager from a zombie while leaving ten villagers from a bunch of zombies or using the same weapon, such as sword, and running around (of which might waste lots of time)

We may improve the approach by keep updating the agent’s strategy, to save more villagers other than one. The agent may also use different weapons to improve the killing. The evaluation data will still be how many villagers alive after each round.

We will visualize how many zombies and how many villagers are on the map, for example, some dots with different colors, and observe how the agent will act. For example, if the agent is using the shortest way to reach the target position, if the agent make the kill instead of walking around without doing anything or evening if the agent would protect more villagers instead of one.

