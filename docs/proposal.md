---
layout:  default
title:    Proposal
---

## Summary of the Project

Project Overview

This project aims to develop an AI agent capable of protecting villagers by eliminating randomly spawning zombies, and minimizing villager casualties. Using reinforcement learning, the agent learns to identify and prioritize threats, optimizing its decision-making to neutralize zombies as efficiently as possible.

Project Goal

The objective is to maximize villager survival by training the agent to execute strategic actions, such as navigating the environment, locating threats, and attacking zombies effectively.

Input Data

Map environment: Defines the terrain where the agent, villagers, and zombies navigate.
Entity positions: Real-time tracking of all objects, including obstacles.
Decision-making algorithm: Guides the agent in selecting optimal actions such as movement and attack strategies.

Output Metrics
 1. Performance tracking: Time survived, number of zombies eliminated, and accumulated score.
 2. Possible Outcomes
    
       The agent is eliminated by zombies.
    
       All villagers are killed.
    
       The agent successfully protects villagers within a set time frame (e.g., 2 minutes).

## AI/ML Algorithms 

Deep Reinforcement Learning (DQN/Double DQN): Enables the agent to continuously refine its strategy through experience.
Policy Optimization: Adjusts decision-making to prioritize high-value targets efficiently.

## Evaluation Plan

Key Metric: Number of villagers saved within a defined time limit (e.g., 2 minutes).

Baseline Performance:

The agent reacts without prioritization, focusing only on killing zombies rather than maximizing villager survival.
Uses basic strategies (e.g., protecting a single villager while ignoring a larger threat, relying on a single weapon, or aimlessly running around).

Proposed Improvements:

1. Implement adaptive strategy updates to improve villager survival rates.
2. Introduce multiple weapon options for more effective threat elimination.
3. Evaluate improvements by tracking the number of surviving villagers at the end of each round.

Visualization & Behavior Analysis:

1. Display entity positions using color-coded markers for villagers, zombies, and the agent.
2. Analyze the agent’s movement efficiency—ensuring it takes optimal paths rather than wandering aimlessly.
3. Observe decision-making effectiveness, such as whether the agent prioritizes protecting multiple villagers instead of just one.
