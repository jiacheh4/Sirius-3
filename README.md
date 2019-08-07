### Minecraft AI Poject:  Sirius-3                                      https://jiacheh4.github.io/Sirius-3/index.html ### 
![Image](docs/image/img.jpg)

### Overview ###
The main idea of the project is to kill random spawning zombies in order to minimize the loss of villagers. By performing enforcement learning on the agent, the agent is able to learn/identify the nearest threat to a villager and eliminate the threat as quick as possible. The goal of the project is to let the agent protects as many villagers as possible by performing actions such as moving to and attacking zombies.

Input information: The map, where the agent and zombies can walk around. The objects on the map and their positions, such as agent, villagers and zombies. The algorithm will guide the agent to perform desired tasks, for example attacking, moving, etc.
Output information: Time survived so far, zombies killed so far, and score received so far. Three possible outcome: 1. Agent was killed by zombies 2. All villagers were killed by zombies 3. Both agent and villagers survived in limited time frame (ex. 2 mins)

### Prerequisites ###
Note: Please install Malmo to run this program. 
      Please install the "requests" module in order to sucessfully submit/POST the answer
      
    pip3 install requests

### Getting Started ###


### Running the Program ###
To run the unittest after you executed the program. Type the line below on the terminal.

    python test_functions.py
