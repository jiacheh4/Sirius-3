### Minecraft AI Poject:  Sirius-3 ### https://jiacheh4.github.io/Sirius-3/index.html
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
The first naive solution takes significantly longer than 30 sec. Therefore, optimization is needed. Since there is no optimization can be done on server's side (The better solution: make only one API request to retrieve all the data), I decided to use Multi-threading and Queue to optimize/decrease the total time. The time this program takes is around 8 to 11 secs. If the datasetId may be repetitious, then it can be optimized further by implementing a cache to store the result of the datasetId with an O(1) time to fetch.

### Running the tests ###
To run the unittest after you executed the program. Type the line below on the terminal.

    python test_functions.py
