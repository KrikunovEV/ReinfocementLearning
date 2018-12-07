# ReinforcementLearning

Here You can see the progress of my **Graduate work**.

### Starcraft 2
1. The main [StarCraft II: A New Challenge for Reinforcement Learning](https://arxiv.org/pdf/1708.04782.pdf);
2. API for Starcraft 2 - [PySC2](https://github.com/deepmind/pysc2);
3. Detailed information about PySC2 is [here](https://github.com/deepmind/pysc2/blob/master/docs/environment.md);

### Some useful links:
1. From REINFORCE to Advantage Actor-Critic - [the book, chapter 13](http://incompleteideas.net/book/bookdraft2017nov5.pdf)
2. The main paper A3C [Asynchronous Actor-Critic Agents](https://arxiv.org/pdf/1602.01783.pdf);
3. [Advantage Actor-Critic](https://hackernoon.com/intuitive-rl-intro-to-advantage-actor-critic-a2c-4ff545978752) in pictures;
4. [Space Invaders](https://nihit.github.io/resources/spaceinvaders.pdf) with DQN;
5. [Frame skipping](http://nn.cs.utexas.edu/downloads/papers/braylan.aaai15.pdf)

### Can be interesting:
1. VIN [Value Iteration Network](http://papers.nips.cc/paper/6046-value-iteration-networks.pdf);
2. DQN [Deep Q Network](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf);
3. [Prioritized experience replay](https://arxiv.org/pdf/1511.05952.pdf);
4. Generalized Advantage Estimation for policy gradient [GAE](https://arxiv.org/pdf/1506.02438.pdf)
5. All environments from [Gym](https://gym.openai.com/envs/#classic_control).

### How to get atari games and PySC2
```python
pip install gym # for classic control
pip install gym[atari] # Atari games
pip install gym[all] # get all games
pip install git+https://github.com/Kojoley/atari-py.git # get atari-py if needed
pip install pysc2
```

### How to collab
```python
# Get your google drive
from google.colab import drive
drive.mount("./googledrive")

# Put ! in beggining of pip command
!pip install gym[atari]

# Run your .py file with %run
%run ./main.py
```
