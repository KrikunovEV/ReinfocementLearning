# ReinforcementLearning

Here You can see the progress of my **Graduate work**.

### Papers and links:
1. The main [StarCraft II: A New Challenge for Reinforcement Learning](https://arxiv.org/pdf/1708.04782.pdf)
2. A3C network [Asynchronous Actor-Critic Agents](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2)
3. VIN [Value Iteration Network](http://papers.nips.cc/paper/6046-value-iteration-networks.pdf)
4. DQN [Deep Q Network](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
5. [Prioritized experience replay](https://arxiv.org/pdf/1511.05952.pdf)

Additionally:

6. Environments for classic control from [GYM](https://github.com/openai/gym/wiki/Leaderboard)
7. All environments from [GYM](https://gym.openai.com/envs/#classic_control)
8. Intuitive RL [A2C](https://hackernoon.com/intuitive-rl-intro-to-advantage-actor-critic-a2c-4ff545978752)
9. API for Starcraft 2 - [PySC2](https://github.com/deepmind/pysc2)
10. Detailed information about PySC is [here](https://github.com/deepmind/pysc2/blob/master/docs/environment.md)
11. In this work authors propose multi-threaded asynchronous variants of one-step Sarsa, one-step Q-learning, n-step Q-learning, and
advantage actor-critic using multiple CPU threads on a single machine instead of separate machines.  [Asynchronous Methods for Deep RL](https://arxiv.org/pdf/1602.01783.pdf#page=9)


### How to get atari-py and atari games
```
pip install gym # for classic control
pip install gym[all] # all games
pip install git+https://github.com/Kojoley/atari-py.git
```
