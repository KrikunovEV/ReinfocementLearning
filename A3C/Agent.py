from ActorCriticModel import ActorCriticModel
import gym
import torch
import numpy as np

def Preprocess(img):
    img = img[::2, ::2]
    return np.mean(img, axis=2)[np.newaxis,:,:].astype(np.float32) / 255.0

class Agent():

    def __init__(self, GlobalACmodel, scope):
        self.GlobalACmodel = GlobalACmodel
        self.scope = scope

        self.LocalACmodel = ActorCriticModel(scope)
        self.LocalACmodel.load_state_dict(self.GlobalACmodel.state_dict())

        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_value = []

        self.env = gym.make('SpaceInvaders-v0')


    def start(self, MAX_EPISODES, MAX_ACTIONS, DISCOUNT_FACTOR, BATCH_SIZE):

        for episode in range(MAX_EPISODES):
            print("Episode", episode)

            episode_buffer = []
            episode_values = []
            episode_frames = []
            episode_reward = 0

            obs = self.env.reset()
            episode_frames.append(obs)
            obs = Preprocess(obs)

            for action_count in range(MAX_ACTIONS):
                probabilities, values = self.LocalACmodel.noGradForward(torch.Tensor(obs[np.newaxis, :, :, :]).cuda())
                action = torch.argmax(probabilities)

                obs_next, reward, done, info = self.env.step(action)

                if done:
                    obs = obs_next
                else:
                    episode_frames.append(obs_next)
                    obs_next = Preprocess(obs_next)

                episode_buffer.append([obs, action, reward, obs_next, done, values])
                episode_values.append(values)

                episode_reward += reward
                obs = obs_next

                if len(episode_buffer) == BATCH_SIZE and not(done):
                    self.train(episode_buffer)

                if done:
                    break


    def train(self, buffer):
        pass
