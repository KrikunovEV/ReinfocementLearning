from ActorCriticModel import ActorCriticModel
import gym
import torch
import numpy as np
from random import randint


def Preprocess(img):
    img = img[::2, ::2]
    img = img[10:len(img) - 7]
    return np.mean(img, axis=2)[np.newaxis,:,:].astype(np.float32) / 255.0


class Agent():

    def __init__(self, cpu):
        self.cpu = cpu
        self.LocalModel = ActorCriticModel()
        self.env = gym.make('SpaceInvaders-v0')


    def letsgo(self, GlobalModel, CriticOptimizer, ActorOptimizer, lock, sender,
               MAX_EPISODES, MAX_ACTIONS, DISCOUNT_FACTOR, STEPS):

        self.CriticOptimizer = CriticOptimizer
        self.ActorOptimizer = ActorOptimizer
        self.GlobalModel = GlobalModel
        self.lock = lock

        for episode in range(1, MAX_EPISODES+1):
            print("cpu thread:", self.cpu+1, ", episode:", episode)

            with lock:
                self.LocalModel.load_state_dict(GlobalModel.state_dict())

            episode_length = 0
            episode_reward = 0
            value_loss = 0
            policy_loss = 0
            done = False

            episode_buffer = []
            episode_values = []
            episode_entropies = []

            self.env.reset()

            obs = None

            for _ in range(randint(1, 30)):
                obs, _, _, _ = self.env.step(1)

            obs = Preprocess(obs)

            for action_count in range(MAX_ACTIONS):

                if self.cpu == 0:
                    self.env.render()

                logit, value = self.LocalModel(torch.Tensor(obs[np.newaxis, :, :, :]))

                prob = torch.nn.functional.softmax(logit, dim=-1)
                log_prob = torch.nn.functional.log_softmax(logit, dim=-1)
                entropy = -(log_prob * prob).sum()

                prob_np = prob.detach().numpy()[0]
                action = np.random.choice(prob_np, 1, p=prob_np)
                action = np.where(prob_np == action)[0][0]
                log_prob = log_prob[0, action]

                obs_next, reward, done, info = self.env.step(action)
                obs_next = Preprocess(obs_next)
                np.clip(reward, -1, 1)
                episode_reward += reward

                episode_buffer.append([reward, entropy, value, log_prob])
                obs = obs_next

                episode_values.append(value.item())
                episode_entropies.append(entropy.item())

                if len(episode_buffer) == STEPS and not(done):
                    value_loss, policy_loss = self.train(episode_buffer, done, obs, DISCOUNT_FACTOR)
                    episode_buffer = []

                if done:
                    episode_length = action_count
                    break

            if len(episode_buffer) != 0:
                value_loss, policy_loss = self.train(episode_buffer, done, obs, DISCOUNT_FACTOR)

            sender.send((episode, episode_reward, episode_length, np.mean(episode_values), np.mean(episode_entropies),
                         value_loss, policy_loss, self.cpu, False))

        # end of agent
        sender.send((0, 0, 0, 0, 0, 0, 0, self.cpu, True))


    def train(self, buffer, done, obs, DISCOUNT_FACTOR):

        rewards = [row[0] for row in buffer]
        entropies = [row[1] for row in buffer]
        values = [row[2] for row in buffer]
        log_probs = [row[3] for row in buffer]

        G = 0
        if not done:
            _, G = self.LocalModel(torch.Tensor(obs[np.newaxis, :, :, :]))
            G = G.detach()

        value_loss = 0
        policy_loss = 0

        for i in reversed(range(len(rewards))):
            G = rewards[i] + DISCOUNT_FACTOR * G
            Advantage = G - values[i]

            value_loss += 0.5 * Advantage.pow(2)
            policy_loss -= Advantage * log_probs[i] + 0.01 * entropies[i]

        with self.lock:
            self.CriticOptimizer.zero_grad()
            self.ActorOptimizer.zero_grad()

            # 0.5 - value loss coef
            (policy_loss + value_loss).backward()

            # 40 - max grad norm
            #torch.nn.utils.clip_grad_norm_(self.LocalModel.parameters(), 40)

            for param, shared_param in zip(self.LocalModel.parameters(), self.GlobalModel.parameters()):
                #if shared_param.grad is not None:
                #    break
                #shared_param._grad = param.grad
                shared_param.grad = param.grad

            self.ActorOptimizer.step()
            self.CriticOptimizer.step()

        return value_loss, policy_loss
