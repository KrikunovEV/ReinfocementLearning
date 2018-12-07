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
        self.sender = sender
        self.DISCOUNT_FACTOR = DISCOUNT_FACTOR

        for episode in range(1, MAX_EPISODES+1):
            print("cpu thread:", self.cpu+1, ", episode:", episode)

            with lock:
                self.LocalModel.load_state_dict(GlobalModel.state_dict())

            episode_reward = 0
            done = False

            values, entropies, log_probs, rewards = [], [], [], []

            obs = self.env.reset()
            for _ in range(randint(1, 30)):
                obs, _, _, _ = self.env.step(1)
            obs = Preprocess(obs)

            for action_count in range(1, MAX_ACTIONS):

                if self.cpu == 0:
                    self.env.render()

                logit, value = self.LocalModel(torch.Tensor(obs[np.newaxis, :, :, :]))

                prob = torch.nn.functional.softmax(logit, dim=-1)
                log_prob = torch.nn.functional.log_softmax(logit, dim=-1)
                entropy = -(log_prob * prob).sum()

                prob_np = prob.detach().numpy()[0]
                action = np.random.choice(prob_np, 1, p=prob_np)
                action = np.where(prob_np == action)[0][0]
                log_prob = log_prob[0][action]

                obs, reward, done, info = self.env.step(action)
                obs = Preprocess(obs)
                np.clip(reward, -1, 1)
                episode_reward += reward

                values.append(value)
                entropies.append(entropy)
                log_probs.append(log_prob)
                rewards.append(reward)

                if done:
                    sender.send((self.cpu, False, 0, 0, 0, episode_reward, False))
                    break

                if action_count % STEPS == 0:
                    self.train(values, entropies, log_probs, rewards, obs, done)
                    values, entropies, log_probs, rewards = [], [], [], []

            self.train(values, entropies, log_probs, rewards, obs, done)

        # end of agent
        sender.send((0, 0, 0, 0, 0, 0, 0, self.cpu, True))
        self.env.close()


    def train(self, values, entropies, log_probs, rewards, obs, done):

        G = 0
        if not done:
            _, G = self.LocalModel(torch.Tensor(obs[np.newaxis, :, :, :]))
            G = G.detach()

        value_loss = 0
        policy_loss = 0

        for i in reversed(range(len(rewards))):
            G = rewards[i] + self.DISCOUNT_FACTOR * G
            Advantage = G - values[i]

            value_loss += 0.5 * Advantage.pow(2)
            policy_loss -= Advantage.detach() * log_probs[i] + 0.01 * entropies[i]

        with self.lock:
            self.CriticOptimizer.zero_grad()
            self.ActorOptimizer.zero_grad()

            # 0.5 - value loss coef
            (policy_loss + value_loss).backward()

            # 40 - max grad norm
            torch.nn.utils.clip_grad_norm_(self.LocalModel.parameters(), 40)

            for param, shared_param in zip(self.LocalModel.parameters(), self.GlobalModel.parameters()):
                #if shared_param.grad is not None:
                    #break
                shared_param._grad = param.grad

            self.ActorOptimizer.step()
            self.CriticOptimizer.step()

        self.sender.send((self.cpu, True, value_loss.item(), policy_loss.item(),
                          np.mean([entropy.detach().numpy() for entropy in entropies]), 0, False))
