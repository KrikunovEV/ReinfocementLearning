from ActorCriticModel import ActorCriticModel
import gym
import torch
import numpy as np


def Preprocess(img):
    img = img[::2, ::2]
    return np.mean(img, axis=2)[np.newaxis,:,:].astype(np.float32) / 255.0


class Agent():

    def __init__(self, cpu):
        self.cpu = cpu
        self.LocalACmodel = ActorCriticModel()
        self.env = gym.make('SpaceInvaders-v0')


    def letsgo(self, GlobalACmodel, lock, sender, MAX_EPISODES, MAX_ACTIONS, DISCOUNT_FACTOR, STEPS):

        optimizer = torch.optim.Adam(GlobalACmodel.parameters(), lr=0.001)

        for episode in range(1, MAX_EPISODES+1):
            print("cpu thread:", self.cpu+1, ", episode:", episode)

            with lock:
                self.LocalACmodel.load_state_dict(GlobalACmodel.state_dict())

            episode_length = 0
            episode_reward = 0
            value_loss = 0
            policy_loss = 0
            done = False

            episode_buffer = []
            episode_values = []
            episode_entropies = []

            obs = Preprocess(self.env.reset())

            for action_count in range(MAX_ACTIONS):

                if self.cpu == 0:
                    self.env.render()

                logit, value = self.LocalACmodel(torch.Tensor(obs[np.newaxis, :, :, :]))

                prob = torch.nn.functional.softmax(logit, dim=-1)
                log_prob = torch.nn.functional.log_softmax(logit, dim=-1)
                entropy = -(log_prob * prob).sum()

                prob_np = prob.detach().numpy()[0]
                action = np.random.choice(prob_np, 1, p=prob_np)
                action = np.where(prob_np == action)[0][0]
                log_prob = log_prob[0, action]
                #action = prob.multinomial(num_samples=1).detach()

                obs_next, reward, done, info = self.env.step(action)
                obs_next = Preprocess(obs_next)
                reward = max(min(reward, 1), -1)
                episode_reward += reward

                episode_buffer.append([reward, entropy, value, log_prob])
                obs = obs_next

                episode_values.append(value.item())
                episode_entropies.append(entropy.item())

                if len(episode_buffer) == STEPS and not(done):
                    value_loss, policy_loss = self.train(episode_buffer, obs, done, DISCOUNT_FACTOR, optimizer, GlobalACmodel, lock)
                    episode_buffer = []

                if done:
                    episode_length = action_count
                    break

            if len(episode_buffer) != 0:
                value_loss, policy_loss = self.train(episode_buffer, obs, done, DISCOUNT_FACTOR, optimizer, GlobalACmodel, lock)

            sender.send((episode, episode_reward, episode_length, np.mean(episode_values), np.mean(episode_entropies),
                         value_loss, policy_loss, self.cpu, False))

        # end of agent
        sender.send((0, 0, 0, 0, 0, 0, 0, self.cpu, True))


    def train(self, buffer, last_obs, done, DISCOUNT_FACTOR, optimizer, GlobalACmodel, lock):

        rewards = [row[0] for row in buffer]
        entropies = [row[1] for row in buffer]
        values = [row[2] for row in buffer]
        log_probs = [row[3] for row in buffer]

        R = torch.Tensor([[0]])
        if not done:
            R = values[-1]

        policy_loss = torch.Tensor([[0]])
        value_loss = torch.Tensor([[0]])

        for i in reversed(range(len(rewards) - 1)):

            R = rewards[i] + DISCOUNT_FACTOR * R
            Advantage = R - values[i]

            # policy update
            policy_loss = policy_loss - log_probs[i] * Advantage.detach() - 0.01 * entropies[i]

            # value update
            value_loss = value_loss + 0.5 * Advantage.pow(2)

        with lock:
            optimizer.zero_grad()

            # 0.5 - value loss coef
            (policy_loss + 0.5 * value_loss).backward()

            # 40 - max grad norm
            torch.nn.utils.clip_grad_norm_(self.LocalACmodel.parameters(), 40)

            for param, shared_param in zip(self.LocalACmodel.parameters(), GlobalACmodel.parameters()):
                #if shared_param.grad is not None:
                #    break
                #shared_param._grad = param.grad
                shared_param.grad = param.grad

            optimizer.step()

        return value_loss, policy_loss
