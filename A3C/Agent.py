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


    def letsgo(self, GlobalACmodel, optimizer, lock, sender, MAX_EPISODES, MAX_ACTIONS, DISCOUNT_FACTOR, STEPS):

        for episode in range(1, MAX_EPISODES+1):
            print("cpu:", self.cpu, ", Episode:", episode)

            self.LocalACmodel.load_state_dict(GlobalACmodel.state_dict())

            episode_length = 0
            episode_mean_value = 0
            episode_mean_entropy = 0
            episode_reward = 0

            episode_buffer = []
            episode_values = []
            episode_entropies = []

            value_loss = 0
            policy_loss = 0

            obs = Preprocess(self.env.reset())

            for action_count in range(MAX_ACTIONS):

                if self.cpu == 0:
                    self.env.render()

                logit, value = self.LocalACmodel(torch.Tensor(obs[np.newaxis, :, :, :]))
                episode_values.append(torch.Tensor(value).detach().numpy())

                prob = torch.nn.functional.softmax(logit, dim=-1)
                log_prob = torch.nn.functional.log_softmax(logit, dim=-1)
                entropy = -(log_prob * prob).sum(1, keepdim=True)
                episode_entropies.append(torch.Tensor(entropy).detach().numpy())

                action = prob.multinomial(num_samples=1).detach()
                log_prob = log_prob.gather(1, action)

                obs_next, reward, done, info = self.env.step(action)
                obs_next = Preprocess(obs_next)
                reward = max(min(reward, 1), -1)

                episode_buffer.append([obs, action, reward, obs_next, entropy, value, log_prob])
                obs = obs_next
                episode_reward += reward

                if len(episode_buffer) == STEPS and not(done):
                    #print(self.train(episode_buffer, DISCOUNT_FACTOR, optimizer, lock, GlobalACmodel))
                    #value_loss, policy_loss = self.train(episode_buffer, DISCOUNT_FACTOR, optimizer, lock, GlobalACmodel)
                    self.train(episode_buffer, DISCOUNT_FACTOR, optimizer, lock, GlobalACmodel)
                    episode_buffer = []

                if done:
                    episode_length = action_count
                    episode_mean_value = np.mean(episode_values)
                    episode_mean_entropy = np.mean(episode_entropies)
                    break

            if len(episode_buffer) != 0:
                #value_loss, policy_loss = self.train(episode_buffer, DISCOUNT_FACTOR, optimizer, lock, GlobalACmodel)
                self.train(episode_buffer, DISCOUNT_FACTOR, optimizer, lock, GlobalACmodel)

            sender.send((episode, episode_reward, episode_length, episode_mean_value, episode_mean_entropy,
                         value_loss, policy_loss, self.cpu, False))

        # end of agent
        sender.send((0, 0, 0, 0, 0, 0, 0, self.cpu, True))


    def train(self, buffer, DISCOUNT_FACTOR, optimizer, lock, GlobalACmodel):
        buffer = np.array(buffer)
        obs = buffer[:, 0]
        actions = buffer[:, 1]
        rewards = buffer[:, 2]
        obs_next = buffer[:, 3]
        entropies = buffer[:, 4]
        values = buffer[:, 5]
        log_probs = buffer[:, 6]

        _, value_next = self.LocalACmodel(torch.Tensor((obs_next[-1])[np.newaxis, :, :, :]))
        policy_loss = 0
        value_loss = 0
        gae = 0
        R = 0
        for i in reversed(range(len(rewards))):
            # A = R - V
            # A's loss 1/2 * A^2
            R = R * DISCOUNT_FACTOR + rewards[i]
            A = R - values[i]
            value_loss = value_loss + 0.5 * A.pow(2)

            # general advantage estimation
            # for policy
            if i == len(rewards) - 1:
                delta_t = rewards[i] + DISCOUNT_FACTOR * value_next - values[i]
            else:
                delta_t = rewards[i] + DISCOUNT_FACTOR * values[i+1] - values[i]
            # 1.0 - tau
            gae = gae * DISCOUNT_FACTOR * 1.0 + delta_t
            # 0.01 - entropy_coef
            policy_loss = policy_loss - log_probs[i] * gae.detach() - 0.01 * entropies[i]

        with lock:
            optimizer.zero_grad()

            # 0.5 - value loss coef
            #print("Policy loss:", policy_loss)
            #print("Value loss:", value_loss)
            Loss = policy_loss + 0.5 * value_loss
            #print("Loss:", Loss)
            #print()
            Loss.backward()

            # 50 - max grad norm
            torch.nn.utils.clip_grad_norm_(self.LocalACmodel.parameters(), 50)

            for param, shared_param in zip(self.LocalACmodel.parameters(), GlobalACmodel.parameters()):
                if shared_param.grad is not None:
                    return
                shared_param._grad = param.grad

            optimizer.step()

        return value_loss, policy_loss
