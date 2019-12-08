
import random
import torch

EPSILON_THRESHOLD = 0.05
epsilon = 1.0

for episode in range(1, MAX_EPISODES):

    obs = preprocess(env.reset())

    for _ in range(MAX_ACTIONS):
        env.render()

        # E-greedy
        epsilon = getEpsilon(epsilon, EPSILON_THRESHOLD)
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = torch.argmax(model.noGradForward(torch.Tensor(obs[np.newaxis,:,:,:]).cuda()))


        # Make a step
        obs_next, reward, done, info = env.step(action)
        obs_next = preprocess(obs_next)

        if done:
            reward = -1
        else:
            reward = getReward(reward)


        # Save experience
        experience.addExperience(obs, obs_next, reward, action, done)
        obs = obs_next
        if len(experience.getExperience(BATCH_SIZE)) < 2:
            continue


        # Train
        exp = experience.getExperience(BATCH_SIZE)

        Qvalues = getQvalues(model, exp, DISCOUNT_FACTOR)

        data = torch.Tensor([exp[i][0] for i in range(len(exp))]).cuda()
        data = model(data)

        loss = loss_fn(data, Qvalues)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if done:
            break

env.close()