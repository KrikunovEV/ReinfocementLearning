from Util import VisdomWrap, Global
import torch
import matplotlib.pyplot as plt
import numpy as np

'''
checkpoints_marines_9 - alpha 0.0005 (1_1)
checkpoints_marines_8 - alpha 0.0002 (1_2)
checkpoints_marines_2 - alpha 0.0001 (1_3)
checkpoints_marines_7 - alpha 0.00005 (1_4)

'''



#path = ['checkpoints_marines_10/1000.pt', 'checkpoints_marines_11/1000.pt', 'checkpoints_marines_12/1000.pt', 'checkpoints_marines_13/1000.pt']
path = ['checkpoints_marines_13/1000.pt']
colors = ['blue', 'red', 'black', 'orange']
labels = ['0.0005', '0.0002', '0.0001', '0.00005']
#DataMgr = VisdomWrap()

for i, p in enumerate(path):

    state = torch.load(p)

    VALUELOSS = state['value_loss']
    VALUELOSS_MEAN = state['value_loss_mean']
    POLICYLOSS = state['policy_loss']
    POLICYLOSS_MEAN = state['policy_loss_mean']
    ENTROPY = state['entropy']
    ENTROPY_MEAN = state['entropy_mean']
    SPATIALENTROPY = state['spatial_entropy']
    SPATIALENTROPY_MEAN = state['spatial_entropy_mean']
    REWARDS = state['reward']
    REWARDS_MEAN = state['reward_mean']
    EPISODES = state['episodes']
    NSTEPITER = state['nstepiter']

    # DataMgr.set_data(VALUELOSS, VALUELOSS_MEAN, POLICYLOSS, POLICYLOSS_MEAN, ENTROPY, ENTROPY_MEAN,
                     #SPATIALENTROPY, SPATIALENTROPY_MEAN, REWARDS, REWARDS_MEAN, EPISODES, NSTEPITER)

    # DataMgr.send_current_data()


    #Global.load('checkpoints_marines_9/')
    #Global.debug_print()

    plt.ylabel('количество морпехов', fontsize=14)
    plt.xlabel('эпизод', fontsize=14)
    plt.title('a = 0.0001', fontsize=14)

    plt.plot(EPISODES, REWARDS)
    plt.plot([x + 4.5 for x in EPISODES[::10]], REWARDS_MEAN, linewidth=3, color='red')

    #plt.plot([x + 4.5 for x in EPISODES[::10]], REWARDS_MEAN, linewidth=3, color=colors[i], label=labels[i])
    max = np.max(REWARDS)
    amax = np.argmax(REWARDS)
    print(max, amax)
    #plt.scatter(amax, max, c=colors[i], marker='o', linewidths=5)
#plt.title('Усреднённые итоговые вознаграждения за 10 эпизодов', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(0, 120)
plt.xlim(0, 1000)
#plt.legend(loc='upper left', fontsize=12)
#plt.show()
