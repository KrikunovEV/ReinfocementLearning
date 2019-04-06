
from Util import *

from pysc2.lib import actions as sc2_actions


FUNCTION_TYPES = sc2_actions.FUNCTION_TYPES
for a in FUNCTIONS[3].args:
    if a is sc2_actions.TYPES.queued:
        continue
    print(len(a.sizes))
#print(FUNCTION_TYPES[FUNCTIONS[331].function_type][1])

action_mask = [  1  , 2   , 4  , 5, 453  , 7, 331 ,332, 333, 334 , 12 , 13 ,274]

actions_ids = [i for i, action in enumerate(MY_FUNCTION_TYPE) if action in action_mask]

probs = [0.1, 0.4, 0.3, 0.2]


prob = np.random.choice(probs, 1, p=probs)
action_id = np.where(probs == prob)[0][0]
prob = probs[action_id]  # to get attached tensor
action_id_result = MY_FUNCTION_TYPE[actions_ids[action_id]]  # to get real id

#print(MY_FUNCTION_TYPE)
#print()
#print("available: ", actions_ids)
#print("prob: ", prob)
#print("from available: ", action_id)
#print("from all my actions: ", actions_ids[action_id])
#print("from actual: ", action_id_result)