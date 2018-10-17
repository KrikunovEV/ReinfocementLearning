from ActorCriticModel import ActorCriticModel



class Agent():

    def __init__(self, GlobalACmodel, scope):
        self.GlobalACmodel = GlobalACmodel
        self.scope = scope

        self.LocalACmodel = ActorCriticModel(scope)
        # need copy params from global


    def start(self, MAX_EPISODES, MAX_ACTIONS, DISCOUNT_FACTOR):

        for episode in range(MAX_EPISODES):

