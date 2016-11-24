import random
import collections
import sys
import numpy as np
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator


GLOBAL_PARAMS = {'epsilon': 0.0, 'alpha': 0.9, 'gamma': 0.2}


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

        # Initialize any additional variables here

        # Q matrix : Q[states][actions]
        self.Q = collections.defaultdict(dict)
        self.possible_actions = [None, 'forward', 'right', 'left']
        possible_states = [(w,x,y,z) for w in ['red','green'] for x in self.possible_actions for y in self.possible_actions for z in ['forward', 'right', 'left']]
        for state in possible_states:
            self.Q[state] = {None:1.0, 'forward':1.0, 'right':1.0, 'left':1.0}

        # Counters
        self.success = 0  # Number of successful trials
        self.success_p = 0  # Number of successful trials if no -1.0 penalty in the last 20 trials
        self.trials = 0  # Number of trials

        # Q Learning Hyperparameters
        self.epsilon = GLOBAL_PARAMS['epsilon'] # Random exploration probability
        self.alpha = GLOBAL_PARAMS['alpha'] # learning rate coefficient
        self.gamma = GLOBAL_PARAMS['gamma'] # discount coefficient

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # Prepare for a new trip; reset any variables here, if required
        self.penalty = False  # True if there is a -1.0 step in the trial
        self.trials += 1

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # Update state
        self.state = (inputs['light'], inputs['oncoming'], inputs['left'], self.next_waypoint)

        # Select action according to your policy

        # Small probability to explore randomly otherwise pick action according to the Q matrix
        if random.random() < self.epsilon:
            action = random.choice(self.possible_actions)
        else:
            # Select the action with the highest q value
            max_q = max(self.Q[self.state].values())
            max_actions = [k for k,v in self.Q[self.state].items() if v == max_q]
            action = max_actions[0]

        # Execute action and get reward
        reward = self.env.act(self, action)
        # Driving infraction detected
        if reward == -1.0:
            self.penalty = True
        # If the reward is greater than 8, the agent has reached its destination
        if reward > 8:
            self.success += 1
            # For the last 20 trials, if there is no penalty on the way, it is a clean ride.
            if self.trials > 80 and self.penalty == False:
                self.success_p += 1

        #print "Q : {}, action : {}, reward : {}".format(self.Q[self.state], action, reward)

        # Learn policy based on state, action, reward
        q = self.Q[self.state][action]
        next_inputs = self.env.sense(self)
        next_state = (next_inputs['light'], next_inputs['oncoming'], next_inputs['left'], self.planner.next_waypoint())  # Next state
        self.Q[self.state][action] = (1 - self.alpha) * q + self.alpha * (reward + self.gamma * max(self.Q[next_state].values()))

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.1, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials

    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

    return a.epsilon, a.alpha, a.gamma, a.success, a.success_p


def parsing_results():
    """Parse the text file containing the output of the exhaustive search."""

    filename = "./gridsearch_results.txt"
    d = {}

    with open(filename, 'r') as f:
        for line in f:
            s = line.split()
            d[(float(s[1]), float(s[3]), float(s[5]))] = (float(s[7]), float(s[9]))

    max_tuple = max(d.values())
    best_parameters = [k for k in d.keys() if d[k]==max_tuple][0]
    return d, best_parameters


def run_exhaustive():
    """Calls run() for all possible combinations of parameters."""
    global GLOBAL_PARAMS

    GLOBAL_PARAMS['epsilon'] = 0.0
    GLOBAL_PARAMS['alpha'] = 0.0
    GLOBAL_PARAMS['gamma'] = 0.0

    for epsilon in [0.0, 0.1, 0.2]:
        for alpha in [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]:
            for gamma in [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]:
                GLOBAL_PARAMS['epsilon'] = epsilon
                GLOBAL_PARAMS['alpha'] = alpha
                GLOBAL_PARAMS['gamma'] = gamma
                successes = []
                successes_p = []
                # run() 5 times then average on success and succes_p
                for i in range(5):
                    epsilon, alpha, gamma, success, success_p = run()
                    print "Simulation {} [epsilon {}, alpha {}, gamma {}] ==> success : {} success_p: {}".format(i, epsilon, alpha, gamma, success, success_p)
                    successes.append(success)
                    successes_p.append(success_p)
                avgsuccess = np.mean(successes)
                avgsuccess_p = np.mean(successes_p)

                filename = "./gridsearch_results.txt"
                with open(filename, 'a') as f:
                    f.write("epsilon {} alpha {} gamma {} avgsuccess {} avgsuccess_p {}\n".format(epsilon, alpha, gamma, avgsuccess, avgsuccess_p))

    d, best_parameters = parsing_results()
    print "Best parameters : {}".format(best_parameters)


if __name__ == '__main__':
    run()
    # Replace run() by run_exhaustive() in case of gridsearch
    #run_exhaustive()
