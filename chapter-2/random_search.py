import tensorflow as tf
import numpy as np
import gym
import logging

logger = logging.getLogger('rl')
logger.setLevel(logging.DEBUG)

# Harness 가 하는 일은 environment 를 제공하고 episode 를 수행하는 것이다.
# next_action 을 하고 env 에서 response 를 받는 것이 전부임.
class Harness:

    def run_episode(self, env, agent):
        observation = env.reset()
        total_reward = 0
        for _ in range(1000):
            action = agent.next_action(observation)
            observation, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
        return total_reward

# env 4 개 parameters 를 random 하게 생성
class LinearAgent:

    def __init__(self):
        self.parameters = np.random.rand(4) * 2 - 1

    def next_action(self, observation):
        return 0 if np.matmul(self.parameters, observation) < 0 else 1


def random_search():
    env = gym.make('CartPole-v0')
    best_parameters = None
    best_reward = 0
    agent = LinearAgent()
    harness = Harness()

    for step in range(1000):
        agent.parameters = np.random.rand(4) * 2 - 1
        reward = harness.run_episode(env, agent)
        if reward > best_reward:
            best_reward = reward
            best_parameters = agent.parameters
            if reward == 200:
                print("achieved 200 on step {}".format(step))
                break
        if step % 100 == 0:
            print(reward)
            print(best_parameters)

random_search()
