import numpy as np
import pickle
import gymnasium as gym
from typing import Tuple, Dict
import os
import time
from gymnasium.wrappers.record_video import RecordVideo


# Hyperparameters
batch_size = 20 # used to perform a RMS prop param update every batch_size steps
learning_rate = 1e-3 # Learning rate
gamma = 0.99 # Discount factor
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2

# Config flags
resume = True # resume training from previous checkpoint
render = False # render video output
save_name = os.path.join("no_downscaling")
save_dir = os.path.join("save_files", save_name)

if not os.path.exists(save_dir):
   os.makedirs(save_dir)

A = 4 # Action space: 0-3 [0: NOOP, 1: FIRE, 2: RIGHTFIRE, 3: LEFTFIRE]
H = 200 # number of hidden layer neurons
D = 76 * 144 # input dimensionality: 76x144 grid



def load_model():
    return pickle.load(open(os.path.join(save_dir, 'save.p'), 'rb'))
def load_running_means():
   return pickle.load(open(os.path.join(save_dir, 'running_means.p'), 'rb'))

model = load_model()


# softmax from: https://gist.github.com/etienne87/6803a65653975114e6c6f08bb25e1522
def softmax(x):
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    return probs

def policy_forward(x: np.ndarray) -> Tuple[float, np.ndarray]:
  """Forward propagation. Input is preprocessed input: 52x80 float column vector"""
  if(len(x.shape)==1):
    x = x[np.newaxis,...]

  hidden_states = np.dot(x, model['W1']) 
  hidden_states[hidden_states < 0] = 0 # ReLU introduces non-linearity
  logp = np.dot(hidden_states, model['W2']) # This is a logits function and outputs a decimal.   (A x H) . (H x A) = A
  return softmax(logp), hidden_states




render_mode = "human" if render else "rgb_array"
env = gym.make("ALE/Atlantis-v5", render_mode=render_mode)

def preprocess(observation: np.ndarray):
    """ preprocess 210x160x3 uint8 frame into 10944 (76x144) float column vector """
    observation = observation[16:92, 8:152] # Delete all areas where the ships can never be
    observation = observation[::,::,0]
    observation[observation != 0] = 1 # Make grayscale

    return observation.astype(float).ravel()




observation: np.ndarray
(observation, info) = env.reset()

prev_observation = None # used in computing the difference frame
episode_number = 0

i = 0
total_rewards = []
reward_sum = 0
prev_checkpoint_reward = -1
# Episode Generation
while(episode_number < 100):
    cur_observation = preprocess(observation)
    observation_delta: np.ndarray = cur_observation - prev_observation if prev_observation is not None else np.zeros(D)
    prev_observation = cur_observation

    # forward prop
    # (action_probs, hidden_states) = policy_forward(observation_delta)
    (action_probs, hidden_states) = policy_forward(cur_observation)

    # sample next action given softmax'ed probabilities
    random = np.random.uniform()
    running_total_prob = 0
    for index, action_prob in enumerate(action_probs[0]):
        running_total_prob += action_prob
        if random <= running_total_prob:
           action = index
           break
    # action = 0
    # max_prob = 0
    # for index, action_prob in enumerate(action_probs[0]):
    #    if action_prob > max_prob:
    #       max_prob = action_prob
    #       action = index  
    
    # Just for safety
    if not action:
       action = 0 

    if i % 10000 == 0:
        print(f"Current reward: {reward_sum}")
        if reward_sum == prev_checkpoint_reward:
          print("Manually terminating. Environment broken")
          should_terminate = True
          print(f"truncated: {truncated}")
        prev_checkpoint_reward = reward_sum

    observation, reward, terminated, truncated, info = env.step(action)
    reward_sum += reward

    if terminated or reward_sum > 2_000_000:
        episode_number += 1
        
        print ('resetting env. episode reward total was %f' % (reward_sum))
        total_rewards.append(reward_sum)
        reward_sum = 0
        (observation, info) = env.reset() # reset env
        prev_x = None

        print(f'ep {episode_number}: game finished')
    i += 1
average_rewards = sum(total_rewards) / len(total_rewards)
print(f"Average: {average_rewards}")
    
