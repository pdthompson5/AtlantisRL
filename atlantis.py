import numpy as np
import pickle
import gymnasium as gym
from typing import Tuple, Dict, List    
from matplotlib import pyplot as plt
import csv
import os
import time
from numba import jit
from scipy.special import softmax

# TODO: Cuda toolkit is required?


# Things to do:
# Save necessary information from episode generation 

A = 4 # Action space: 0-3
H = 200 # number of hidden layer neurons
batch_size = 50 # used to perform a RMS prop param update every batch_size steps
learning_rate = 1e-3 # Learning rate
gamma = 0.99 # Discount factor
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2

# Config flags - video output and res
resume = False # resume training from previous checkpoint (from save.p  file)?
render = False # render video output?

save_name = "array_model"
save_dir = os.path.join("save_files", save_name)

if not os.path.exists(save_dir):
   os.makedirs(save_dir)

D = 60 * 80 # input dimensionality: 60x80 grid

def initialize_model():
    model: List[np.ndarray] = []
    model.append(np.random.randn(D,H) / np.sqrt(D)) # "Xavier" initialization
    model.append(np.random.randn(H,A) / np.sqrt(H)) # Shape will be H
    return model

def load_model():
    return pickle.load(open(os.path.join(save_dir, 'save.p'), 'rb'))
def load_running_means():
   return pickle.load(open(os.path.join(save_dir, 'running_means.p'), 'rb'))

model = initialize_model() if not resume else load_model()
running_means = [] if not resume else load_running_means()

def save_model():
    pickle.dump(model, open(os.path.join(save_dir, 'save.p'), 'wb'))
def save_running_means():
    pickle.dump(running_means, open(os.path.join(save_dir, 'running_means.p'), "wb"))

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

# softmax from: https://gist.github.com/etienne87/6803a65653975114e6c6f08bb25e1522
def softmax(x):
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    return probs

# TODO: Will I need to change this because of the larger action space?
# TODO: The initial impl of this will just consider left or right fire, no center




# Takes about 2 ms
@jit(nopython=True)
def policy_forward_jit(x: np.ndarray, model: np.ndarray) -> Tuple[float, np.ndarray]:
    """Forward propagation. Input is preprocessed input: 60x80 float column vector"""
    hidden_states = x.dot(model[0]) # (H x D) . (D x 1) = (H x 1) (200 x 1)
    # hidden_states[hidden_states < 0] = 0 # ReLU introduces non-linearity
    for state in hidden_states[0]:
        if state < 0:
            state = 0
    logp = hidden_states.dot(model[1]) # This is a logits function and outputs a decimal.   (1 x H) . (H x 1) = 1 (scalar)
    #   sigmoid_prob = sigmoid(logp)  # squashes output to  between 0 & 1 range

    # probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    # probs /= np.sum(probs, axis=1, keepdims=True)
    # probs = softmax(logp)
    return logp, hidden_states # return probability of taking action 2 (RIGHTFIRE), and hidden state

def policy_forward(x: np.ndarray):
    if len(cur_observation.shape) == 1:
        x = x[np.newaxis,...]
    logp, hidden_states = policy_forward_jit(x, model)
    # print(logp)
    # print(hidden_states)
    return (softmax(logp), hidden_states)


def policy_backward(eph: np.ndarray, epx: np.ndarray, epdlogp: np.ndarray):
  """ Manual implementation of a backward prop"""
  """ It takes an array of the hidden states that corresponds to all the images that were
  fed to the NN (for the entire episode, so a bunch of games) and their corresponding logp"""
  dW2 = eph.T.dot(epdlogp)
  dh = epdlogp.dot(model[1].T)
  dh[eph <= 0] = 0 # backpro prelu
  dW1 = epx.T.dot(dh)
  return [dW1, dW2]



# TODO: I wonder if I should increase the difficulty to make it learn better?
# TODO: Calculate the performance of the random policy
# env = gym.make("ALE/Atlantis-v5", render_mode="human")


render_mode = "human" if render else "rgb_array"
env = gym.make("ALE/Atlantis-v5", render_mode=render_mode)

def print_np_array(observation: np.ndarray):
    for row in observation:
        for col in row:
            print(col, end=" ")
        print()



# TODO: Might need to handle screen flashes
def preprocess(observation: np.ndarray):
    """ preprocess 210x160x3 uint8 frame into 4800 (60x80) float column vector """
    # Takes an average of .06 ms
    observation = observation[:120] # Delete everything below row 121. Only the upper part of the screen is important

    #TODO: I might not be able to downscale   this much, might not be enough detail
    observation = observation[::2,::2,0] # downscale by factor of 2
    observation[observation != 0] = 1 # Make grayscale

    return observation.astype(float).ravel()

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)):
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r


grad_buffer = [np.zeros_like(v) for v in model ] # update buffers that add up gradients over a batch
rmsprop_cache = [np.zeros_like(v) for v in model ] # rmsprop memory

observation: np.ndarray
(observation, info) = env.reset()

prev_observation = None # used in computing the difference frame
observation_deltas, hidden_states_stack, dlogps, rewards = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0

i = 0

# Episode generate takes ~2.5 seconds
    # Per total step env step takes 1ms, forward prop takes 2ms
episode_start_times = time.time()
while(True):
    init_time = time.time()
    if render: env.render()

    t  = time.time()
    cur_observation = preprocess(observation)
    observation_delta: np.ndarray = cur_observation - prev_observation if prev_observation is not None else np.zeros(D)
    prev_observation = cur_observation

    # print((time.time()-t)*1000, ' ms, @prepo')

    # forward prop
    # t  = time.time()

    (action_probs, hidden_states) = policy_forward(cur_observation)
    # print((time.time()-t)*1000, ' ms, @forward')

    # sample next action given softmax'ed probabilities
    random = np.random.uniform()
    running_total_prob = 0
    for index, action_prob in enumerate(action_probs[0]):
        running_total_prob += action_prob
        if random <= running_total_prob:
           action = index
           break

    # Just for safety
    if not action:
       action = 0 
    # print(action)

    observation_deltas.append(observation_delta)
    hidden_states_stack.append(hidden_states)

    dlogsoftmax = action_probs.copy()
    # This represents the difference between the action probabilities that we received and action probabilities that 
    # would always result in the action that we chose
    dlogsoftmax[0,action] -= 1
    # print(dlogsoftmax)
    # TODO: For some reason I need to invert this -> The reason why is the plus sign in rms prop updates
    dlogsoftmax[0] *= -1
    # print(dlogsoftmax)
    dlogps.append(dlogsoftmax)


    # TODO: For some reason we need to no-op between actions. holding down the button doesn't work I guess
    # It seems like this is for preventing button spamming.
    t  = time.time()
    # Takes about 1ms
    observation, reward, terminated, truncated, info = env.step(action)
    reward_sum += reward
    rewards.append(reward) # record reward (has to be done after we call step() to get reward for previous action)
    # print((time.time()-t)*1000, ' ms, @env.step')
    # Takes about 2ms per step
    # print((time.time()-init_time)*1000, ' ms, @whole.step')

    if terminated:
        print((time.time()-episode_start_times)*1000, ' ms, @episode generated')
        t  = time.time()
        episode_number += 1
        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        # Takes 45 ms
        ep_inputs = np.vstack(observation_deltas)
        ep_hidden_states = np.vstack(hidden_states_stack)
        epdlogp = np.vstack(dlogps)
        ep_rewards = np.vstack(rewards)
        # print((time.time()-t)*1000, ' ms, @stack_conversion')

        observation_deltas, hidden_states_stack, dlogps, rewards = [],[],[],[] # reset array memory


        # Discount and normalize rewards - 7ms
        t  = time.time()
        discounted_ep_rewards = discount_rewards(ep_rewards)
        discounted_ep_rewards -= np.mean(discounted_ep_rewards)
        discounted_ep_rewards /= np.std(discounted_ep_rewards)
        # print((time.time()-t)*1000, ' ms, @discounting rewards')

        # for reward in discounted_ep_rewards:
        #    print(reward)

        # modulate the gradient with advantage - Almost instant
        epdlogp *= discounted_ep_rewards

        # for dlogp in epdlogp:
        #    print(dlogp)

        t  = time.time()
        grad = policy_backward(ep_hidden_states, ep_inputs, epdlogp) # Backpropagate the gradient 
        # print((time.time()-t)*1000, ' ms, @backprop') - Takes about 30ms

        # for q, elm in grad.items():
        #    for j in elm:
        #       print(j)
        for k in range(0, len(model)): grad_buffer[k] += grad[k] # accumulate grad over batch 

        t  = time.time()
        if episode_number % batch_size == 0:
            for k,v in enumerate(model):
                g = grad_buffer[k] # gradient
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer
            # print((time.time()-t)*1000, ' ms, @rmsprop update') - about 10 ms at 50 batch size
        

        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print ('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
        running_means.append(running_reward)
        if episode_number % 100 == 0: 
           save_model()
           save_running_means()
           print(running_means)
        reward_sum = 0
        (observation, info) = env.reset() # reset env
        prev_x = None

        print(f'ep {episode_number}: game finished')
        episode_start_times = time.time()
    i += 1
    


# TODO: Nearly random got score of 54,000

# plt.imshow(observation)
# plt.show()



# for i in range(10000):
#     action = env.action_space.sample()  # agent policy that uses the observation and info
#     observation, reward, terminated, truncated, info = env.step(action)
#     # if reward > 0:
#     if i % 1000 == 0: print(observation)

#     if terminated or truncated:
#         observation, info = env.reset()

# TODO: Consider if we need to learn movement or if that can be implicit

env.close()


# Things to try
    # Done: Expand action space to 3 -> Right-fire, left-fire, no-op -> In order to do this I will need softmax
    # Done: Keep logs of running mean and graph it
    # Consider changing the input to include some history. A key point of difficulty in this game is figuring out how fast the ships are moving
    # Figure out how long it takes to generate an episode vs updating the NN
    # Try using my GPU
    # Try keeping the left gun in frame
    # Try eliminating downscaling
    # Try increasing the difficulty 
    # Change the plus to a minus in rmsprop update
    # Turn off sticky keys
    # Experiment with different batch sizes. Try 50

# I think that increasing the size of the NN shouldn't matter much since backprop is not costly. It might make forward prop take longer :(

# Key Elements of this env
    # The AI becomes good at the game when it learns how to predict the movement of the ships
        # They move at different speeds
    # Spam-firing is not allowed, you need no-ops between firing
        # It appears that the AI implicitly learns this, the most common action is a no-op
