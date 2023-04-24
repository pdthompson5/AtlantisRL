import numpy as np
import pickle
import gymnasium as gym
from typing import Tuple
from matplotlib import pyplot as plt


# Things to do:
# Save necessary information from episode generation 

A = 4 # Action space -> 0-3
H = 200 # number of hidden layer neurons
batch_size = 10 # used to perform a RMS prop param update every batch_size steps
learning_rate = 1e-3 # Learning rate
gamma = 0.99 # Discount factor
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2

# Config flags - video output and res
resume = False # resume training from previous checkpoint (from save.p  file)?
render = False # render video output?

D = 60 * 80 # input dimensionality: 60x80 grid

def initialize_model():
    model = {}
    model['W1'] = np.random.randn(D,H) / np.sqrt(D) # "Xavier" initialization
    model['W2'] = np.random.randn(H,A) / np.sqrt(H) # Shape will be H
    return model

def save_model():
    pickle.dump(model, open('save.p', 'wb'))
def load_model():
    return pickle.load(open('save.p', 'rb'))


model = initialize_model() if not resume else load_model()

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

# softmax from: https://gist.github.com/etienne87/6803a65653975114e6c6f08bb25e1522
def softmax(x):
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    return probs

# TODO: Will I need to change this because of the larger action space?
# TODO: The initial impl of this will just consider left or right fire, no center
def policy_forward(x: np.ndarray) -> Tuple[float, np.ndarray]:
  """Forward propagation. Input is preprocessed input: 60x80 float column vector"""
  if(len(x.shape)==1):
    x = x[np.newaxis,...]

  hidden_states = x.dot(model['W1']) # (H x D) . (D x 1) = (H x 1) (200 x 1)
  hidden_states[hidden_states < 0] = 0 # ReLU introduces non-linearity
  logp = hidden_states.dot(model['W2']) # This is a logits function and outputs a decimal.   (1 x H) . (H x 1) = 1 (scalar)
#   sigmoid_prob = sigmoid(logp)  # squashes output to  between 0 & 1 range
  probs = softmax(logp)
  print(probs)
#   print("sigmoid", sigmoid_prob)
  return probs, hidden_states # return probability of taking action 2 (RIGHTFIRE), and hidden state

def policy_backward(eph, epx, epdlogp):
  """ Manual implementation of a backward prop"""
  """ It takes an array of the hidden states that corresponds to all the images that were
  fed to the NN (for the entire episode, so a bunch of games) and their corresponding logp"""
  dW2 = np.dot(eph.T, epdlogp).ravel()
  dh = np.outer(epdlogp, model['W2'])
  dh[eph <= 0] = 0 # backpro prelu
  dW1 = np.dot(dh.T, epx)
  return {'W1':dW1, 'W2':dW2}



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


grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory

observation: np.ndarray
(observation, info) = env.reset()

prev_observation = None # used in computing the difference frame
observation_deltas, hidden_states_stack, dlogps, rewards = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0

i = 0
while(True):
    if render: env.render()

    cur_observation = preprocess(observation)
    observation_delta: np.ndarray = cur_observation - prev_observation if prev_observation is not None else np.zeros(D)
    prev_observation = cur_observation

    # forward prop
    (action_probs, hidden_states) = policy_forward(cur_observation)

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

    observation_deltas.append(observation_delta)
    hidden_states_stack.append(hidden_states)

    dlogsoftmax = action_probs.copy()
    # This represents the difference between the action probabilities that we received and action probabilities that 
    # would always result in the action that we chose
    dlogsoftmax[0,action] -= 1
    dlogps.append(dlogsoftmax)


    # TODO: For some reason we need to no-op between actions. holding down the button doesn't work I guess
    # It seems like this is for preventing button spamming.
    observation, reward, terminated, truncated, info = env.step(action)
    reward_sum += reward
    rewards.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

    if terminated:
        episode_number += 1
        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        ep_inputs = np.vstack(observation_deltas)
        ep_hidden_states = np.vstack(hidden_states_stack)
        epdlogp = np.vstack(dlogps)
        ep_rewards = np.vstack(rewards)

        observation_deltas, hidden_states_stack, dlogps, rewards = [],[],[],[] # reset array memory


        # Discount and normalize rewards
        discounted_ep_rewards = discount_rewards(ep_rewards)
        discounted_ep_rewards -= np.mean(discounted_ep_rewards)
        discounted_ep_rewards /= np.std(discounted_ep_rewards)

        # for reward in discounted_ep_rewards:
        #    print(reward)

        # modulate the gradient with advantage
        epdlogp *= discounted_ep_rewards

        # for dlogp in epdlogp:
        #    print(dlogp)


        grad = policy_backward(ep_hidden_states, ep_inputs, epdlogp) # Backpropagate the gradient 

        # for q, elm in grad.items():
        #    for j in elm:
        #       print(j)
        for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch 

        if episode_number % batch_size == 0:
            for k,v in model.items():
                g = grad_buffer[k] # gradient
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer


        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print ('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
        if episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))
        reward_sum = 0
        (observation, info) = env.reset() # reset env
        prev_x = None

        print(f'ep {episode_number}: game finished')
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
    # Expand action space to 3 -> Right-fire, left-fire, no-op -> In order to do this I will need softmax
