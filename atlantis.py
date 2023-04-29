import numpy as np
import pickle
import gymnasium as gym
from typing import Tuple, Dict
import os
import time

A = 4 # Action space: 0-3
H = 200 # number of hidden layer neurons
batch_size = 10 # used to perform a RMS prop param update every batch_size steps
learning_rate = 1e-3 # Learning rate
gamma = 0.99 # Discount factor
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2

# Config flags - video output and res
resume = True # resume training from previous checkpoint (from save.p  file)?
render = True # render video output?
should_truncate = False # Truncates episodes to 100k steps

save_name = "delta_proper_small_batch"
save_dir = os.path.join("save_files", save_name)

if not os.path.exists(save_dir):
   os.makedirs(save_dir)


D = 76 * 144 # input dimensionality: 76x144 grid
p = 0
def initialize_model():
    model: Dict[str: np.ndarray] = {}
    model['W1'] = np.random.randn(D,H) / np.sqrt(D) # "Xavier" initialization
    model['W2'] = np.random.randn(H,A) / np.sqrt(H) # Shape will be H
    return model

def load_model():
    return pickle.load(open(os.path.join(save_dir, 'save.p'), 'rb'))
def load_running_means():
   return pickle.load(open(os.path.join(save_dir, 'running_means.p'), 'rb'))

model = initialize_model() if not resume else load_model()
running_means = [] if not resume else load_running_means()

def save_model(path):
    pickle.dump(model, open(path, 'wb'))
def save_running_means(path):
    pickle.dump(running_means, open(path, "wb"))

def save_checkpoint(checkpoint_name: str):
    os.makedirs(os.path.join(save_dir, checkpoint_name), exist_ok=True)
    save_model(os.path.join(save_dir, checkpoint_name, 'save.p'))
    save_running_means(os.path.join(save_dir, checkpoint_name, 'running_means.p'))

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
  logp = np.dot(hidden_states, model['W2'])
  return softmax(logp), hidden_states

def policy_backward(eph: np.ndarray, epx: np.ndarray, epdlogp: np.ndarray):
  """ Manual implementation of a backward prop"""
  dW2 = np.dot(eph.T, epdlogp)
  dh = np.dot(epdlogp, model['W2'].T)
  dh[eph <= 0] = 0 # backpro prelu
  dW1 = epx.T.dot(dh)
  return {'W1':dW1, 'W2':dW2}


render_mode = "human" if render else "rgb_array"
env = gym.make("ALE/Atlantis-v5", render_mode=render_mode)

def preprocess(observation: np.ndarray):
    """ preprocess 210x160x3 uint8 frame into 10944 (76x144) float column vector """
    observation = observation[16:92, 8:152] # Delete all areas where the ships can never be
    observation = observation[::,::,0]
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
running_reward = None if not resume else running_means[-1]
reward_sum = 0
episode_number = 0

time_step = 0
episode_time_step = 0
should_terminate = False
prev_checkpoint_reward = -1
while(True):
    episode_time_step += 1

    cur_observation = preprocess(observation)
    observation_delta: np.ndarray = cur_observation - prev_observation if prev_observation is not None else np.zeros(D)
    prev_observation = cur_observation

    # forward prop
    (action_probs, hidden_states) = policy_forward(observation_delta)

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
    dlogsoftmax[0] *= -1
    dlogps.append(dlogsoftmax)

    observation, reward, terminated, truncated, info = env.step(action)
    reward_sum += reward
    rewards.append(reward) # record reward (has to be done after we call step() to get reward for previous action)


    if time_step == 0:
       save_checkpoint("init")
    if time_step == 10000:
       save_checkpoint("10k")
    if time_step == 100000:
       save_checkpoint("100k")
    if time_step == 1000000:
       save_checkpoint(f"1Mil_{episode_number}")
    if time_step == 10000000:
       save_checkpoint(f"10Mil_{episode_number}")
    if time_step == 50000000:
       save_checkpoint(f"50Mil_{episode_number}")
    if time_step == 50000000:
       save_checkpoint(f"100Mil_{episode_number}")
    if time_step % 10000 == 0:
       print(f"Current Reward: {reward_sum}")
       print(f"Episode time step: {episode_time_step}")

    # Truncate so infinite episodes will cease
    if(should_truncate and episode_time_step > 100_000):
        truncated = True

    if terminated or truncated:
        episode_number += 1
        episode_time_step = 0
        
        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        ep_inputs = np.vstack(observation_deltas)
        ep_hidden_states = np.vstack(hidden_states_stack)
        epdlogp = np.vstack(dlogps)
        ep_rewards = np.vstack(rewards)

        observation_deltas, hidden_states_stack, dlogps, rewards = [],[],[],[] # reset array memory

        discounted_ep_rewards = discount_rewards(ep_rewards)
        discounted_ep_rewards -= np.mean(discounted_ep_rewards)
        discounted_ep_rewards /= np.std(discounted_ep_rewards)

        # modulate the gradient with advantage 
        epdlogp *= discounted_ep_rewards

        grad = policy_backward(ep_hidden_states, ep_inputs, epdlogp) # Backpropagate the gradient 

        for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch 

        if episode_number % batch_size == 0:
            for k,v in model.items():
                g = grad_buffer[k] # gradient
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer
        
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print ('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
        running_means.append(running_reward)


        if episode_number == 100:
           save_checkpoint("100_episodes")
        if episode_number == 200:
           save_checkpoint("200_episodes")
        if episode_number % 2 == 0: 
           save_model(os.path.join(save_dir, 'save.p'))
           save_running_means(os.path.join(save_dir, 'running_means.p'))
        reward_sum = 0
        (observation, info) = env.reset() # reset env
        prev_observation = None

        print(f'ep {episode_number}: game finished, time step: {time_step}')
        episode_start_times = time.time()
    time_step += 1
