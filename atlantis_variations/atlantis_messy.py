import numpy as np
import pickle
import gymnasium as gym
from typing import Tuple, Dict
import os
import time

A = 4 # Action space: 0-3
H = 200 # number of hidden layer neurons
batch_size = 10 # used to perform a RMS prop param update every batch_size steps
learning_rate = 1e-4 # Learning rate
gamma = 0.99 # Discount factor
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2

# Config flags - video output and res
resume = True # resume training from previous checkpoint (from save.p  file)?
render = True # render video output?

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

# TODO: Will I need to change this because of the larger action space?
# TODO: The initial impl of this will just consider left or right fire, no center

# Takes about 2 ms
def policy_forward(x: np.ndarray) -> Tuple[float, np.ndarray]:
  """Forward propagation. Input is preprocessed input: 52x80 float column vector"""
  
  if(len(x.shape)==1):
    x = x[np.newaxis,...]

  t = time.time()
  hidden_states = np.dot(x, model['W1']) # TODO: This is the slow part of this function
  hidden_states[hidden_states < 0] = 0 # ReLU introduces non-linearity
  logp = np.dot(hidden_states, model['W2']) # This is a logits function and outputs a decimal.   (1 x H) . (H x 1) = 1 (scalar)
#   print((time.time()-t)*1000, ' ms, @non-softmax')
  return softmax(logp), hidden_states # return probability of taking action 2 (RIGHTFIRE), and hidden state

def policy_backward(eph: np.ndarray, epx: np.ndarray, epdlogp: np.ndarray):
  """ Manual implementation of a backward prop"""
  """ It takes an array of the hidden states that corresponds to all the images that were
  fed to the NN (for the entire episode, so a bunch of games) and their corresponding logp"""
  dW2 = np.dot(eph.T, epdlogp)
  dh = np.dot(epdlogp, model['W2'].T)
  dh[eph <= 0] = 0 # backpro prelu
  dW1 = epx.T.dot(dh)
  return {'W1':dW1, 'W2':dW2}



# TODO: I wonder if I should increase the difficulty to make it learn better?
# TODO: Calculate the performance of the random policy
# env = gym.make("ALE/Atlantis-v5", render_mode="human")


render_mode = "human" if render else "rgb_array"
# register(
#      id="ALE/Atlantis-v5-no-limit",
#      entry_point="ale_py.env.gym:AtariEnv",
# )
env = gym.make("ALE/Atlantis-v5", render_mode=render_mode)

# TODO: Might need to handle screen flashes

def preprocess(observation: np.ndarray):
    """ preprocess 210x160x3 uint8 frame into 10944 (76x144) float column vector """
    observation = observation[16:92, 8:152] # Delete all areas where the ships can never be
    observation = observation[::,::,0]
    observation[observation != 0] = 1 # Make grayscale
    # if(p % 400 == 0):
    #     plt.imshow(observation)
    #     plt.show()
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

time_step = 3_640_062
episode_time_step = 0
# TODO: Train at 1mil then combine the running means at a point that makes sense
# Episode generate takes ~2.5 seconds
    # Per total step env step takes 1ms, forward prop takes 2ms
episode_start_times = time.time()
should_terminate = False
prev_checkpoint_reward = -1
while(True):
    episode_time_step += 1
    init_time = time.time()

    t  = time.time()
    cur_observation = preprocess(observation)
    observation_delta: np.ndarray = cur_observation - prev_observation if prev_observation is not None else np.zeros(D)
    prev_observation = cur_observation

    # print((time.time()-t)*1000, ' ms, @prepo')

    # forward prop
    # t  = time.time()
    (action_probs, hidden_states) = policy_forward(observation_delta)
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

    # if time_step == 0:
    #    save_checkpoint("init")
    # if time_step == 10000:
    #    save_checkpoint("10k")
    # if time_step == 100000:
    #    save_checkpoint("100k")
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

    # Truncate so infinite episode will cease
    # TODO: Try just discarding these long episodes?
    if(episode_time_step > 100_000):
        truncated = True

    #TODO: Truncating while training is probably for the best when you have a model capable of infinite steps
    #TODO: I think what worked very well before is just throwing away the infinite games 
    #TODO: Maybe truncate at 15_000?
    if terminated or truncated:
        # print((time.time()-episode_start_times)*1000, ' ms, @episode generated')

        t  = time.time()
        episode_number += 1
        episode_time_step = 0
        
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

        # modulate the gradient with advantage 
        epdlogp *= discounted_ep_rewards

        t  = time.time()
        grad = policy_backward(ep_hidden_states, ep_inputs, epdlogp) # Backpropagate the gradient 

        for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch 

        t  = time.time()
        if episode_number % batch_size == 0:
            for k,v in model.items():
                g = grad_buffer[k] # gradient
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer
            # print((time.time()-t)*1000, ' ms, @rmsprop update') - about 10 ms at 50 batch size
        

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
    

# Key Elements of this env
    # The AI becomes good at the game when it learns how to predict the movement of the ships
        # They move at different speeds
    # Spam-firing is not allowed, you need no-ops between firing
        # It appears that the AI implicitly learns this, the most common action is a no-op
