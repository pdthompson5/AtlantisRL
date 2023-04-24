""" Majority of this code was copied directly from Andrej Karpathy's gist:
https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5 """

""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import pickle
import gymnasium as gym

# TODO: It appears that this no longer works with the newer versions

from gymnasium import wrappers
# TODO: rmsprop is important -> root mean square propagation
# hyperparameters to tune
H = 200 # number of hidden layer neurons
batch_size = 10 # used to perform a RMS prop param update every batch_size steps
# When should we update the NNs
learning_rate = 1e-3 # learning rate used in RMS prop
# Alpha 
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2

# Config flags - video output and res
resume = False # resume training from previous checkpoint (from save.p  file)?
render = False # render video output?

# model initialization
D = 75 * 80 # input dimensionality: 75x80 grid
if resume:
  model = pickle.load(open('save.p', 'rb'))
else:
  model = {}
  # TODO: Why is there a w1 and a w2? -> There are two layers to the neural network
  # TODO why are you dividing?
  model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization - Shape will be H x D
  model['W2'] = np.random.randn(H) / np.sqrt(H) # Shape will be H

grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory

# Normalize in the interval of 0 and 1
def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

# Extract features from the image
# Mostly just decreasing the dimensions of the inputs
def prepro(I: np.ndarray):
  """ prepro 210x160x3 uint8 frame into 6000 (75x80) 1D float vector """
  # I is a 210x160 array of RGB values
  I = I[35:185] # crop - remove 35px from start & 25px from end of image in x, to reduce redundant parts of image (i.e. after ball passes paddle)
  
  I = I[::2,::2,0] # downsample by factor of 2. I think that down sampling just gets rid of every second pixel on height and width
  I[I == 144] = 0 # erase background (background type 1)
  
  I[I == 109] = 0 # erase background (background type 2)
  # I think that this eliminates all colors that are the background
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1. this makes the image grayscale effectively
  temp = I.astype(float).ravel()
  return temp# ravel flattens an array and collapses it into a column vector

# Just apply gamma to all fot he rewards
def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  """ this function discounts from the action closest to the end of the completed game backwards
  so that the most recent action has a greater weight """
  # TODO: Is the array that this returns have rewards corresponding to reversed timestamps? A: No, it just starts at the end
  # how this works: reward is zero for all times except game boundaries
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)): # xrange is no longer supported in Python 3, replace with range
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

# This function is the policy function
def policy_forward(x):
  """This is a manual implementation of a forward prop"""
  # Forward prop is basically just providing an input and getting an output from the NN
  # Input: X is the input to the nn, likely the state

  # TODO: What is h?
  h = np.dot(model['W1'], x) # (H x D) . (D x 1) = (H x 1) (200 x 1)
  h[h<0] = 0 # ReLU introduces non-linearity
  # ReLU -> if less then 0 then 0
  # TODO: What is W2, I still don't know
  logp = np.dot(model['W2'], h) # This is a logits function and outputs a decimal.   (1 x H) . (H x 1) = 1 (scalar)
  p = sigmoid(logp)  # squashes output to  between 0 & 1 range
  return p, h # return probability of taking action 2 (UP), and hidden state

# Apparently this is pretty standard 
# Normally you would just use PyTorch or tensorflow
def policy_backward(eph, epx, epdlogp):
  """ backward pass. (eph is array of intermediate hidden states) """
  """ Manual implementation of a backward prop"""
  """ It takes an array of the hidden states that corresponds to all the images that were
  fed to the NN (for the entire episode, so a bunch of games) and their corresponding logp"""
  dW2 = np.dot(eph.T, epdlogp).ravel() #each state times delta J?
  dh = np.outer(epdlogp, model['W2'])
  dh[eph <= 0] = 0 # backpro prelu
  dW1 = np.dot(dh.T, epx)
  return {'W1':dW1, 'W2':dW2}

env = gym.make("Pong-v0")
# env = gym.make("ALE/Pong-v5")

#env = wrappers.Monitor(env, 'tmp/pong-base', force=True) # record the game as as an mp4 file
(observation, info) = env.reset()
# TODO: observation is no longer just an ndarray
prev_x = None # used in computing the difference frame
xs,hs,dlogps,drs = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0
while True:
  if render: env.render()

  # preprocess the observation, set input to network to be difference image
  cur_x = prepro(observation)
  # we take the difference in the pixel input, since this is more likely to account for interesting information
  # e.g. motion
  # x is the difference in pixels from one time step to another. It used as the input to the nn
  x: np.ndarray = cur_x - prev_x if prev_x is not None else np.zeros(D)
  prev_x = cur_x

  # forward the policy network and sample an action from the returned probability
  # Aprob is action probability, what is h?
  # I think that h is probably just a state that the episode passed through. It is stored because it needs to be updated
  result_tuple = policy_forward(x)
  # This gets an example action and all of the hidden states associated with it

  aprob: np.float64 = result_tuple[0]
  h: np.ndarray = result_tuple[1]

  # h has shape of 200
  # Possibly h is just all of the hidden neurons that were traversed
  # print(h.shape)
  # print(aprob)
  # The following step is randomly choosing a number which is the basis of making an action decision
  # If the random number is less than the probability of UP output from our neural network given the image
  # then go down.  The randomness introduces 'exploration' of the Agent
  action = 2 if np.random.uniform() < aprob else 5 # roll the dice! 2 is UP, 3 is DOWN, 0 is stay the same

  # record various intermediates (needed later for backprop).
  # This code would have otherwise been handled by a NN library
  xs.append(x) # observation
  hs.append(h) # hidden state (TODO: is this just the state, what's up with this?)
  # TODO Why are we using a fake label?
  y = 1 if action == 2 else 0 # a "fake label" - this is the label that we're passing to the neural network
  # to fake labels for supervised learning. It's fake because it is generated algorithmically, and not based
  # on a ground truth, as is typically the case for Supervised learning

  # I strongly suspect that dlogp is the gradient of the log of the policy function 
  dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

  # step the environment and get new measurements
  observation, reward, done, truncated, info = env.step(action)
  reward_sum += reward
  drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

  if done: # an episode finished
    episode_number += 1

    # stack together all inputs, hidden states, action gradients, and rewards for this episode
    
    # Episode inputs
    epx = np.vstack(xs)

    # Episode hidden states?. Why is this useful? What is h?
    eph = np.vstack(hs)

    # Action gradients
    epdlogp = np.vstack(dlogps)

    # Episode rewards
    epr = np.vstack(drs)
    # reward stacked
    xs,hs,dlogps,drs = [],[],[],[] # reset array memory


    # This next section is for discounting the rewards and normalizing them a bit 

    # compute the discounted reward backwards through time
    # Discounted epr = is discounted episode reward
    discounted_epr = discount_rewards(epr)

    discounted_epr -= np.mean(discounted_epr)
      # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    # This is the baselining thing
    discounted_epr /= np.std(discounted_epr)

    # the result of this is almost delta theta. All you need now is discounting factor and step size
    epdlogp *= discounted_epr # modulate the gradient with advantage (Policy Grad magic happens right here.)
    # This is the policy gradient theorem

    grad = policy_backward(eph, epx, epdlogp) # Backpropagate the gradient 
    for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch

    # perform rmsprop parameter update every batch_size episodes
    # If we should update the NN this round
    # Batch size is usually constrained by memory size also in how many samples can be generated
    # TODO: Is the NN only updated every 10 episodes? seems like 
    if episode_number % batch_size == 0:
      for k,v in model.items():
        g = grad_buffer[k] # gradient
        print(g.shape)
        rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
        model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
        grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

    # boring book-keeping
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    print ('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
    if episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))
    reward_sum = 0
    (observation, info) = env.reset() # reset env
    prev_x = None

    if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
        print(f'ep {episode_number}: game finished, reward: {reward}', '' if reward == -1 else ' !!!!!!!!')

