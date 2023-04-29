from matplotlib import pyplot as plt
import os
import pickle

def load_running_means():
   with open(os.path.join(save_dir, 'running_means.p'), 'rb') as f:
      running_means = pickle.load(f)
   return running_means

save_name = "delta_proper_small_batch"
save_dir = os.path.join("save_files", save_name)

running_means = load_running_means()
plt.scatter(range(0, len(running_means)), running_means)
plt.xlabel("Episode Number")
plt.ylabel("Average Running Reward")
plt.title("Average Running Return Over Training")
plt.show()

