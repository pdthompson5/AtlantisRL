from matplotlib import pyplot as plt
import os
import pickle

def load_running_means():
   return pickle.load(open(os.path.join(save_dir, 'running_means.p'), 'rb'))

save_name = "testing_saving"
save_dir = os.path.join("save_files", save_name)

running_means = load_running_means()

plt.scatter(range(0, len(running_means)), running_means)
plt.xlabel("Episode Number")
plt.ylabel("Average Running Reward")
plt.show()

