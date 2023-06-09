# AtlantisRL
This project is a implementation of the REINFORCE policy gradient algorithm for training an agent to perform well at the game Atlantis for the Atari 2600. It utilizes OpenAI Gymnasium for its environment. A video of a trained model playing the game can be found here: [https://youtu.be/OGztMCp0YdY](https://youtu.be/OGztMCp0YdY)

## Dependencies
The python requirements for running this project are listed in `requirements.txt`. You may also need to install Atari ROMs for Gymnasium. To prevent conflicts, I recommend installing these requirements in a virtual environment.

## Running
The main entry point for this project is `atlantis.py`. It can be used to train the agents. Run this entrypoint with the command `python3 atlantis.py`. Note that there are important configuration options at the top of the file. 

`evaluate_performance.py` can be used to evaluate the performance of pre-trained agents. The script generates 100 episodes using the agent and prints the average game score. Episodes are terminated after 2 million points are achieved.

`graph_returns.py` can be used to generate graphs of the average running returns over training for a given model.

## Pre-trained models
Included in this repository are three pre-trained models. `model1` is the best trained model. It is capable of consistent infinite games. `model2` is capable of infinite games, but it is not consistent. `pixels_only` is a converged model that utilizes the pixels of the game as input rather than the difference between the pixels. To use this last model modify line 122 of `atlantis.py` to 
`(action_probs, hidden_states) = policy_forward(cur_observation)`.

## Running infinite episodes
To run infinite episodes, you must disable episode truncation in Gymnasium. You can accomplish this via the following steps:
1. Install the requirements
2. Find the package `shimmy` in your python site-packages folder.
3. Find the file `registration.py`
4. Change line 220 to `"max_num_frames_per_episode": None,`
