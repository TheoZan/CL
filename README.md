# Install

1 fork the repository

2 install the following

``pip install citylearn``

``pip install stable_baselines3``

``pip install sb3_contrib``

``pip install tensorboard``


# Run
Run the discrete.ipynb file to train an agent on a single building with only a battery.

## Reward
There is 3 ways to compute the reward
 * the basic reward of the environment: ``- net electricity consumption``
 * custom_reward = 1: ``-consumption * (price + emission) * zeta`` (not stable use custome_reward=3)
 * custom_reward = 2: ``- consumption_no_storage - consumption * (price + emission)``
 * custom_reward = 3: same as custom_reward = 1, but different coding (stable)

# Results
The saved agents are located in the ``weights`` folder.

The saved information about the learning curves are located in the ``train`` folder.
Use tensorboard to vizualize the curves. (on vs code CTRL+MAJ+P, Python: Launch TensorBoard, select ./train folder)


# Clustering
A module is implemented to cluster buildings based on their Non-shiftable load (NSL).
Run the test_cluster.ipynb file to apply time series clustering on the buildings of the first climate zone.
We defined k=3 clusters grouping the buildings. The value of k was found empirically.
