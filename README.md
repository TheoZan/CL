# Install

``pip install citylearn``

``pip install stable_baselines3``

``pip install sb3_contrib``

``pip install tensorboard``


# Run
Run the discrete.ipynb file to train an agent on a single building with only a battery.

## Reward
There is 3 ways to compute the reward
 * the basic reward of the environment: - net electricity consumption
 * custom_reward = 1: compute consumption * (price + emission) * zeta
 * custom_reward = 2: compute consumption_no_storage - consumption * (price + emission)
 * custom_reward = 3: same as custom_reward = 1, but different coding

# Results
The saved agents are located in the ``weights`` folder.

The saved information about the learning curves are located in the ``train`` folder.
Use tensorboard to vizualize the curves.


