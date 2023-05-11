import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn
import math

from citylearn.citylearn import CityLearnEnv
from citylearn.utilities import read_json

import citylearn
from citylearn.cost_function import CostFunction
from citylearn.energy_model import HeatPump

import gym
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import gym
from gym.spaces import Box
from stable_baselines3 import PPO, A2C, DDPG, TD3, SAC


from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

import argparse

def action_space_to_dict(aspace):
    """ Only for box space """
    return { "high": aspace.high,
             "low": aspace.low,
             "shape": aspace.shape,
             "dtype": str(aspace.dtype)
    }

def env_reset(env):
    observations = env.reset()
    action_space = env.action_space
    observation_space = env.observation_space
    building_info = env.get_building_information()
    building_info = list(building_info)
    action_space_dicts = [action_space_to_dict(asp) for asp in action_space]
    observation_space_dicts = [action_space_to_dict(osp) for osp in observation_space]
    obs_dict = {"action_space": action_space_dicts,
                "observation_space": observation_space_dicts,
                "building_info": building_info,
                "observation": observations }
    return obs_dict


class BuildingDevices:
  """
    Keeps track of all storage devices of a building.
  """
  def __init__(self, building, num_building):
    self.num_building = num_building
    self.building = building
    self.devices = {'battery' : Device_(building.electrical_storage, 'battery'),
                    'cooling' : None,
                    'dhw' : None}
    
  def compute_bounds(self):
    bounds = [self.bounds_action(i) for i,j in self.devices.items() if j is not None]
    return gym.spaces.Box(low=np.array([i[0] for i in bounds]), high=np.array([i[1] for i in bounds]), dtype=np.float64)
  
# ACTION 0 :  cooling
# ACTION 1 : dhw
# ACTION 2 : battery
  
  def bounds_action(self, type_action):
    device = self.devices[type_action].device
    if device is None:
        return None # if return none building doest have battery
    if type_action == 'battery':
        capacity = device.capacity_history[-2] if len(device.capacity_history) > 1 else device.capacity
        #HIGH
        #get max energy that the storage unit can use to charge [kW]
        #if trying to put more than the battery can accept reject action
        high1 = device.get_max_input_power()/capacity
        high2 = (device.capacity - device.soc_init)/(0.95*device.capacity) #approxim (efficiency = 0.95)
        high = min(high1, high2, 1)

        #LOW
        low1 = -device.get_max_input_power()/capacity
        low2 = (-device.soc_init*0.95)/device.capacity #approxim (efficiency = 0.95)
        low = max(low1, low2, -1)

    else:
        bool_h2, bool_l2 = False, False
        if type_action == 'cooling':
            # print('\ncooling')
            space_demand = self.building.cooling_demand[self.building.time_step]
            max_output = self.building.cooling_device.get_max_output_power(self.building.weather.outdoor_dry_bulb_temperature[self.building.time_step], False)
            # print('space_demand',space_demand)
            # print('max_output', max_output)
            # print('capacity:', device.capacity)
        else: #dhw
            # print('\ndhw')
            space_demand = self.building.dhw_demand[self.building.time_step]
            max_output = self.building.dhw_device.get_max_output_power(self.building.weather.outdoor_dry_bulb_temperature[self.building.time_step], False)\
            if isinstance(self.building.dhw_device, HeatPump) else self.building.dhw_device.get_max_output_power()
            # print('space_demand',space_demand)
            # print('max_output', max_output)
        space_demand = 0 if space_demand is None or math.isnan(space_demand) else space_demand # case where space demand is unknown

        #HIGH
        high1 = (max_output-space_demand) / device.capacity
        # print('high1', high1)
        if device.max_input_power is not None:
            bool_h2 = True
            high2 = device.max_input_power / device.capacity
            # print('high2', high2)
        high3 = (device.capacity - device.soc_init) / (device.capacity*device.efficiency)
        # print(device.capacity, device.soc_init)
        # print('high3', high3)
        
        if bool_h2:
            high = min(high1, high2, high3, 0.5)
        else:
            high = min(high1, high3, 0.5)


        #LOW
        low1 = -space_demand / device.capacity
        # print('low1', low1)
        if device.max_output_power is not None:
            bool_l2 = True
            low2 = -device.max_output_power / device.capacity
            # print('low2',low2)
        low3 = (-device.soc_init*device.efficiency) / device.capacity
        # print('low3',low3)

        if bool_l2:
            low = max(low1, low2, low3, -0.5)
        else:
            low = max(low1, low3, -0.5)

    return (low, high)
  

class Device_:
  def __init__(self, device, storage_type):
    self.device = device
    # self.price_cost = 0
    # self.emission_cost = 0
    self.cost = 0
    self.storage_type = storage_type

  def loss(self, cost_t, pv_offset, battery_offset):
    """
    get avg price between (battery release, grid release and PV- direct consumption)
    add relative incertainty, but true in pratice as the energy is added up in a global consumption pool 

    battery: if battery releases, price = avg((total released by battery - remaining conso), grid) in the case of thermal
    in the case of battery, avg price with PV
    """
    if not self.device:
      print('not device')
      raise ValueError

    energy_used = self.device.energy_balance[-1]
    if isinstance(energy_used, np.ndarray):
      print('probleme energy used array instead of float')
      energy_used = energy_used[0]

    #charge
    if energy_used > 0:
      #if pv production, part of the energy is free
      if pv_offset > 0:
        energy_used = max(0, energy_used-pv_offset)
      #if usage of battery, part of energy has been already taken into account so free
      if battery_offset > 0:
        energy_used = max(0, energy_used-battery_offset)
      # self.price_cost = ((self.price_cost*self.device.soc[-2])+(energy_used*price))/self.device.soc[-1]
      # self.emission_cost = ((self.emission_cost*self.device.soc[-2])+(energy_used*emission))/self.device.soc[-1]
      
      total = self.device.soc[-1]
      if isinstance(total, np.ndarray):
        print('probleme soc-1 array instead of float')
        total = total[0]

      prev = self.device.soc[-2]
      if isinstance(prev, np.ndarray):
        print('probleme soc-2 array instead of float')
        prev = prev[0]

      self.cost = ((self.cost*prev) + (energy_used*cost_t)) / total
      return energy_used, None, None #energy_used > 0

    #discharge
    else:
      #energy_processed is total energy used during charge/discharge process including losses
      #energy_used is the energy_processed minus the losses (used by building)
      energy_processed = self.device.soc[-2]-self.device.soc[-1]
      return -energy_used, energy_processed, self.cost # -energy_used > 0, energy_processed > 0 
    

def get_offset(building, mode):
  """
  building is env.buildings[i]:
  mode = 'pv' or 'battery'

  each conso gets an equally distributed offset based on solar generation or battery
  discharge
  """
  if mode == 'pv':
    if not building.solar_generation is None:
      return 0
    demands = [building.non_shiftable_load_demand[-2], building.electrical_storage.energy_balance[-1],
             building.dhw_demand[-2], building.dhw_storage.energy_balance[-1],
             building.cooling_demand[-2], building.cooling_storage.energy_balance[-1]]
    count = len([i for i in demands if i > 0])
    return -building.solar_generation[-2]/count
  else:
    if not building.solar_generation is None:
      return 0
    if building.electrical_storage.energy_balance[-1] >=0:
      return 0
    demands = [building.non_shiftable_load_demand[-2], building.dhw_demand[-2],
            building.dhw_storage.energy_balance[-1], building.cooling_demand[-2],
             building.cooling_storage.energy_balance[-1]]
    count = len([i for i in demands if i > 0])
    return -building.electrical_storage.energy_balance[-1]/count

def compute_loss(building, building_devices, price, emission, outdoor_dry_bulb_temperature, zeta):
  loss = 0
  pv_offset = get_offset(building, 'pv') 
  battery_offset = get_offset(building, 'battery')
  # print('pv offset',pv_offset)
  # print('battery_offset', battery_offset)

  #1) compute loss for storage devices use or update cost in storage
  for name,device in building_devices.devices.items():
    #if the device exists in building
    if device:
      energy_used, energy_processed, cost = device.loss(price*emission, pv_offset, battery_offset)
    #else consider it exists and set energy used = 0
    #so we can compute the remaining demand associated with the device
    else:
      energy_used = 0

    if not energy_processed: #charge
      #account for a part of the cost at charging time
      loss += (price * emission) * (energy_used * zeta)
    else: #discharge
      loss += cost * (energy_processed * (1 - zeta))

    #2) compute remaining thermal demand and add cost of remaining direct demand to answer
    if name == 'cooling':
      #cooling and dhw stored energy is thermal not electrical
      remaining = building.cooling_demand[-2] - energy_used
      # print('remaining', remaining)
      if remaining > 0:
        energy = max(0, building.cooling_device.get_input_power(remaining, outdoor_dry_bulb_temperature, False) - pv_offset - battery_offset)
        # print('energy', energy)
        loss += (price + emission) * energy

    elif name == 'dhw':
      remaining = building.dhw_demand[-2]
      # print('remaining', remaining)
      if remaining > 0:
        energy = max(0, building.dhw_device.get_input_power(remaining) - pv_offset - battery_offset)
        # print('energy', energy)
        loss += (price + emission) * energy

  #3) compute additionnal loss coming from nsl
  nsl = max(0, building.non_shiftable_load_demand[-2] - pv_offset - battery_offset)
  loss += (price + emission) * nsl
  # print(loss)

  return loss


class EnvCityGym(gym.Env):
    """
    Env wrapper coming from the gym library.
    """
    def __init__(self, env, devices, discrete, custom_reward, sum_cost, cost_ESU, zeta, stop=None):
        # print(schema_filepath)

        # new obs
        self.index_keep = [0,1,2,3,22,23,27]
        self.index_norm = [12,7,24,1,1,1,1,1]

        self.custom_reward = custom_reward
        self.sum_cost = sum_cost
        self.cost_ESU = cost_ESU
        self.zeta = zeta
        self.discrete = discrete

        self.env = env
        #list of names of devices [[]]
        self.devices = devices
        self.building_devices = []
        # get the number of buildings
        self.num_buildings = len(self.env.action_space)

        low = self.env.observation_space[0].low
        high = self.env.observation_space[0].high

        #if sum cost
        if self.sum_cost:
            cost_l = low[19]+low[28]
            cost_h = high[19]+high[28]

        d_low, d_high = [], []
        for i in devices[0]:
            if i == 'battery':
                d_low.append(low[26])
                d_high.append(high[26])
            elif i == 'cooling':
                d_low.append(low[24])
                d_high.append(high[24])
            elif i == 'dhw':
                d_low.append(low[25])
                d_high.append(high[25])

        low = [low[i] for i in self.index_keep]
        high = [high[i] for i in self.index_keep]

        low = low + d_low
        high = high + d_high

        #if sum cost
        if self.sum_cost:
            low.append(cost_l)
            high.append(cost_h)

        #if cost ESU, chage if multiple buildings
        if self.cost_ESU:
            for i in range(len(devices[0])):
                low.append(0)
                high.append(cost_h)

        if self.discrete:
            self.action_space = gym.spaces.Discrete(21)
            self.action_map = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,
                                0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        else:
            self.action_space = env.action_space[0]
        self.observation_space = gym.spaces.Box(low=np.array(low), high=np.array(high), dtype=np.float32)

        #keep last outdoor temp for each building
        self.temp = []
        self.stop = stop

        self.print_config()

        # TO THINK : normalize the observation space

    def reset(self):
        obs_dict = env_reset(self.env)
        obs = self.env.reset()

        for i,e in enumerate(self.env.buildings):
          self.building_devices.append(BuildingDevices(e,i))
          self.temp.append(obs[i][3])

        return self.get_obs(obs)

    def step(self, action):
        """
        we apply the same action for all the buildings
        """
        t = self.env.time_step
        # print('action', action,'\n')
        
        if self.discrete:
            action = [[self.action_conversion(action)]]
            action = action[0]
        action = [action]

        # we do a step in the environment
        obs, reward, done, info = self.env.step(action)
        if t == self.stop:
            done = True
        # print(obs[0][3])
        
        if self.custom_reward:
            rewards = []
            for i,e in enumerate(self.env.buildings):
                # print(env.temp)
                rewards.append(compute_loss(e, self.building_devices[i], self.env.buildings[i].pricing.electricity_pricing[t-1],
                self.env.buildings[i].carbon_intensity.carbon_intensity[t-1], self.temp[i], self.zeta))
                self.temp[i] = obs[i][3]
            # print(obs)
            #compute reward
            # print('reward', -reward[0])
            
            return np.array(self.get_obs(obs)), -rewards[0], done, info

        else:
            return np.array(self.get_obs(obs)), reward[0], done, info

    def get_obs(self, obs):
        obs_ = [[o[i]/n for i,n in zip(self.index_keep, self.index_norm)] for o in obs]
        # obs_ = list(itertools.chain(*obs_))

        for o in range(len(obs_)):
            if 'battery' in self.devices[o]:
                i = obs[o][26]
                if isinstance(i, np.ndarray):
                    print('probleme array instead of float soc battery obs')
                    i = i[0]
                obs_[o].append(i)
            if 'cooling' in self.devices[o]:
                obs_[o].append(obs[o][24])
            if 'dhw' in self.devices[o]:
                obs_[o].append(obs[o][25])

        if self.sum_cost is True:
            for o in range(len(obs_)):
                #modify for buildings that dont have same nb of obs
                obs_[o].append(obs[o][19]+obs[o][28])
            # print(obs)
        if self.cost_ESU is True:
            for o in range(len(obs_)):
                for name, device in self.building_devices[o].devices.items():
                    if device:
                        # print(device.cost)
                        obs_[o].append(device.cost)
        return np.array(obs_)
    
    def action_conversion(self, action):
        return self.action_map[action]
    
    def valid_action_mask(self):
        mod_action_space = self.building_devices[0].compute_bounds()
        act = np.array(self.action_map)
        index = list(np.where((act>mod_action_space.low[0]) & (act<mod_action_space.high[0]))[0])
        act = [True if i in index else False for i in range(21)]
        act[10] = True #noop always valid
        return act
    
    def print_config(self):
        print('INIT ENV:')
        act = 'Discrete' if self.discrete else 'Continuous'
        print(f'ACTION SPACE: {act}')
        print(f'Use of custom reward: {self.custom_reward}')
        if self.custom_reward:
            print(f'    zeta: {self.zeta}')
        print('Observations kept:')
        for i in self.index_keep:
            print(f'    {i}: {self.env.observation_names[0][i]}')
        for i in self.devices[0]:
            if i == 'battery':
                print('    26: '+self.env.observation_names[0][26])
            elif i == 'cooling':
                print('    24: '+self.env.observation_names[0][24])
            elif i == 'dhw':
                print('    25: '+self.env.observation_names[0][25])
        if self.sum_cost or self.cost_ESU:
            print(f'Observations ADDED:')
            if self.sum_cost:
                print(f'    sum_cost: {self.env.observation_names[0][19]} + {self.env.observation_names[0][28]}')
            if self.cost_ESU:
                print('    cost_ESU: see Device.loss')



def mask_fn(env: gym.Env) -> np.ndarray:
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.valid_action_mask()


def get_exp_name(model_name, env, total_timesteps):
    action_space = 'Discrete' if env.discrete else 'Continuous'
    reward = f'customR_{int(env.custom_reward)}'
    if env.custom_reward:
        reward += f'_zeta_{env.zeta}'
    equipment = 'devices'+'-'.join([str(len(i)) for i in env.devices])

    p = [model_name, str(env.num_buildings)+'building', equipment, action_space,
        reward, 'sum_cost_'+str(int(env.sum_cost)), 
        'cost_ESU_'+str(int(env.cost_ESU)), str(total_timesteps)]

    return '_'.join(p)

def train_save_model(model_name, devices, discrete, custom_reward, sum_cost,
                    cost_ESU, zeta, stop, checkpoint_path='./results', total_timesteps=None):

    # first we initialize the environment (petting zoo)
    if not total_timesteps:
        total_timesteps = 1_500_000

    schema_filepath = 'schema2.json'
    schema = read_json(schema_filepath)
    schema['root_directory'] = './'
    env = CityLearnEnv(schema)
    env = EnvCityGym(env, devices=devices, discrete=discrete, custom_reward=custom_reward,
                        sum_cost=sum_cost, cost_ESU=cost_ESU, zeta=zeta, stop=stop)
    if 'mask' in model_name:
        env = ActionMasker(env, mask_fn)
    obs = np.array(env.reset())

    exp_name = get_exp_name(model_name, env,total_timesteps)
    # load model if exist
    if model_name == 'ppo_mask':
        model = MaskablePPO(MaskableActorCriticPolicy, env,
                        verbose=1, tensorboard_log='./train', device='cuda')
    elif model_name == 'ppo':
        model = PPO('MlpPolicy', env, verbose=1, gamma=0.99, tensorboard_log="./train/", device='cuda')
    else:
        print('model not recognized')
        return None

    print(f'Model: {model_name}')
    # Train the agent
    model.learn(total_timesteps=total_timesteps, tb_log_name=exp_name,log_interval=5)

    print('saving model')
    model.save(checkpoint_path+exp_name+'.zip')
    return model


def main():
    parser = argparse.ArgumentParser(description='Training of RL model')
    # parser.add_argument('-p','--part',
    #                     help='part of zeta space 0 or 1',
    #                     required=True)
    parser.add_argument('-m','--model_name',
                        help='model ppo or ppo_mask',
                        required=True)
    
    # args = vars(parser.parse_args())
    # if args['part'] == '0':
    #    l_zeta = [0,0.1,0.2,0.3,0.4,0.5]
    # else:
    #    l_zeta = [0.6,0.7,0.8,0.9,1]
    l_zeta = [0.6]
    for zeta in l_zeta:
        _ = train_save_model(model_name=args['model_name'] , devices=[['battery']], discrete=True,
                    custom_reward=True, sum_cost=True, cost_ESU=True, zeta=zeta,
                    stop=None, checkpoint_path='./weights/', total_timesteps=1_500_000)


if __name__ == "__main__":
    
    main()