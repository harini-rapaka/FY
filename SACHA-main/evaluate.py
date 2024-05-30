import argparse
import pickle
from tqdm import tqdm
import numpy as np
import torch
from environment import POMAPFEnv
from model import AttentionPolicy
import matplotlib.pyplot as plt
import csv

# config
import yaml
config = yaml.safe_load(open("./configs/config_base.yaml", 'r'))
config = config | yaml.safe_load(open("./configs/config_eval.yaml", 'r'))
OBSTACLE, FREE_SPACE = config['grid_map']['OBSTACLE'], config['grid_map']['FREE_SPACE']
num_instances_per_test = config['num_instances_per_test']
test_settings = config['test_settings']
max_timesteps = config['max_timesteps']
hidden_dim = config['hidden_dim']

def read_csv(filename):
    data = {'success_rate': {}, 'avg_step': {}}  
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            map_name = row[2]
            num_agents = int(row[5])
            success_rate = float(row[6])  
            avg_step = float(row[7])  
            if map_name not in data['success_rate']:
                data['success_rate'][map_name] = {}
                data['avg_step'][map_name] = {}
            data['success_rate'][map_name][num_agents] = success_rate
            data['avg_step'][map_name][num_agents] = avg_step
    return data


sacha_data = read_csv('log/sacha.csv')
sacha_ADAM_data = read_csv('log/adam.csv')
sacha_ADAM_HEIR_data = read_csv('log/heir.csv')


def test_one_case(model, grid_map, starts, goals, horizon):    
    env = POMAPFEnv(config)
    env.load(grid_map, starts, goals)
    obs = env.observe()
    step = 0
    num_agents = len(starts)
    hidden = torch.zeros(num_agents, hidden_dim, dtype=float)
    paths = [[] for _ in range(num_agents)]
    while step <= horizon:
        curr_state = env.curr_state
        for i, loc in enumerate(curr_state):
            paths[i].append(tuple(loc))
        action, _, _ = model(torch.from_numpy(obs), hidden, curr_state)
        obs, reward, done, _, info = env.step(action)
        for i, loc in enumerate(curr_state):
            paths[i].append(tuple(loc))
        if all(done):
            break
        step += 1
    avg_step = 0.0
    for i in range(num_agents):
        while (len(paths[i]) > 1 and paths[i][-1] == paths[i][-2]):
            paths[i] = paths[i][:-1]
        avg_step += len(paths[i]) / num_agents
    return np.array_equal(env.curr_state, env.goal_state), avg_step, paths


def main(args):
    state_dict = torch.load(args.load_from_dir)
    model = AttentionPolicy(args.communication)
    model.load_state_dict(state_dict['policy']['target_policy'])
    num_instances = num_instances_per_test
    for map_name, num_agents in test_settings:
        file_name = f"./benchmarks/test_set/{map_name}_{num_agents}agents.pth"
        with open(file_name, 'rb') as f:
            instances = pickle.load(f)
        print(f"Testing instances for {map_name} with {num_agents} agents ...")
        success_rate, avg_step = 0.0, 0.0
        for grid_map, starts, goals in tqdm(instances[0: num_instances]):
            done, steps, paths = test_one_case(model, np.array(grid_map), list(starts), list(goals), max_timesteps[map_name])
            if done:
                success_rate += 1 / num_instances
                avg_step += steps / num_instances
            else:
                avg_step += max_timesteps[map_name] / num_instances
        with open(f"./log/SACHA/heir.csv", 'a+') as f:
            height, width = np.shape(grid_map)
            num_obstacles = sum([row.count(OBSTACLE) for row in grid_map])
            method_name = 'SAHCA' if not args.communication else 'SACHA(C)'
            f.write(f"{method_name},{num_instances},{map_name},{height * width},{num_obstacles},{num_agents},{success_rate},{avg_step}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--communication", action='store_true')
    parser.add_argument("--load_from_dir", default="")
    args = parser.parse_args()
    main(args)

for map_name in sacha_data['success_rate'].keys():
    plt.figure(figsize=(12, 6))
    
    # Plot success rate
    plt.subplot(1, 2, 1)
    plt.plot(list(sacha_data['success_rate'][map_name].keys()), list(sacha_data['success_rate'][map_name].values()), marker='o', label='sacha')
    plt.plot(list(sacha_ADAM_data['success_rate'][map_name].keys()), list(sacha_ADAM_data['success_rate'][map_name].values()), marker='o', label='sacha_ADAM')
    plt.plot(list(sacha_ADAM_HEIR_data['success_rate'][map_name].keys()), list(sacha_ADAM_HEIR_data['success_rate'][map_name].values()), marker='o', label='sacha_ADAM_HEIR')
    plt.title(f'Success Rate Comparison for {map_name}')
    plt.xlabel('Number of Agents')
    plt.ylabel('Success Rate (%)')
    plt.xticks(list(sacha_data['success_rate'][map_name].keys()))
    plt.legend()
    plt.grid(True)
    
    # Plot avg step
    plt.subplot(1, 2, 2)
    plt.plot(list(sacha_data['avg_step'][map_name].keys()), list(sacha_data['avg_step'][map_name].values()), marker='o', label='sacha')
    plt.plot(list(sacha_ADAM_data['avg_step'][map_name].keys()), list(sacha_ADAM_data['avg_step'][map_name].values()), marker='o', label='sacha_ADAM')
    plt.plot(list(sacha_ADAM_HEIR_data['avg_step'][map_name].keys()), list(sacha_ADAM_HEIR_data['avg_step'][map_name].values()), marker='o', label='sacha_ADAM_HEIR')
    plt.title(f'Average Step Comparison for {map_name}')
    plt.xlabel('Number of Agents')
    plt.ylabel('Average Step')
    plt.xticks(list(sacha_data['avg_step'][map_name].keys()))
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
