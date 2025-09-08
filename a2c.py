import os
import glob
import random
import time
import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.nn.utils import clip_grad_norm_

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- Action space 설정 ---
steer_options_deg = [-12, -8, -4, 0, 4, 8, 12]
steer_options = [np.radians(a) for a in steer_options_deg]
speed_options = [4.5, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]

NUM_STEER_ACTIONS = len(steer_options)
NUM_SPEED_ACTIONS = len(speed_options)
TOTAL_ACTIONS = NUM_STEER_ACTIONS * NUM_SPEED_ACTIONS

# --- Hyperparameters ---
learning_rate = 0.0002
gamma = 0.98
weight_decay = 1e-4
max_grad_norm = 0.5
entropy_coef = 0.01  # 탐험 유도


RACETRACK = 'Oschersleben'
# RACETRACK = 'map_easy3'

def get_today():
    now = time.localtime()
    return "%04d-%02d-%02d_%02d-%02d-%02d" % (
        now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

def get_now():
    now = time.localtime()
    return "%02d-%02d-%02d" % (
        now.tm_hour, now.tm_min, now.tm_sec)

def preprocess_lidar(ranges):
    return np.array(ranges)

# --- Policy Network ---
class PolicyNet(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256) # 1080 ->
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 128)
        self.action_head = nn.Linear(128, n_actions)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        logits = self.action_head(x)
        return self.softmax(logits)

# --- Value Network ---
class ValueNet(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 128)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.value_head(x)

# --- Action 선택 ---
def select_action(policy_net, state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy_net(state)
    m = torch.distributions.Categorical(probs)
    action = m.sample()
    return action.item(), m.log_prob(action), m.entropy()

# --- 리턴 계산 ---
def compute_returns(rewards, gamma):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

# --- 시각화 ---
def plot_durations(laptimes):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(laptimes, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Lap Time')
    plt.plot(durations_t.numpy())
    if len(durations_t) >= 10:
        means = durations_t.unfold(0, 10, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(9), means))
        plt.plot(means.numpy())
    plt.pause(0.001)
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

# --- 메인 학습 함수 ---
def main():
    # 설정
    today = get_today()
    work_dir = f"./{today}_{RACETRACK}"
    os.makedirs(work_dir, exist_ok=True)

    env = gym.make('f110_gym:f110-v0',
                   map=f"{os.path.abspath(os.path.dirname(__file__))}/maps/{RACETRACK}",
                   map_ext=".png", num_agents=1)

    # poses = np.array([[0., 0., np.radians(270)]])
    poses = np.array([[0., 0., np.radians(160)]])
    obs, _, _, _ = env.reset(poses=poses)
    s = preprocess_lidar(obs['scans'][0])
    obs_dim = s.shape[0]
    print(obs_dim)

    policy_net = PolicyNet(obs_dim, TOTAL_ACTIONS)
    value_net = ValueNet(obs_dim)

    policy_optimizer = optim.AdamW(policy_net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    value_optimizer = optim.AdamW(value_net.parameters(), lr=learning_rate * 5, weight_decay=weight_decay)

    # 하이퍼파라미터
    laptimes, fastlap = [], 10000.0
    lapfinish_count = 0
    print_interval = 10

    policy_net.train()

    for n_epi in range(10000):
        obs, _, done, _ = env.reset(poses=poses)
        s = preprocess_lidar(obs['scans'][0])
        log_probs, rewards, states_for_value, entropies = [], [], [], []
        laptime = 0.0

        while not done:
            # env.render(mode='human_fast')

            action_idx, log_prob, entropy = select_action(policy_net, s)
            steer = steer_options[action_idx // NUM_SPEED_ACTIONS]
            speed = speed_options[action_idx % NUM_SPEED_ACTIONS]
            obs, r, done, _ = env.step(np.array([[steer, speed]]))
            s_prime = preprocess_lidar(obs['scans'][0])

            log_probs.append(log_prob)
            entropies.append(entropy)
            rewards.append(r)
            states_for_value.append(s)
            s = s_prime
            laptime += r

        returns = torch.tensor(compute_returns(rewards, gamma), dtype=torch.float)
        states_tensor = torch.tensor(np.array(states_for_value), dtype=torch.float)
        values = value_net(states_tensor).squeeze(-1)

        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Critic 업데이트
        value_loss = F.mse_loss(values, returns)
        value_optimizer.zero_grad()
        value_loss.backward()
        clip_grad_norm_(value_net.parameters(), max_grad_norm)
        value_optimizer.step()

        policy_loss = [
            -log_prob * adv - entropy_coef * ent
            for log_prob, adv, ent in zip(log_probs, advantages, entropies)
        ]
        loss = torch.stack(policy_loss).sum()

        policy_optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(policy_net.parameters(), max_grad_norm)
        policy_optimizer.step()

        laptimes.append(laptime)
        plot_durations(laptimes)

        lap = round(obs['lap_times'][0], 3)
        if int(obs['lap_counts'][0]) == 2 and fastlap > lap:
            torch.save(policy_net.state_dict(), f"{work_dir}/fast-model{lap}_{n_epi}.pt")
            fastlap = lap
            lapfinish_count += 1

        if n_epi % print_interval == 0 and n_epi != 0:
            now_time = get_now()
            print(f"{now_time}|[EP {n_epi}] Avg Score: {laptime / print_interval:.1f}, FastLap: {fastlap:.2f}, finish_ratio: {lapfinish_count}, ValueLoss: {value_loss.item():.4f}")
            lapfinish_count = 0

    print("Training finished")
    env.close()    

# --- 평가 루프 ---
def eval(model_path, fastlap):
    env = gym.make('f110_gym:f110-v0',
                   map=f"{os.path.abspath(os.path.dirname(__file__))}/maps/{RACETRACK}",
                   map_ext=".png", num_agents=1)
    
    poses = np.array([[0., 0., np.radians(160)]])
    obs, _, done, _ = env.reset(poses=poses)
    obs_dim = preprocess_lidar(obs['scans'][0]).shape[0]

    policy_net = PolicyNet(obs_dim, TOTAL_ACTIONS)
    policy_net.load_state_dict(torch.load(model_path))  
    policy_net.eval()

    for episode in range(5):
        obs, _, done, _ = env.reset(poses=poses)
        s = preprocess_lidar(obs['scans'][0])
        laptime = 0.0

        # env.render()
        while not done:
            with torch.no_grad():
                probs = policy_net(torch.from_numpy(s).float().unsqueeze(0))
                action = torch.argmax(probs).item()

                # probs = policy_net(torch.from_numpy(s).float().unsqueeze(0))
                # m = torch.distributions.Categorical(probs)
                # action = m.sample().item()

            steer = steer_options[action // NUM_SPEED_ACTIONS]
            speed = speed_options[action % NUM_SPEED_ACTIONS]
            obs, r, done, _ = env.step(np.array([[steer, speed]]))
            s = preprocess_lidar(obs['scans'][0])
            laptime += r
            env.render(mode='human_fast')

        print(f"Episode {episode+1} lap time: {obs['lap_times'][0]:.3f}")
        if obs['lap_counts'][0] == 2:
            print('Finish') # 두바퀴 돌긴했는지 체크
            if fastlap[0] > obs['lap_times'][0]:
                fastlap[0] = obs['lap_times'][0]
                fastlap[1] = model_path
            
    env.close()

# --- 제일 빠른거 테스트 & 분석용 --- 못썼음...

if __name__ == "__main__":
    main()  

    
    # # #### 폴더 지정해서 평가 ####
    path = "./"
    fastlap = [10000, '']

    # # 여기에 각 폴더에서 가져오는 코드
    folders = [f.path for f in os.scandir(path)
               if f.is_dir() and f.name.startswith("2025")]
    
    if not folders:
        print("No matching folders found.")
    else:
        for folder in folders:
            fastlap = [10000, '']

            print(f"\n--- Evaluating folder: {folder} ---")
            model_files = glob.glob(os.path.join(folder, "*.pt"))
            if not model_files:
                print(f"No .pt files found in {folder}")
                continue

            for model_file in model_files:
                print("\n" + "="*50)
                print(f"Evaluating model: {os.path.basename(model_file)}")
                print("="*50)
                print("="*50 + "\n")
            

            print(f"In /{folder} | Fast Lap: {fastlap[0]}, model: {fastlap[1]}")