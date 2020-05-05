import gym
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import collections

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    # hidden layer sizes
    H1_SIZE = 32

    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=self.H1_SIZE)
        self.output = nn.Linear(in_features=self.H1_SIZE, out_features=output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.output(x)

class DQN:
    REPLAY_MEMORY_SIZE = 10000
    BATCH_SIZE = 10
    NUM_EPISODES = 2000
    GAMMA = 0.99
    MAX_STEPS = 200
    EPSILON = 0.5
    LEARNING_RATE = 1e-4
    MOMENTUM = 0.9
    TARGET_UPDATE_FREQ = 100
    

    def __init__(self, env):
        self.env = gym.make(env)
        self.input_dim = self.env.observation_space.shape[0]
        self.output_dim = self.env.action_space.n
        self.network = QNetwork(self.input_dim, self.output_dim).to(DEVICE)
        self.target_network = QNetwork(self.input_dim, self.output_dim).to(DEVICE)
        
        # init replay buffer
        self.replay_buffer = ReplayMemory(self.REPLAY_MEMORY_SIZE, self.BATCH_SIZE)
        
        # init target network
        self.target_network.load_state_dict(self.network.state_dict())     

        # loss function and optimizer
        self.criterion = nn.MSELoss()  
        self.optimizer = optim.SGD(self.network.parameters(), lr=self.LEARNING_RATE, momentum=self.MOMENTUM)

    def train(self):
        total_steps = 0
        for episode in range(self.NUM_EPISODES):
            steps =0
            s = self.env.reset()
            while True:
                # epsilon-greedy policy
                a = self.policy(s)
                
                # step in environment
                new_s, r, done, _ = self.env.step(a)
                
                # add experience to replay buffer
                self.replay_buffer.add(s, a, r, new_s, done)    

                # Q-learning step
                self.Q_learning()

                # update target network weights
                if total_steps % self.TARGET_UPDATE_FREQ == 0:
                    self.target_network.load_state_dict(self.network.state_dict())

                s = new_s
                steps += 1
                total_steps += 1

                # check for episode termination
                if done or steps > self.MAX_STEPS:
                    break

            print(episode, ':', steps)


    def Q_learning(self):
        # skip until replay buffer has sufficient number of samples
        if len(self.replay_buffer) < self.BATCH_SIZE:
            return        
                        
        # sample mini-batch
        batch = self.replay_buffer.sample()

        # approximate optimal targets
        y = batch.rewards

        # to include bootstrapping term for non-terminal states (no gradient calculation)
        with torch.no_grad():
            y -= self.target_network(batch.new_states).max(1, keepdim=True)[0]*(batch.flags - 1)
        
        # Q-network outputs for the actions taken in the corresponding states
        q = self.network(batch.states).gather(1, batch.actions)

        # calculate MSE loss
        loss = self.criterion(q, y).to(DEVICE)

        # optimization step
        self.optimizer.zero_grad()
        loss.backward()
        # clipping gradients as specified in the paper
        for w in self.network.parameters():
            w.grad.data.clamp_(-1, 1)
        self.optimizer.step()


    def policy(self, s):
        if np.random.rand() < self.EPSILON:
            action = np.random.randint(self.output_dim)
        else:
            s = torch.from_numpy(s).float().to(DEVICE)
            with torch.no_grad():
                action = (self.network(s).max(0)[1]).item()

        return action


# to handle experiences
class ReplayMemory:
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = []
        self.pos = 0
        self.experiences = collections.namedtuple('Experiences', ['states', 'actions', 'rewards', 'new_states', 'flags'])

    def add(self, s, a, r, new_s, done):
        e = [s, a, r, new_s, int(done)]
        # if replay buffer not full, append new experience
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(e)
        else:
            self.buffer[self.pos] = e
        self.pos += 1
        self.pos = self.pos % self.buffer_size

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        
        # experiences mini-batch
        states = torch.from_numpy(np.vstack([e[0] for e in batch])).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack([e[1] for e in batch])).long().to(DEVICE)
        rewards = torch.from_numpy(np.vstack([e[2] for e in batch])).float().to(DEVICE)
        new_states = torch.from_numpy(np.vstack([e[3] for e in batch])).float().to(DEVICE)
        flags = torch.from_numpy(np.vstack([e[4] for e in batch])).float().to(DEVICE)

        return self.experiences(states, actions, rewards, new_states, flags)

    def __len__(self):
        return len(self.buffer)


if __name__ == "__main__":
    dqn = DQN('CartPole-v0')
    dqn.train()