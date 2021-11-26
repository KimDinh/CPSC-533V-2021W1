import gym
import math
import random
from itertools import count
import torch
from eval_policy import eval_policy, device
from model import MyModel
from replay_buffer import ReplayBuffer


BATCH_SIZE = 256
GAMMA = 0.99
EPS_EXPLORATION = 0.2
TARGET_UPDATE = 10
NUM_EPISODES = 1000
TEST_INTERVAL = 25
LEARNING_RATE = 1e-3
RENDER_INTERVAL = 20
ENV_NAME = 'CartPole-v0'
PRINT_INTERVAL = 5

env = gym.make(ENV_NAME)
state_shape = len(env.reset())
n_actions = env.action_space.n

model = MyModel(state_shape, n_actions).to(device)
target = MyModel(state_shape, n_actions).to(device)
target.load_state_dict(model.state_dict())
target.eval()

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
memory = ReplayBuffer()

'''
def choose_action(state, test_mode=False):
    if test_mode or random.random() > EPS_EXPLORATION:
        return torch.argmax(model(torch.tensor([state])).detach(), dim=1).view(1,1)
    else:
        return torch.tensor(random.randint(0,1)).view(1, 1)

def optimize_model(state, action, next_state, reward, done):
    target_value = torch.tensor(reward).view(1)
    if not done:
        target_value += GAMMA * model(torch.tensor([next_state])).max(1)[0].detach()
    output = model(torch.tensor([state]))[0, action].view(1)
    loss = torch.nn.MSELoss()(output, target_value)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
'''

def choose_action(state, test_mode=False):
    if test_mode or random.random() > EPS_EXPLORATION:
        return model.select_action(torch.tensor([state]))
    else:
        return torch.tensor(random.randint(0,1)).view(1, 1)

def optimize_model(state, action, next_state, reward, done):
    target_value = reward + GAMMA * (1-done) * model(next_state).max(1)[0].detach()
    output = model(state).gather(1, action.type(torch.long)).squeeze(1)
    
    loss = torch.nn.MSELoss()(output, target_value)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train_reinforcement_learning(render=False):
    steps_done = 0
    best_score = -float("inf")

    for i_episode in range(1, NUM_EPISODES+1):
        episode_total_reward = 0
        state = env.reset()
        for t in count():
            action = choose_action(state).cpu().numpy()[0][0]
            next_state, reward, done, _ = env.step(action)
            steps_done += 1
            episode_total_reward += reward
            
            memory.push(state, action, next_state, reward, done)
            if steps_done >= BATCH_SIZE:
                optimize_model(*memory.sample(BATCH_SIZE))
            #optimize_model(state, action, next_state, reward, done)
            state = next_state

            if render:
                env.render(mode='human')

            if done:
                if i_episode % PRINT_INTERVAL == 0:
                    print('[Episode {:4d}/{}] [Steps {:4d}] [reward {:.1f}]'
                        .format(i_episode, NUM_EPISODES, t, episode_total_reward))
                break

        if i_episode % TARGET_UPDATE == 0:
            target.load_state_dict(model.state_dict())

        if i_episode % TEST_INTERVAL == 0:
            print('-'*10)
            score = eval_policy(policy=model, env=ENV_NAME, render=render)
            if score > best_score:
                best_score = score
                torch.save(model.state_dict(), "best_model_{}.pt".format(ENV_NAME))
                print('saving model.')
            print("[TEST Episode {}] [Average Reward {}]".format(i_episode, score))
            print('-'*10)


if __name__ == "__main__":
    train_reinforcement_learning()
