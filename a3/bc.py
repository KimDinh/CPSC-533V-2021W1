import gym
import torch
import numpy as np
from eval_policy import eval_policy, device
from model import MyModel
from dataset import Dataset

BATCH_SIZE = 64
TOTAL_EPOCHS = 30
LEARNING_RATE = 10e-4
PRINT_INTERVAL = 500
TEST_INTERVAL = 2

ENV_NAME = 'CartPole-v0'

dataset = Dataset(data_path="./{}_dataset.pkl".format(ENV_NAME))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=4)
'''
print("Dataset size: {}".format(len(dataset)))
print("The dimension of a state: {}".format(np.shape(dataset.data[0][0])))
print("Range of dimension 0: [{}, {}]".format(np.min([d[0][0] for d in dataset.data]), np.max([d[0][0] for d in dataset.data])))
print("Range of dimension 1: [{}, {}]".format(np.min([d[0][1] for d in dataset.data]), np.max([d[0][1] for d in dataset.data])))
print("Range of dimension 2: [{}, {}]".format(np.min([d[0][2] for d in dataset.data]), np.max([d[0][2] for d in dataset.data])))
print("Range of dimension 3: [{}, {}]".format(np.min([d[0][3] for d in dataset.data]), np.max([d[0][3] for d in dataset.data])))
'''

env = gym.make(ENV_NAME)

# TODO INITIALIZE YOUR MODEL HERE
model = MyModel(4, 2)

def train_behavioral_cloning():
    
    # TODO CHOOSE A OPTIMIZER AND A LOSS FUNCTION FOR TRAINING YOUR NETWORK
    optimizer = torch.torch.optim.Adam(model.parameters())
    loss_function = torch.nn.CrossEntropyLoss()

    gradient_steps = 0

    for epoch in range(1, TOTAL_EPOCHS + 1):
        for iteration, data in enumerate(dataloader):
            data = {k: v.to(device) for k, v in data.items()}

            output = model(data['state'])
            loss = loss_function(output, data["action"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if gradient_steps % PRINT_INTERVAL == 0:
                print('[epoch {:4d}/{}] [iter {:7d}] [loss {:.5f}]'
                    .format(epoch, TOTAL_EPOCHS, gradient_steps, loss.item()))
            
            gradient_steps += 1

        if epoch % TEST_INTERVAL == 0:
            score = eval_policy(policy=model, env=ENV_NAME)
            print('[Test on environment] [epoch {}/{}] [score {:.2f}]'
                .format(epoch, TOTAL_EPOCHS, score))

    model_name = "behavioral_cloning_{}.pt".format(ENV_NAME)
    print('Saving model as {}'.format(model_name))
    torch.save(model.state_dict(), model_name)


if __name__ == "__main__":
    train_behavioral_cloning()
