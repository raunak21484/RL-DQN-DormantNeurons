"""
@author: Viet Nguyen <nhviet1009@gmail.com>
@author: Raunak Thakur <raunak21484@iiitd.ac.in>
@author: Vidur Goel
@author: Nalish Jain
"""
import argparse
import os
import shutil
from random import random, randint, sample
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from src.deep_q_network import DeepQNetwork
from src.flappy_bird import FlappyBird
from src.utils import pre_processing
import torch.nn.init as init
class ActivationHook:
    def __init__(self):
        self.activations = []

    def hook_fn(self, module, input, output):
        # Store the activation values
        self.activations.append(output.detach().cpu().numpy())
def getOverlapCoefficient(layer1, layer2,tao):
    intersectMod = (np.sum(np.intersect1d(layer1 < tao,layer2< tao)))
    minMod = min(np.sum(layer1 < tao),np.sum(layer2 < tao))
    return intersectMod/minMod

def custom_init_neuron(layer, neuron_index):
    if isinstance(layer, nn.Linear):
        # Set outgoing weights to 0
        layer.weight.data[neuron_index, :] = 0.0
        init.uniform_(layer.weight.data[neuron_index, :], -0.1, 0.1)
        if layer.bias is not None:
            init.constant_(layer.bias.data[neuron_index], 0.0)

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Flappy Bird""")
    parser.add_argument("--image_size", type=int, default=84, help="The common width and height for all images")
    parser.add_argument("--batch_size", type=int, default=32, help="The number of images per batch")
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adam"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=0.1)
    parser.add_argument("--final_epsilon", type=float, default=1e-4)
    parser.add_argument("--num_iters", type=int, default=2000000)
    parser.add_argument("--replay_memory_size", type=int, default=50000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    model = DeepQNetwork()
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()
    game_state = FlappyBird()
    image, reward, terminal = game_state.next_frame(0)
    image = pre_processing(image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size, opt.image_size)
    image = torch.from_numpy(image)
    if torch.cuda.is_available():
        model.cuda()
        image = image.cuda()
    state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]

    replay_memory = []
    iter = 0
    time_pd=0
    tao_value=0.001
    layer1 = []
    overlap = []
    oldNP = np.array([1]*128)
    # BIGPD=0
    reducelr = 10
    while iter < opt.num_iters and iter < 25001:
        time_pd+=1
        hook1,hook2 = ActivationHook(),ActivationHook()
        target_layer1 = model.fc1
        target_layer2 = model.fc2
        hook_handle1 = target_layer1.register_forward_hook(hook1.hook_fn)
        hook_handle2 = target_layer2.register_forward_hook(hook2.hook_fn)
        prediction = model(state)[0]
        # BIGPD+=1
        if (iter % 2000 == 0):
            tao_value = tao_value/10
        
        if(time_pd== 500):
            time_pd=0
            activations1 = hook1.activations
            activations2 = hook2.activations

            # print(type(activations1))
            actnp1 = np.array(activations1)
            actnp1 = actnp1/np.mean(actnp1)
            actnp1 = actnp1.reshape(-1)
            actnp2 = np.array(activations2)
            actnp2 = actnp2/np.mean(actnp2)
            actnp2 = actnp2.reshape(-1)
            # current = np.concatenate((actnp1,actnp2))
            # print(actnp1)
            # print("np2:", actnp2)
            current = actnp1
            print(current)
            # print(current)
            overlap.append(getOverlapCoefficient(current,oldNP,tao_value))
            oldNP = current
            layer1.append((np.sum(current < tao_value))/128 * 100)
            print(layer1)
            # print(actnp1)
            indices = np.where(actnp1<tao_value)
            for i in indices:
                custom_init_neuron(model.fc1, i)


            # print(actnp2)
        hook_handle1.remove()
        hook_handle2.remove()
        # Exploration or exploitation
        epsilon = opt.final_epsilon + (
                (opt.num_iters - iter) * (opt.initial_epsilon - opt.final_epsilon) / opt.num_iters)
        u = random()
        random_action = u <= epsilon
        if random_action:
            # print("Exploring")
            action = randint(0, 1)
        else:
            # print("Exploiting")
            action = torch.argmax(prediction).item()

        next_image, reward, terminal = game_state.next_frame(action)
        # print("Reward: ",reward)
        next_image = pre_processing(next_image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size,
                                    opt.image_size)
        next_image = torch.from_numpy(next_image)
        if torch.cuda.is_available():
            next_image = next_image.cuda()
        next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]
        replay_memory.append([state, action, reward, next_state, terminal])
        if len(replay_memory) > opt.replay_memory_size:
            del replay_memory[0]
        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(*batch)

        state_batch = torch.cat(tuple(state for state in state_batch))
        action_batch = torch.from_numpy(
            np.array([[1, 0] if action == 0 else [0, 1] for action in action_batch], dtype=np.float32))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.cat(tuple(state for state in next_state_batch))

        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()
        current_prediction_batch = model(state_batch)
        next_prediction_batch = model(next_state_batch)

        y_batch = torch.cat(
            tuple(reward if terminal else reward + opt.gamma * torch.max(prediction) for reward, terminal, prediction in
                  zip(reward_batch, terminal_batch, next_prediction_batch)))

        q_value = torch.sum(current_prediction_batch * action_batch, dim=1)
        optimizer.zero_grad()
        # y_batch = y_batch.detach()
        loss = criterion(q_value, y_batch)
        loss.backward()
        optimizer.step()

        state = next_state
        iter += 1
        if iter % 500 == 0:
            print("Iteration: {}/{}, Action: {}, Loss: {}, Epsilon {}, Reward: {}, Q-value: {}".format(
                iter + 1,
                opt.num_iters,
                action,
                loss,
                epsilon, reward, torch.max(prediction)))
        writer.add_scalar('Train/Loss', loss, iter)
        writer.add_scalar('Train/Epsilon', epsilon, iter)
        writer.add_scalar('Train/Reward', reward, iter)
        writer.add_scalar('Train/Q-value', torch.max(prediction), iter)


        if (iter+1) % 1000000 == 0:
            torch.save(model, "{}/flappy_bird_{}".format(opt.saved_path, iter+1))
    
    time = np.arange(len(layer1))
    plt.figure(figsize=(10,12))
    plt.plot(time, layer1, marker='o', linestyle='-')
    plt.title('Plot of dormant neurons Against Time')
    plt.xlabel('Time')
    plt.ylabel('Data Values')
    plt.grid(True)
    
# Show the plot
    plt.show()

    plt.figure(figsize=(10,12))
    plt.plot(time,overlap,marker='o',linestyle= '-')
    plt.title('Plot of Overlaps Against Time')
    plt.xlabel('Time')
    plt.ylabel('Data Values')
    plt.grid(True)

# Show the plot
    plt.show()
    torch.save(model, "{}/flappy_bird".format(opt.saved_path))


if __name__ == "__main__":
    opt = get_args()
    train(opt)
