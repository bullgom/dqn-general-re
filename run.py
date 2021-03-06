from agent import Agent
from strategy import EpsilonGreedy
from environment import ALE
from trainer import Trainer, OffPolicyTrainer
from replay_buffer import ReplayBuffer, Transition
from architecture import Network
import torch
import torchvision.transforms as tf
import preprocessing as prep
from recorder import Recorder
from plotter import Plotter

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu = torch.device("cpu")

    capacity = 100000000
    batch_size = 64
    gamma = 0.999
    start = 0.9
    end = 0.05
    decay_steps = 100000
    total_episodes = 10000
    max_steps_per_episode = 5000
    mean_duration = 100
    record_interval = 50
    plot_interval = 150
    plot_path = "hi.png"
    img_size = 84
    frameskip = 4
    frames = 3

    steps_per_update = 20
    swap_interval = 5 * steps_per_update

    env = ALE(
        "ALE/Breakout-v5", 
        frameskip,
        device,
        [
            prep.WrappedProcessing(tf.ToPILImage()),
            prep.ToTensor(),
            prep.Resize((img_size, img_size)),
            prep.Grayscale(),
            prep.AddBatchDim(),
            prep.MultiFrame(frames)
        ]
    )

    input_size, output_size = env.size()

    buffer = ReplayBuffer(capacity, device)

    network = Network(input_size, output_size, device)
    strategy = EpsilonGreedy(start, end, decay_steps)
    agent = Agent(network, strategy)
    optimizer = torch.optim.Adam(network.parameters())
    trainer = OffPolicyTrainer(
        swap_interval,
        buffer, 
        network, 
        optimizer, 
        batch_size, 
        gamma,
        steps_per_update=steps_per_update
    )
    recorder = Recorder(mean_duration, record_interval)
    plotter = Plotter()
    
    losses = []

    agent.reset()
    trainer.reset()
    recorder.on_game_reset()

    steps = 0
    for episode in range(total_episodes):
        state = env.reset()

        for i_episode in range(max_steps_per_episode):
            steps +=1

            action = agent.step(state)
            
            next_state, r, done = env.step(action)
            buffer.push(Transition(state, action, r, next_state, done))
            state = next_state
            loss = trainer.step()

            recorder.step(r.item(), loss.item())

            if steps % plot_interval == 0:
                plotter.plot(*recorder.data())
                plotter.save(plot_path)

            if done.item():
                break
        
        recorder.ep_duration.append(i_episode)
                
    
