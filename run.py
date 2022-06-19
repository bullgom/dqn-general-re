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

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    capacity = 10000
    batch_size = 128
    gamma = 0.999
    start = 0.9
    end = 0.05
    decay_steps = 300
    total_episodes = 10000
    steps_per_update = 100
    max_steps_per_episode = 5000
    mean_duration = 100
    record_interval = 50
    plot_interval = 150
    plot_path = "hi.png"
    img_size = 84
    frameskip = 4
    frames = 3
    swap_interval = 50

    env = ALE(
        "ALE/Breakout-v5", 
        frameskip,
        device,
        [
            prep.WrappedProcessing(tf.ToPILImage()),
            prep.ToTensor(),
            prep.Resize((img_size, img_size)),
            prep.WrappedProcessing(tf.Grayscale()),
            prep.AddBatchDim(),
            prep.MultiFrame(frames),
            prep.ToDevice(device),
        ]
    )

    input_size, output_size = env.size()

    buffer = ReplayBuffer(capacity, device, transition=Transition(
        s_now=torch.empty((0, *input_size[1:])),
        a=torch.empty((0, 1), dtype=torch.int),
        r=torch.empty((0,1)),
        s_next=torch.empty((0, *input_size[1:])),
        done=torch.empty((0,1), dtype=torch.bool)
    ))

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

                
    
