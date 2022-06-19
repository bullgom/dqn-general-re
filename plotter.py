import matplotlib.pyplot as plt

class Plotter:

    def __init__(self):
        fig, ax = plt.subplots(2)
        self.figure = fig
        self.rpe_axes = ax[0]
        self.loss_axes = ax[1]
        self.rpe_axes.set_xlabel("Episodes")
        self.rpe_axes.set_ylabel("Rewards")
        
        self.loss_axes.set_xlabel("Steps")
        self.loss_axes.set_ylabel("Loss")
        
        plt.tight_layout()
        plt.ion()
    
    def plot(
        self, 
        rewards: list[float], 
        means: list[float], 
        losses: list[float],
        best: float,
    ) -> None:
        self.rpe_axes.clear()
        self.rpe_axes.set_title("Reward Per Episode")
        self.rpe_axes.plot(rewards, color="black", linewidth=1)
        self.rpe_axes.plot(means, color="red",
                           linestyle="--", linewidth=1.5)
        self.rpe_axes.axhline(y=best, label="best",color="red")
        #self.rpe_axes.axhline(y=last, label="last", color="yellow")
        self.rpe_axes.grid()
        
        self.loss_axes.clear()
        self.loss_axes.set_title("Loss Per Episode")
        self.loss_axes.plot(losses, color="black",linewidth=1)
        self.loss_axes.grid()
    
    def save(self, path: str) -> None:
        plt.savefig(path)
    
    def show(self) -> None:
        plt.pause(0.005)



