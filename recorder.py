from base import Base
import matplotlib.pyplot as plt


class Recorder():

    def __init__(self, mean_duration: int) -> None:
        super().__init__()

        self.mean_duration = mean_duration

    def on_game_reset(self):
        self.rewards = []
        self.means = []
        self.losses = []
        self.best = -float('inf')
        self.reset_accumulated()

    def reset_accumulated(self) -> None:
        self.accum_rewards = []
        self.accum_means = []
        self.accum_losses = []

    def last_mean(self, r: list[float] = None) -> float:
        if not r:
            r = self.rewards
        return sum(r[-self.mean_duration:])/len(r[-self.mean_duration:])
    
    def accumulate(self, reward: float, loss: float) -> None:
        
        self.accum_rewards.append(reward)
        self.accum_losses.append(loss)

    def aggregate(self) -> None:
        
        self.rewards.append(sum(self.accum_rewards))
        self.losses.append(sum(self.accum_losses))
        self.means.append(self.last_mean())

        if self.rewards[-1] > self.best:
            self.best = self.rewards[-1]

    def data(self) -> tuple[list[float], list[float], list[float], float]:
        return self.rewards, self.means, self.losses, self.best
