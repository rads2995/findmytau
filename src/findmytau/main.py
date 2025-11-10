import matplotlib.pyplot as plt
import numpy as np

from .sde import SDE

def main():
    sde: SDE = SDE()
    sde.simulate()
    print(sde.optimal_stopping())

    plt.figure(figsize=(10,6))
    for i in range(5):
        path = sde.X[:, i]
        plt.plot(np.linspace(0, sde.T, sde.M+1), path.get(), lw=1, alpha=0.7)
        
        # mark the stopping point with a red dot
        t_stop = sde.stopping_times[i].get()
        plt.scatter(t_stop * sde.dt, path[t_stop].get(), color="red", zorder=3)

    plt.axhline(1.0, color="black", ls="--", lw=1)  # strike boundary
    plt.title("Sample Trajectories with Optimal Stopping Decisions")
    plt.xlabel("Time")
    plt.ylabel("Underlying Price")
    plt.savefig("result.png")
