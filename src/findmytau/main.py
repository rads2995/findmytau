import matplotlib.pyplot as plt
import numpy as np

from .sde import SDE

def main():
    
    print("Pricing of American Style Put Options:")

    prices: list[float] = [36.0, 38.0, 40.0, 42.0, 44.0]

    for price in prices:
        sde: SDE = SDE(
            N = 100000,
            M = 50,
            mu = 0.06,
            sigma = 0.2,
            x0 = price,
            strike = 40.0,
            T = 1.0
        )
        sde.simulate()
        sde.find_optimal_stopping()
        print(
            "For S = {:.2f}, sigma = {:.2f}, and T = {:.2f}, the optimal stopping value is {:.12f}".format(
                sde.x0,
                sde.sigma,
                sde.T,
                sde.optimal_stopping
            )
        )

        plt.figure(figsize=(10,6))
        for i in range(10):
            path = sde.X[:, i]
            plt.plot(np.linspace(0, sde.T, sde.M+1), path.get(), lw=1, alpha=0.7)
            
            # mark the stopping point with a red dot
            t_stop = sde.stopping_times[i].get()
            plt.scatter(t_stop * sde.dt, path[t_stop].get(), color="red", zorder=3)

        plt.axhline(sde.strike, color="black", ls="--", lw=1)  # strike boundary
        plt.title("Sample Trajectories with Optimal Stopping Decisions")
        plt.xlabel("Time")
        plt.ylabel("Underlying Price")
        plt.savefig("quantile_stock_price_" + str(price) + ".png")
