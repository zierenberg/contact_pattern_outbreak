import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, LogLocator
import os

base_path = os.path.join(*os.path.normpath(__file__).split(os.path.sep)[:-2])


from addict import Dict
clrs = Dict()
clrs.n_high = "#C31B2B"  # "#F83546"
clrs.n_low = "#5295C8"
clrs.n_psn = "#E7E7B6"

def main():
    filename="/{}/out/branching_process_Copenhagen_filtered_15min.h5".format(base_path)
    f = h5py.File(filename,'r')

    fig, axs = plt.subplots(1,1)
    ax = axs

    samples=int(1e5)
    #for name in ["data","rand"]:
    #    for lat in [2,6]:
    survival = np.array(f["data/infectious_3.00_latent_2.00/survival_probability/N0=1/{:d}".format(samples)])
    ax.plot(survival, label=r"data with $T_\mathrm{{lat}}=2$", color=clrs.n_low)
    survival = np.array(f["data/infectious_3.00_latent_6.00/survival_probability/N0=1/{:d}".format(samples)])
    ax.plot(survival, label=r"data with $T_\mathrm{{lat}}=6$", color=clrs.n_high)
    survival = np.array(f["rand/infectious_3.00_latent_2.00/survival_probability/N0=1/{:d}".format(samples)])
    ax.plot(survival, label=r"data with $T_\mathrm{{lat}}=2$", color=clrs.n_psn)
    survival = np.array(f["rand/infectious_3.00_latent_6.00/survival_probability/N0=1/{:d}".format(samples)])
    ax.plot(survival, '-.', label=r"data with $T_\mathrm{{lat}}=6$", color=clrs.n_psn)

    ax.set_ylabel(r"survival probability")
    ax.set_xlabel(r"generation")

    ax.legend(loc='upper right')

    # has to be before set size
    fig.tight_layout()

    #plt.show()
    plt.savefig("/{}/results/survival_probability_generation.pdf".format(base_path))


if __name__ == "__main__":
    main()
