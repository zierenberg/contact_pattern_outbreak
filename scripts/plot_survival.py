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

    samples=int(1e5)

    run = True
    if run:
        fig, axs = plt.subplots(1,1)
        ax = axs
        # Survival as a function of generation
        survival = np.array(f["data/infectious_3.00_latent_2.00/survival_probability_generation/p=0.122137/N0=1/{:d}".format(samples)])
        ax.plot(survival, label=r"data with $T_\mathrm{{lat}}=2$", color=clrs.n_low)
        survival = np.array(f["data/infectious_3.00_latent_6.00/survival_probability_generation/p=0.122137/N0=1/{:d}".format(samples)])
        ax.plot(survival, label=r"data with $T_\mathrm{{lat}}=6$", color=clrs.n_high)
        survival = np.array(f["rand/infectious_3.00_latent_2.00/survival_probability_generation/p=0.122137/N0=1/{:d}".format(samples)])
        ax.plot(survival, label=r"data with $T_\mathrm{{lat}}=2$", color=clrs.n_psn)
        survival = np.array(f["rand/infectious_3.00_latent_6.00/survival_probability_generation/p=0.122137/N0=1/{:d}".format(samples)])
        ax.plot(survival, '-.', label=r"data with $T_\mathrm{{lat}}=6$", color=clrs.n_psn)

        ax.set_ylabel(r"survival probability")
        ax.set_xlabel(r"generation")

        ax.legend(loc='upper right')

        # has to be before set size
        fig.tight_layout()

        #plt.show()
        plt.savefig("/{}/results/survival_probability_generation.pdf".format(base_path))


    # Survival as a function of contact infection probability
    fig, axs = plt.subplots(1,2)
    ax = axs[0]
    survival = np.array(f["data/infectious_3.00_latent_2.00/survival_probability_p/N0=1/{:d}".format(samples)])
    ax.plot(survival[0],survival[2], label=r"data with $T_\mathrm{{lat}}=2$", color=clrs.n_low)
    survival = np.array(f["data/infectious_3.00_latent_6.00/survival_probability_p/N0=1/{:d}".format(samples)])
    ax.plot(survival[0],survival[2], label=r"data with $T_\mathrm{{lat}}=6$", color=clrs.n_high)
    survival = np.array(f["rand/infectious_3.00_latent_2.00/survival_probability_p/N0=1/{:d}".format(samples)])
    ax.plot(survival[0],survival[2], label=r"data with $T_\mathrm{{lat}}=2$", color=clrs.n_psn)
    survival = np.array(f["rand/infectious_3.00_latent_6.00/survival_probability_p/N0=1/{:d}".format(samples)])
    ax.plot(survival[0],survival[2], '-.', label=r"data with $T_\mathrm{{lat}}=6$", color=clrs.n_psn)

    ax.set_ylabel(r"survival probability")
    ax.set_xlabel(r"probability to infect contacts")

    ax.legend(loc='lower right')

    # has to be before set size
    fig.tight_layout()

    #plt.show()
    plt.savefig("/{}/results/survival_probability_infection_probability.pdf".format(base_path))

    # Survival as a function of effective R
    ax = axs[1]
    survival = np.array(f["data/infectious_3.00_latent_2.00/survival_probability_p/N0=1/{:d}".format(samples)])
    ax.plot(survival[1],survival[2], label=r"data with $T_\mathrm{{lat}}=2$", color=clrs.n_low)
    survival = np.array(f["data/infectious_3.00_latent_6.00/survival_probability_p/N0=1/{:d}".format(samples)])
    ax.plot(survival[1],survival[2], label=r"data with $T_\mathrm{{lat}}=6$", color=clrs.n_high)
    survival = np.array(f["rand/infectious_3.00_latent_2.00/survival_probability_p/N0=1/{:d}".format(samples)])
    ax.plot(survival[1],survival[2], label=r"data with $T_\mathrm{{lat}}=2$", color=clrs.n_psn)
    survival = np.array(f["rand/infectious_3.00_latent_6.00/survival_probability_p/N0=1/{:d}".format(samples)])
    ax.plot(survival[1],survival[2], '-.', label=r"data with $T_\mathrm{{lat}}=6$", color=clrs.n_psn)

    ax.set_ylabel(r"survival probability")
    ax.set_xlabel(r"effective $R_0$")

    ax.legend(loc='lower right')

    # has to be before set size
    fig.tight_layout()

    #plt.show()
    plt.savefig("/{}/results/survival_probability_infection_probability.pdf".format(base_path))

if __name__ == "__main__":
    main()
