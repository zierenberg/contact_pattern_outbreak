# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2021-02-09 18:58:52
# @Last Modified: 2022-09-20 17:35:53
# ------------------------------------------------------------------------------ #
# plotting for all figures of the manuscript.
# requires julia to run the analysis beforehand.
#
# # Example
# ```
#   cd resonance_contact_disease
#   python -i ./plotting/plot_helper.py
# ```
# ```python
# # adjust what to draw in every panel, these are global settings
# show_title = True
# show_xlabel = True
# show_ylabel = True
# show_legend = True
# show_legend_in_extra_panel = False
# use_compact_size = False  # this recreates the small panel size of the manuscript

# # load the main file
# h5f = bnb.hi5.recursive_load(
#    file_path_shorthand("data"), dtype=bdict, keepdim=True
# )

# # create figures, and, at any point, save whatever is currently open.
# figure_1(h5f)
# figure_2(h5f)
# figure_3(h5f)
# figure_4(h5f)
# figure_5(h5f)  # this guy needs the extra files in './out_mf/'
# save_all_figures("./figs/", fmt="pdf", dpi=300)
# ```
# ------------------------------------------------------------------------------ #

# select things to draw for every panel for every panel
show_title = False
show_xlabel = False
show_ylabel = False
show_legend = False
show_legend_in_extra_panel = False
use_compact_size = True  # this recreates the small panel size of the manuscript
figures_only_to_disk = True
debug = True


# fmt: off
import os
from shutil import which
import sys
import glob
import argparse
import inspect
import re
import logging
import numpy as np
import pandas as pd
import xarray as xr
import scipy.stats
from scipy.special import gamma, comb
from scipy.optimize import curve_fit
from itertools import product
from sklearn.metrics import average_precision_score
from tqdm import tqdm


# some settings that will be applied to all figures
import matplotlib
# matplotlib.rcParams['font.sans-serif'] = "Arial"
# matplotlib.rcParams['font.family'] = "sans-serif"
# matplotlib.rcParams['axes.linewidth'] = 0.3
matplotlib.rcParams["axes.labelcolor"] = "black"
matplotlib.rcParams["axes.edgecolor"] = "black"
matplotlib.rcParams["xtick.color"] = "black"
matplotlib.rcParams["ytick.color"] = "black"
matplotlib.rcParams["xtick.labelsize"]= 8
matplotlib.rcParams["ytick.labelsize"]= 8
matplotlib.rcParams['xtick.major.pad']= 2
matplotlib.rcParams['ytick.major.pad']= 2
matplotlib.rcParams["axes.titlesize"]= 10
matplotlib.rcParams["axes.labelsize"]= 8
matplotlib.rcParams["legend.fontsize"] = 6
matplotlib.rcParams["legend.facecolor"] = "#D4D4D4"
matplotlib.rcParams["legend.framealpha"] = 0.8
matplotlib.rcParams["legend.frameon"] = True
matplotlib.rcParams["legend.edgecolor"] = (1,1,1,0)
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["figure.figsize"] = [3.4, 2.7]  # APS single column
# matplotlib.rcParams["figure.figsize"] = [2.4, 1.9]  # third of a ~16cm figure
matplotlib.rcParams['figure.dpi'] = 300

# only useful when using the _set_size() helper to create small panels, to avoid
# clipping of axis labels and ticks
# matplotlib.rcParams["figure.subplot.left"] =   0.3
# matplotlib.rcParams["figure.subplot.right"] =  0.7
# matplotlib.rcParams["figure.subplot.bottom"] = 0.3
# matplotlib.rcParams["figure.subplot.top"] =    0.7
# matplotlib.rcParams["figure.subplot.wspace"] = 0.66
# matplotlib.rcParams["figure.subplot.hspace"] = 0.66


matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler("color",
    ["#233954","#ea5e48","#1e7d72","#f49546","#e8bf58",  # dark
     "#5886be","#f3a093","#53d8c9","#f2da9c","#f9c192",  # light
    ])  # qualitative, somewhat color-blind friendly, in mpl words 'tab5'
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator, LogLocator, NullFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s",
    datefmt="%y-%m-%d %H:%M",
)
log = logging.getLogger(__name__)
log.setLevel("INFO")

import seaborn as sns
import palettable as pt

import functools
from benedict import benedict
# we want "/" as keypath separator instead of ".", set it as default argument
bdict = functools.partial(benedict, keypath_separator="/")
# now we can call bdict() to get a dict that can be accessed as foo["some/level/bar"]

import bitsandbobs as bnb

# enable code formatting with black
# fmt: on

figure_path = "./figs/v3"

clrs = dict(
    n_high="#C31B2B",
    n_low="#5295C8",
    n_psn="#CBCB89",
    psn_unweighted="#E3B63F",
    psn="#C1A61F",
    wbl_unweighted="#ea5e48",
    wbl="#C54532",
    tlrd="#E07D13",
    data="#233954",
    data_rand="#CBCB89",
    data_rand_all="#888888",
    #
    #
    # viral_load = "#1e7d72",
    # activity = "#6EB517",
    # weekday = "#143163",
    # weekend = "#2E72A8",
    # medium = "#46718C",
    # weekday_psn = "#E3B63F",
    # weekend_psn = "#F1CD79",
)


def file_path_shorthand(which):
    path = "./out/"
    # add all the base paths
    if which.startswith("data"):
        path += "data"

    elif which.startswith("psn_unweighted"):
        path += "surrogate_inhomogeneous_poisson"

    elif which.startswith("psn"):
        path += "surrogate_inhomogeneous_poisson_weighted"

    elif which.startswith("wbl_unweighted"):
        path += "surrogate_weibull"

    elif which.startswith("wbl"):
        path += "surrogate_weibull_weighted"

    elif which.startswith("tlrd"):
        path += "surrogate_tailored"
    else:
        log.warning(f"Unknown shorthand: {which}")
        return None

    # for randomized, we appended a string to the filename
    try:
        if which[-5:] == "_rand":
            path += "_randomized_per_train"
        elif which[-9:] == "_rand_all":
            path += "_randomized_all"
    except:
        pass

    # and we have a suffix for filtering
    path += "_Copenhagen_filtered_15min.h5"

    # checks
    if not os.path.exists(path):
        log.error(f"{which} -> {path} does not exist")

    return path


# default marker size
ms_default = 2


def default_plot_kwargs(which, for_errorbars=False):

    kwargs = dict()
    kwargs["label"] = which
    kwargs["lw"] = 1.0
    kwargs["zorder"] = 1

    try:
        kwargs["color"] = clrs[which]
    except:
        pass

    # if which == "data":
    # per default data in the back?
    # set by adding lines in the right order, we dont specify zorders here.
    # kwargs["zorder"] = 1

    if "rand" in which:
        kwargs["marker"] = "o"
        # kwargs["markerfacecolor"] = kwargs["color"]
        kwargs["markeredgewidth"] = 0.0
        kwargs["markersize"] = 1.5
        # kwargs["lw"] = 1.0

    if for_errorbars:
        kwargs["fmt"] = "o"
        kwargs["markersize"] = 1.5
        kwargs["markerfacecolor"] = kwargs["color"]
        kwargs["ecolor"] = kwargs["color"]
        # kwargs["alpha"]=1
        kwargs["elinewidth"] = 1
        kwargs["capsize"] = 0

    return kwargs


# cm to inch
cm = 0.3937


def main_manuscript():
    figure_1()
    figure_2()
    figure_3()
    figure_4()


# ------------------------------------------------------------------------------ #
# macro functions to create figures
# ------------------------------------------------------------------------------ #


def figure_1():
    log.info("Figure 1")

    # ax = plot_etrain_rasters()
    # save_ax(ax, f"{figure_path}/f1_trains.pdf")

    kwargs = default_plot_kwargs("data_rand", for_errorbars=False)
    kwargs["zorder"] = 2
    kwargs["lw"] = 0

    ax = plot_dist_encounters_per_train(
        which=["data", "data_rand"], plot_kwargs={"data_rand": kwargs}
    )
    save_ax(ax, f"{figure_path}/f1_dist_per_train.pdf")

    ax = plot_dist_encounters_per_day(
        which=["data", "data_rand"], plot_kwargs={"data_rand": kwargs}
    )
    save_ax(ax, f"{figure_path}/f1_dist_per_day.pdf")


def figure_2():
    log.info("Figure on encounter distributions and survival probability")

    # plot_etrain_raster_example(h5f)
    # ax = plot_disease_dist_infectious_encounters(h5f, ax=None, k="k_inf", periods="slow")
    # ax.set_xlim(-5, 150)
    # ax.set_ylim(1e-4, 1e-1)

    ax = None
    ax = plot_disease_dist_infectious_encounters(
        ax=ax, k="k_inf", periods="slow", which=["data"]
    )
    # for randomized, we only plot one period. they overlap.
    ax = plot_disease_dist_infectious_encounters(
        ax=ax,
        k="k_inf",
        periods="2_3",
        which=["data_rand"],
    )
    ax = plot_disease_dist_infectious_encounters(
        ax=ax,
        k="k_inf",
        periods="2_3",
        which=["data_rand_all"],
    )

    ax.get_figure().tight_layout()
    ax.set_ylim(1e-4, 1.4e-1)
    ax.set_xlim(-5, 130)
    bnb.plt.set_size(ax, w=5.2, h=2.1, l=1.5, b=1.0, t=0.5, r=0.1)
    save_ax(ax, f"{figure_path}/f2_distribution_of_infectious_encounters.pdf")

    # different file conventions, this is hardcoded in the function, no `which` arg
    ax = plot_extinction_probability()
    bnb.plt.set_size(ax, w=5.2, h=2.1, l=1.0, b=0.6, t=0.2, r=0.1)
    save_ax(ax, f"{figure_path}/f2_extinction_probability.pdf")

    ax = None
    ax = plot_disease_dist_secondary_infections(
        ax=ax, R=3.0, which=["data_rand", "data_rand_all"], periods="2_3"
    )
    ax = plot_disease_dist_secondary_infections(
        ax=ax, R=3.0, which=["data"], periods="2_3"
    )
    bnb.plt.set_size(ax, w=2.1, h=1.6, l=1.0, b=0.6, t=0.2, r=0.1)
    sns.despine(ax=ax, trim=False, offset=3)
    save_ax(ax, f"{figure_path}/f2_offspring_distribution.pdf")

    # dispersion vs latent period
    ax = None
    ax = plot_dispersion_cutplane(
        ax=ax,
        coords=dict(R0=3, infectious=3),
        par="disp",
        which="data_rand_all",
    )
    ax = plot_dispersion_cutplane(
        ax=ax,
        coords=dict(R0=3, infectious=3),
        par="disp",
        which="data_rand",
    )
    ax = plot_dispersion_cutplane(
        ax=ax,
        coords=dict(R0=3, infectious=3),
        par="disp",
        which="data",
    )
    bnb.plt.set_size(ax, w=1.6, h=1.0, l=1.0, b=0.6, t=0.2, r=0.1)
    sns.despine(ax=ax, trim=False, offset=3)
    save_ax(ax, f"{figure_path}/f2_dispersion_vs_latent.pdf")

    # dispersion vs R0
    ax = None
    ax = plot_dispersion_cutplane(
        ax=ax,
        coords=dict(latent=2, infectious=3),
        par="disp",
        which="data_rand",
    )
    ax = plot_dispersion_cutplane(
        ax=ax,
        coords=dict(latent=2, infectious=3),
        par="disp",
        which="data_rand_all",
    )
    ax = plot_dispersion_cutplane(
        ax=ax,
        coords=dict(latent=2, infectious=3),
        par="disp",
        which="data",
        color=clrs["n_low"],
    )
    ax = plot_dispersion_cutplane(
        ax=ax,
        coords=dict(latent=6, infectious=3),
        par="disp",
        which="data",
        color=clrs["n_high"],
    )
    bnb.plt.set_size(ax, w=.4, h=1.0, l=0.3, b=0.6, t=0.2, r=0.1)
    # sns.despine(ax=ax, trim=False, bottom=True, left=True, offset=3)
    sns.despine(ax=ax, trim=False, offset=3)
    # ax.xaxis.set_visible(False)
    # ax.yaxis.set_visible(False)
    save_ax(ax, f"{figure_path}/f2_dispersion_vs_R.pdf")


def figure_3():
    log.info("Figure on conditional enc rate and pace of spread R_0")

    rand_kwargs = default_plot_kwargs("data_rand")
    rand_kwargs["marker"] = None
    ax = plot_conditional_rate(
        which=["data_rand", "data"],
        shaded_regions=True,
        plot_kwargs={"data_rand": rand_kwargs},
    )
    save_ax(ax, f"{figure_path}/f3_conditional_encounter_rate.pdf")

    ax = plot_disease_mean_number_of_infectious_encounter_cutplane(
        ax=None,
        t_inf=3,
        which=["data_rand", "data"],
        plot_kwargs={
            "data": dict(
                color="black", linestyle=(0, (0.01, 2)), dash_capstyle="round", zorder=1
            ),
            "data_rand": rand_kwargs,
        },
    )
    save_ax(ax, f"{figure_path}/f3_cutplane.pdf")

    ax = plot_disease_mean_number_of_infectious_encounter_2d(
        which="data", relative_to="data_rand", control_plot=False
    )
    save_ax(ax, f"{figure_path}/f3_2d.pdf")

    # hard coded file path, no `which` arg
    ax = plot_case_numbers()
    _set_size(ax, 3.3 * cm, 2.5 * cm)
    save_ax(ax, f"{figure_path}/f3_case_numbers.pdf")

    # hard coded file path, no `which` arg
    ax = plot_growth_rate()
    _set_size(ax, 3.3 * cm, 2.5 * cm)
    save_ax(ax, f"{figure_path}/f3_growth_rate.pdf")


def figure_4(create_distirbution_insets=False):
    log.info("Figure comparing models")

    with plt.rc_context(
        {
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
        }
    ):
        for process in ["psn", "psn_unweighted", "wbl", "wbl_unweighted", "tlrd"]:

            # set zorder by adding `which` in the right order

            kwargs = default_plot_kwargs(process, for_errorbars=False)
            kwargs["lw"] = 0.5
            ax = plot_conditional_rate(
                which=["data", process], plot_kwargs={process: kwargs}
            )
            _set_size(ax, 2.5 * cm, 1.8 * cm)
            save_ax(ax, f"{figure_path}/f4_conditional_rate_{process}.pdf")

            ax = plot_etrain_rate(which=["data", process])
            _set_size(ax, 2.8 * cm, 1.8 * cm)
            save_ax(ax, f"{figure_path}/f4_etrain_rate_{process}.pdf")

            kwargs = default_plot_kwargs(process, for_errorbars=False)
            kwargs["lw"] = 1.5
            ax = plot_dist_inter_encounter_interval(
                which=[process, "data"], plot_kwargs={process: kwargs}
            )
            _set_size(ax, 2.3 * cm, 1.8 * cm)
            save_ax(ax, f"{figure_path}/f4_iei_{process}.pdf")

            kwargs = default_plot_kwargs(process, for_errorbars=False)
            kwargs["lw"] = 1.5
            kwargs["zorder"] = 2
            ax = plot_dist_encounters_per_train(
                which=["data", process], plot_kwargs={process: kwargs}
            )
            _set_size(ax, 2.3 * cm, 1.8 * cm)
            save_ax(ax, f"{figure_path}/f4_encouners_per_train_{process}.pdf")

            # 2d plots
            ax = plot_disease_mean_number_of_infectious_encounter_2d(
                which=process, relative_to=f"{process}_rand", control_plot=True
            )
            _set_size(ax, 2.0 * cm, 2.0 * cm)
            save_ax(ax, f"{figure_path}/f4_2d_{process}.pdf")

            # for distributions, show different latent periods
            for period in ["2_3", "6_3"]:

                # in there, plot styles are hardcoded
                ax = compare_disease_dist_encounters_generative(
                    which=["data", process], periods=[period]
                )
                ax.set_xlim(-5, 150)
                ax.set_ylim(1e-4, 1e-1)
                _set_size(ax, 2.5 * cm, 1 * cm)

                if create_distirbution_insets:
                    f = functools.partial(
                        compare_disease_dist_encounters_generative,
                        which=["data", process],
                        periods=[period],
                        set_size=False,
                        annotate=False,
                    )
                    axins = create_inset(
                        plot_func=f,
                        ax=ax,
                        width="40%",
                        height="45%",
                        xlim=(-1.0, 10),
                        ylim=(1.4e-3, 1.4e-1),
                        borderpad=0.0,
                        inset_loc=1,
                        con_loc1=1,
                        con_loc2=4,
                        mark_zorder=-1,
                    )
                    _detick([axins.xaxis, axins.yaxis])
                    if period == "2_3":
                        _detick(ax.xaxis, keep_ticks=True)
                        ax.set_xlabel("")

                save_ax(ax, f"{figure_path}/f4_dist_encounters_{process}_{period}.pdf")


def create_inset(
    ax,
    width,
    height,
    xlim,
    ylim,
    plot_func,
    borderpad=0.25,
    inset_loc=3,
    con_loc1=2,
    con_loc2=4,
    mark_zorder=5,
):
    gray = _alpha_to_solid_on_bg("gray", 0.5)
    with plt.rc_context(
        {
            "axes.edgecolor": gray,
            "xtick.color": gray,
            "ytick.color": gray,
            "axes.spines.right": True,
            "axes.spines.top": True,
        }
    ):
        axins = inset_axes(
            ax, width=width, height=height, loc=inset_loc, borderpad=borderpad
        )
        plot_func(ax=axins)

    axins.set_xlim(*xlim)
    axins.set_ylim(*ylim)
    mark_inset(
        ax,
        axins,
        lw=0.75,
        loc1=con_loc1,
        loc2=con_loc2,
        fc="none",
        ec=gray,
        zorder=mark_zorder,
        clip_on=False,
    )

    return axins


# figure 3 in v1
def figure_sm_features_explained(h5f):
    log.info("Figure feautres explained")
    for which in [
        "data",
        "poisson_homogeneous",
        "poisson_homogeneous_weighted_trains",
        "poisson_inhomogeneous",
        "poisson_inhomogeneous_weighted_trains",
        "weibul_renewal_process",
        "weibul_renewal_process_weighted_trains",
    ]:
        plot_disease_mean_number_of_infectious_encounter_2d(
            h5f, which=which, how="relative", control_plot=True
        )

    which = ["data", "poisson_inhomogeneous", "poisson_inhomogeneous_weighted_trains"]
    plot_conditional_rate(h5f, which=which, control_plot=True)

    which = ["data", "weibul_renewal_process", "weibul_renewal_process_weighted_trains"]
    plot_conditional_rate(h5f, which=which, control_plot=True)


# figure 4 in v1
def figure_sm_dispersion(h5f):
    log.info("Figure dispersion")
    plot_disease_viral_load_examples()
    plot_gamma_distribution()
    plot_disease_dist_infectious_encounters(h5f, k="k_inf", periods="slow")
    plot_disease_dist_infectious_encounters(h5f, k="k_10.0", periods="slow")
    plot_dispersion_scan_k(h5f, periods="slow")


def figure_sm_external(h5f):
    log.info("Figure SM external")
    plot_controls_distributions_infectious_encounters(
        h5f, which="gamma_6_3", k_sel="k_10.0", inset=False
    )
    plot_controls_distributions_infectious_encounters(
        h5f, which="gamma_2_3", k_sel="k_10.0", inset=False
    )
    plot_controls_means_infectious_encounters(
        h5f, which_list=["gamma_6_3", "gamma_2_3"], k_sel="k_10.0"
    )


def figure_sm_overview(h5f):
    log.info("Figure SM overview")
    plot_etrain_rate(h5f)
    plot_dist_inter_encounter_interval(h5f)
    plot_disease_mean_number_of_infectious_encounter_2d(
        h5f, which="data", how="relative", control_plot=False
    )
    plot_conditional_rate(h5f, which=["data"], control_plot=True)
    plot_disease_dist_infectious_encounters(h5f, k="k_inf", periods="slow")
    plot_disease_dist_infectious_encounters(h5f, k="k_10.0", periods="slow")
    plot_disease_dist_infectious_encounters(h5f, k="k_inf", periods="fast")
    plot_disease_dist_infectious_encounters(h5f, k="k_10.0", periods="fast")
    plot_dispersion_scan_k(h5f, periods="slow")
    plot_dispersion_scan_k(h5f, periods="fast")


def figure_sm_dispersion(h5f):
    log.info("Figure SM dispersion")
    plot_disease_dist_infectious_encounters(h5f, k="k_1.0", periods="slow")
    plot_disease_dist_infectious_encounters(h5f, k="k_10.0", periods="slow")
    plot_disease_dist_infectious_encounters(h5f, k="k_inf", periods="slow")
    plot_disease_dist_infectious_encounters(h5f, k="k_1.0", periods="fast")
    plot_disease_dist_infectious_encounters(h5f, k="k_10.0", periods="fast")
    plot_disease_dist_infectious_encounters(h5f, k="k_inf", periods="fast")
    plot_dispersion_scan_k(h5f, periods="slow")
    plot_dispersion_scan_k(h5f, periods="fast")


def figure_sm_rssi_duration(how="absolute"):
    """
    Controls for different distance and contact duration
    creates plots for 2d potentially inf. encounters and conditional encounter rate
    using hardcoded paths
    """

    # define some helpers with defaul arguments
    load = functools.partial(
        bnb.hi5.recursive_load, dtype=bdict, keepdim=True, skip=["trains"]
    )

    plot_2d = functools.partial(
        plot_disease_mean_number_of_infectious_encounter_2d,
        which="data",
        how=how,
        control_plot=False,
    )

    h5ref = load(
        file_path_shorthand("data"),
    )

    fig_kws = dict(dpi=300, transparent=True)

    # Contact duration
    fig, ax1d = plt.subplots(figsize=(8 * cm, 5 * cm))

    kwargs = dict(label="15min (main)")
    plot_conditional_rate(
        h5ref, ax=ax1d, which=["data"], control_plot=True, kwargs_overwrite=kwargs
    )
    ax = plot_2d(h5ref)
    ax.set_title("15min (main)")
    ax.get_figure().savefig(f"./figs/mins/2d_15min_{how}.pdf", **fig_kws)

    h5f = load("./out_min_sweep/results_Copenhagen_filtered_5min.h5")
    kwargs = dict(label="5min", lw=0.5)
    plot_conditional_rate(
        h5f, ax=ax1d, which=["data"], control_plot=True, kwargs_overwrite=kwargs
    )
    ax = plot_2d(h5f)
    ax.set_title("5min")
    ax.get_figure().savefig(f"./figs/mins/2d_5min_{how}.pdf", **fig_kws)

    h5f = load("./out_min_sweep/results_Copenhagen_filtered_10min.h5")
    kwargs = dict(label="10min", lw=0.5)
    plot_conditional_rate(
        h5f, ax=ax1d, which=["data"], control_plot=True, kwargs_overwrite=kwargs
    )
    ax = plot_2d(h5f)
    ax.set_title("10min")
    ax.get_figure().savefig(f"./figs/mins/2d_10min_{how}.pdf", **fig_kws)

    h5f = load("./out_min_sweep/results_Copenhagen_filtered_20min.h5")
    kwargs = dict(label="20min", lw=0.5)
    plot_conditional_rate(
        h5f, ax=ax1d, which=["data"], control_plot=True, kwargs_overwrite=kwargs
    )
    ax = plot_2d(h5f)
    ax.set_title("20min")
    ax.get_figure().savefig(f"./figs/mins/2d_20min_{how}.pdf", **fig_kws)

    h5f = load("./out_min_sweep/results_Copenhagen_filtered_30min.h5")
    kwargs = dict(label="30min", lw=0.5)
    plot_conditional_rate(
        h5f, ax=ax1d, which=["data"], control_plot=True, kwargs_overwrite=kwargs
    )
    ax = plot_2d(h5f)
    ax.set_title("30min")
    ax.get_figure().savefig(f"./figs/mins/2d_30min_{how}.pdf", **fig_kws)

    _set_size(fig.axes[0], 5.0 * cm, 3.5 * cm)
    fig.axes[0].set_ylim(0, 100)
    fig.savefig(f"./figs/mins/cer.pdf", **fig_kws)

    # RSSI conditional encounter rate
    fig, ax = plt.subplots(figsize=(8 * cm, 5 * cm))

    kwargs = dict(label="-80db (main)")
    plot_conditional_rate(
        h5ref, ax=ax, which=["data"], control_plot=True, kwargs_overwrite=kwargs
    )

    h5f = load("./out_rssi75/results_Copenhagen_filtered_15min.h5")
    kwargs = dict(label="-75db")
    plot_conditional_rate(
        h5f, ax=ax, which=["data"], control_plot=True, kwargs_overwrite=kwargs
    )

    h5f = load("./out_rssi95/results_Copenhagen_filtered_15min.h5")
    kwargs = dict(label="-95db")
    plot_conditional_rate(
        h5f, ax=ax, which=["data"], control_plot=True, kwargs_overwrite=kwargs
    )

    _set_size(fig.axes[0], 5.0 * cm, 3.5 * cm)
    fig.axes[0].set_ylim(0, 120)
    fig.savefig(f"./figs/rssi/cer.pdf", **fig_kws)

    # RSSI 2d plots
    ax = plot_2d(h5ref)
    ax.set_title("-80db (main)")
    ax.get_figure().savefig(f"./figs/rssi/2d_80_{how}.pdf", **fig_kws)

    h5f = load("./out_rssi75/results_Copenhagen_filtered_15min.h5")
    ax = plot_2d(h5f)
    ax.set_title("-75db")
    ax.get_figure().savefig(f"./figs/rssi/2d_75_{how}.pdf", **fig_kws)

    h5f = load("./out_rssi95/results_Copenhagen_filtered_15min.h5")
    ax = plot_2d(h5f)
    ax.set_title("-95db")
    ax.get_figure().savefig(f"./figs/rssi/2d_95_{how}.pdf", **fig_kws)


def figure_sm_rate_and_iei_complete(h5f):
    """
    plots for encounter rate and inter-encounter-intervals (fig 1, bottom)
    showing sampled process along the data
    """
    ax = plot_etrain_rate(h5f, sm_generative_processes=True)
    _set_size(ax, 5.0 * cm, 3.5 * cm)
    ax.get_figure().savefig(f"./figs/sm_ecr.pdf", **fig_kws)

    ax = plot_dist_inter_encounter_interval(h5f, sm_generative_processes=True)
    _set_size(ax, 5.0 * cm, 3.5 * cm)
    ax.get_figure().savefig(f"./figs/sm_iei.pdf", **fig_kws)


# decorator for lower level plot functions to continue if subplot fails
def warntry(func):
    def wrapper(*args, **kwargs):
        if debug:
            return func(*args, **kwargs)
        try:
            return func(*args, **kwargs)
        except Exception as e:
            log.exception(f"{func.__name__}: {e}")

    return wrapper


# Fig 1b
@warntry
def plot_etrain_rasters(h5f=None):

    if h5f is None:
        h5f = bnb.hi5.recursive_load(
            file_path_shorthand("data"), dtype=bdict, keepdim=True
        )

    fig, ax = plt.subplots()
    ax.set_rasterization_zorder(0)

    # load the trains
    trains = h5f["trains"]

    for idx, id_name in enumerate(trains["ids"]):
        train = trains[f"train_{idx+1}"] / 60 / 60 / 24
        # log.info(f"{len(train)} {trains['sort_by'][id_name]}")
        # if idx == 3: break
        # if len(train) == 0: continue
        ax.eventplot(
            positions=train,
            # lineoffsets = idx,
            lineoffsets=idx,
            linelengths=12.9,
            lw=0.2,
            alpha=0.5,
            # color=f'C{idx%5}',
            # color="#46718C",
            color=clrs["data"],
            zorder=-1,
            # color = mpl.cm.get_cmap("Spectral")(idx % 50 / 50)
        )
    ax.xaxis.set_major_locator(MultipleLocator(7))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_xlabel("days")
    ax.set_xlim(0, None)
    ax.yaxis.set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_title("Encounter trains")
    ax.get_figure().tight_layout()

    return ax


# Fig 1c
@warntry
def plot_etrain_rate(
    ax=None,
    which=["data"],
    plot_kwargs=None,
):
    if ax is None:
        with plt.rc_context(
            {
                "xtick.labelsize": 6,
                "ytick.labelsize": 6,
                # labels are weekdays, need more space. and we place them on the minors
                "xtick.minor.pad": 6,
            }
        ):
            fig, ax = plt.subplots(figsize=(6.5 * cm, 4.5 * cm))

    else:
        fig = ax.get_figure()
    ax.set_rasterization_zorder(0)

    norm_rate = 1 / 60 / 60 / 24

    for wdx, w in enumerate(which):
        file = file_path_shorthand(w)
        dset = "rate"

        data = bnb.hi5.load(file, dset, raise_ex=True, keepdim=True)

        r_time = data[0, :] / 60 / 60 / 24
        r_full = data[1, :] / norm_rate
        r_jack = data[2, :] / norm_rate
        r_errs = data[3, :] / norm_rate

        try:
            kwargs = plot_kwargs[w].copy()
        except:
            # no customizations set, use the defaults
            if w == "data":
                # data gets plotted as errorbars
                kwargs = default_plot_kwargs(w, for_errorbars=True)
            else:
                kwargs = default_plot_kwargs(w, for_errorbars=False)

        if w == "data":
            ax.errorbar(x=r_time, y=r_full, yerr=r_errs, **kwargs)
        else:
            ax.plot(r_time, r_full, **kwargs)

    ax.set_xlim(0, 7)
    ax.set_ylim(0, None)
    ax.yaxis.set_minor_locator(MultipleLocator(5))

    if show_xlabel:
        ax.set_xlabel(r"Time (days of the week)")
    if show_ylabel:
        ax.set_ylabel(r"Rate (encounters per day)")
    if show_title:
        ax.set_title(r"Encounter rate (1 / day)", loc="left", fontsize=8)
    if show_legend:
        ax.legend()
    if show_legend_in_extra_panel:
        _legend_into_new_axes(ax)

    # weekdays on x axis
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    # this is only true for copenhagen
    days = ["Su", "Mo", "Tu", "Wd", "Th", "Fr", "Sa"]

    try:
        # dirty workaround for invs15
        if "InVS15" in h5f["h5/filename"]:
            days = ["Mo", "Tu", "Wd", "Th", "Fr", "Sa", "Su"]
    except:
        pass

    def tick(x, pos):
        if x % 1 == 0.5 and x < 7:
            return days[int(x)]
        else:
            return ""

    ax.xaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(tick))
    ax.tick_params(which="minor", axis="x", length=0)

    ax.margins(x=0, y=0.0)
    # ax.legend()

    # fig.tight_layout()
    _set_size(ax, 3.1 * cm, 2.0 * cm)

    return ax


# Fig 1d, SM
@warntry
def plot_dist_inter_encounter_interval(
    ax=None, which=["data"], log_or_lin="log", plot_kwargs=None
):
    with plt.rc_context({"xtick.labelsize": 6, "ytick.labelsize": 6}):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6.5 * cm, 4.5 * cm))
        else:
            fig = ax.get_figure()
        ax.set_rasterization_zorder(0)

    # data is in seconds
    iei_norm = 1 / 60 / 60 / 24

    assert log_or_lin in ["log", "lin"]

    for wdx, w in enumerate(which):
        file = file_path_shorthand(w)
        dset = "distribution_inter_encounter_intervals"

        if log_or_lin == "log":
            dset += "_logbin"

        dat = bnb.hi5.load(file, dset, raise_ex=True, keepdim=True)[:]

        iei = dat[0, :] * iei_norm
        prob = dat[1, :]
        jack_iei = dat[2, :] * iei_norm
        jack_prob = dat[3, :]
        errs_iei = dat[4, :] * iei_norm
        errs_prob = dat[5, :]

        # ------------------------------------------------------------------------------ #
        # plot
        # ------------------------------------------------------------------------------ #

        try:
            kwargs = plot_kwargs[w].copy()
        except:
            # no customizations set, use the defaults
            if w == "data":
                # data gets plotted as errorbars
                kwargs = default_plot_kwargs(w, for_errorbars=True)
            else:
                kwargs = default_plot_kwargs(w, for_errorbars=False)

        # for data, we have error bars in x and y
        if w == "data":
            e_step = 1
            ax.errorbar(
                x=iei[::e_step],
                y=prob[::e_step],
                xerr=errs_iei[::e_step],
                yerr=errs_prob[::e_step],
                **kwargs,
            )

        else:
            ax.plot(iei, prob, **kwargs)

    # annotations
    ls_kwargs = dict(
        linestyle=(0, (0.01, 2)),
        dash_capstyle="round",
        color="#BFBFBF",
        lw=0.8,
        zorder=-1,
    )
    ax.axvline(1 / 24 / 60 * 5, 0, 1, **ls_kwargs)  # 5 min
    ax.axvline(1 / 24, 0, 1, **ls_kwargs)  # hour
    ax.axvline(1, 0, 1, **ls_kwargs)  # day
    ax.axvline(7, 0, 1, **ls_kwargs)  # week

    ax.set_xscale("log")
    ax.set_yscale("log")

    if show_xlabel:
        ax.set_xlabel(r"Inter-encounter inverval (days)")
    if show_ylabel:
        ax.set_ylabel(r"Probability density P(iei)")
    if show_title:
        ax.set_title("Distribution of iei", fontsize=8)
    if show_legend:
        ax.legend()
    if show_legend_in_extra_panel:
        _legend_into_new_axes(ax)

    ax.set_ylim(1e-11, 1e-3)

    _pretty_log_ticks(ax.xaxis, prec=2)

    # show less ticks in main manuscript
    _fix_log_ticks(
        ax.yaxis, every=2, hide_label_condition=lambda idx: not (idx + 2) % 4 == 3
    )
    _fix_log_ticks(
        ax.xaxis, every=1, hide_label_condition=lambda idx: not (idx + 2) % 1 == 0
    )

    fig.tight_layout()

    # _set_size(ax, 2.5 * cm, 2.0 * cm)
    _set_size(ax, 3.1 * cm, 2.0 * cm)
    return ax


# Fig 2b
@warntry
def plot_dist_encounters_per_day(
    which=["data", "data_rand"], ax=None, plot_kwargs=None, show_fit=False
):

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    for wdx, w in enumerate(which):
        log.info(f"distribution of encounter per day for {w}")
        file = file_path_shorthand(w)
        dset = "distribution_daily_number_encounters"

        data = bnb.hi5.load(file, dset, raise_ex=True, keepdim=True)

        num_full = data[0, :]
        p_full = data[1, :]
        p_errs = data[3, :]

        try:
            kwargs = plot_kwargs[w].copy()
        except:
            # no customizations set, use the defaults
            if w == "data":
                # data gets plotted as errorbars
                kwargs = default_plot_kwargs(w, for_errorbars=True)
            else:
                kwargs = default_plot_kwargs(w, for_errorbars=False)

        if w == "data":
            ax.errorbar(
                x=num_full,
                y=p_full,
                yerr=p_errs,
                **kwargs,
            )
        else:
            ax.plot(num_full, p_full, **kwargs)

        try:
            if show_fit is True or show_fit[w] is True:
                # here, we didnt do the fit in julia ...
                # fit exponential
                def fitfunc(x, offset, slope):
                    return offset + slope * x

                fitstart = 1
                fitend = None
                np.random.seed(815)
                y = np.log(p_full[fitstart:fitend])
                valid_idx = np.isfinite(y)
                x = num_full[fitstart:fitend]
                popt, pcov = curve_fit(
                    fitfunc,
                    xdata=x[valid_idx],
                    ydata=y[valid_idx],
                )

                # offset a bit so we can see it
                popt[0] += 2
                p_full_fit = np.exp(fitfunc(num_full, *popt))
                fitstart = 15
                fitend = 51
                # styling of fits is somewhat limited, but the kwargs
                # for errorbars are not really that useful.
                ax.plot(
                    num_full[fitstart:fitend],
                    p_full_fit[fitstart:fitend],
                    color=_alpha_to_solid_on_bg(kwargs["color"], 0.5),
                    label=kwargs["label"] + f" fit",
                )
        except:
            pass

    if show_xlabel:
        ax.set_xlabel(r"Number of encounters (per day)")
    if show_ylabel:
        ax.set_ylabel(r"Distribution")
    if show_title:
        ax.set_title(r"Encounters per Day", fontsize=8)
    if show_legend:
        ax.legend()
    if show_legend_in_extra_panel:
        _legend_into_new_axes(ax)

    # ax.legend()
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1)
    ax.set_xlim(-2, 54.5)

    _fix_log_ticks(ax.yaxis, every=2, hide_label_condition=lambda idx: (idx) % 2 == 0)
    ax.xaxis.set_major_locator(MultipleLocator(20))
    fig.tight_layout()
    _set_size(ax, 6.3 * cm, 2.2 * cm)

    # tiny = False
    tiny = True
    if tiny:
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(6)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(6)
        _set_size(ax, 2.6 * cm, 1.7 * cm)

    return ax


# Fig 2a
@warntry
def plot_dist_encounters_per_train(
    ax=None, which=["data", "data_rand"], plot_kwargs=None, show_fit=False
):

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    for wdx, w in enumerate(which):
        log.info(f"distribution of encounter per train for {w}")
        file = file_path_shorthand(w)
        dset = "distribution_total_number_encounter_linbin"

        data = bnb.hi5.load(file, dset, raise_ex=True, keepdim=True)

        # fit result, this is a number.
        try:
            exp_scale = bnb.hi5.load(
                file, dset.replace("_linbin", "_fit_exp"), raise_ex=True, keepdim=False
            )
        except:
            exp_scale = np.nan
            log.debug(f"couldnt load fit result for {w}")

        num_full = data[0, :]
        p_full = data[1, :]
        num_errs = data[4, :]
        p_errs = data[5, :]

        try:
            kwargs = plot_kwargs[w].copy()
        except:
            # no customizations set, use the defaults
            if w == "data":
                # data gets plotted as errorbars
                kwargs = default_plot_kwargs(w, for_errorbars=True)
            else:
                kwargs = default_plot_kwargs(w, for_errorbars=False)

        if w == "data":
            ax.errorbar(
                x=num_full,
                y=p_full,
                xerr=num_errs,
                yerr=p_errs,
                **kwargs,
            )
        else:
            ax.plot(num_full, p_full, **kwargs)

        try:
            if show_fit is True or show_fit[w] is True:
                num_c_max = np.nanmax(num_full) * 1.1
                # styling of fits is somewhat limited, but the kwargs
                # for errorbars are not really that useful.
                ax.plot(
                    np.arange(num_c_max),
                    scipy.stats.expon.pdf(np.arange(num_c_max), loc=0, scale=exp_scale),
                    color=_alpha_to_solid_on_bg(kwargs["color"], 0.5),
                    label=kwargs["label"] + f" fit",
                )
        except:
            pass

    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1.1e-2)
    ax.xaxis.set_major_locator(MultipleLocator(200))
    ax.xaxis.set_minor_locator(MultipleLocator(50))

    if show_xlabel:
        ax.set_xlabel(r"Number of encounters (per train)")
    if show_ylabel:
        ax.set_ylabel(r"Distribution")
    if show_title:
        ax.set_title(r"Encounters per Train", fontsize=8)
    if show_legend:
        ax.legend()
    if show_legend_in_extra_panel:
        _legend_into_new_axes(ax)

    fig.tight_layout()
    # _set_size(ax, 2.4 * cm, 1.4 * cm)
    _set_size(ax, 6.3 * cm, 2.2 * cm)

    # tiny = False
    tiny = True
    if tiny:
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(6)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(6)
        _set_size(ax, 2.6 * cm, 1.7 * cm)

    return ax


# Fig 2c
@warntry
def plot_etrain_raster_example(h5f, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    ax.set_rasterization_zorder(0)

    # usually we do not load the trains because they bloat space, so need to load
    # tid = 91
    tid = 134
    real_train = (
        bnb.hi5.load(h5f["h5/filename"], f"/data/trains/train_{tid}") / 60 / 60 / 24
    )
    surr_train = (
        bnb.hi5.load(
            h5f["h5/filename"],
            f"/data_surrogate_randomize_per_train/trains/train_{tid}",
        )
        / 60
        / 60
        / 24
    )

    ax.eventplot(
        positions=real_train,
        lineoffsets=2,
        linelengths=1,
        alpha=1,
        lw=0.5,
        color=clrs.weekday,
        label="data",
    )

    ax.eventplot(
        positions=surr_train,
        lineoffsets=4,
        linelengths=1,
        alpha=1,
        lw=0.5,
        color=clrs.data_randomized,
        label="surrogate",
    )

    ax.xaxis.set_major_locator(MultipleLocator(7))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    # ax.set_xlabel("Time (days)")
    ax.set_xlim(0, 28)
    ax.yaxis.set_visible(False)
    ax.get_figure().tight_layout()

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)

    return ax


# Fig 2e
@warntry
def plot_disease_mean_number_of_infectious_encounter_cutplane(
    ax=None,
    which=["data"],
    relative_to=None,
    plot_kwargs=None,
    t_inf=3,
):
    """
    # Parameters
    how : str, "relative" or "absolute"
    t_inf : for which infectious period [in days] to draw the cutplane
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(6 * cm, 2.5 * cm))
    else:
        fig = ax.get_figure()

    # load the data, but we need some extra details so load the recursrive,
    # and filter later
    dset = "disease/delta/scan_mean_number_infectious_encounter"

    def get_2d(w, dset):
        file = file_path_shorthand(w)
        h5f = bnb.hi5.recursive_load(
            file,
            dtype=bdict,
            keepdim=True,
            skip=["trains"],
        )
        h5f = h5f[dset]

        data_2d = h5f["mean"][:]
        range_inf = h5f["range_infectious"][:]
        range_lat = h5f["range_latent"][:]

        return data_2d, range_inf, range_lat

    for wdx, w in enumerate(which):

        data_2d, range_inf, range_lat = get_2d(w, dset)

        if relative_to is not None:
            norm_2d, norm_range_inf, norm_range_lat = get_2d(relative_to, dset)
            assert np.all(range_lat == norm_range_lat)
            assert np.all(range_inf == norm_range_inf)
            data_2d /= norm_2d
            data_2d *= 100  # in percent

        rdx = np.where(range_inf == t_inf)[0][0]
        data_1d = data_2d[rdx, :]

        try:
            kwargs = plot_kwargs[w].copy()
        except:
            kwargs = default_plot_kwargs(w)

        ax.plot(range_lat, data_1d, **kwargs)

    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.margins(x=0, y=0)
    if relative_to is None:
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.yaxis.set_minor_locator(MultipleLocator(1))
        ax.set_ylim(22, 30)

    if show_xlabel:
        ax.set_xlabel("Latent period (days)")
    if show_ylabel:
        how = "absolute" if relative_to is None else "relative"
        ax.set_ylabel(f"Encounters ({how})")

    fig.tight_layout()

    return ax


# Fig 2d, Fig 3b, SM
@warntry
def plot_conditional_rate(
    ax=None, which=["data"], shaded_regions=False, plot_kwargs=None
):
    """
    plot the conditional encounter rate from data or as obtained for generative processes

    # Parameters
    which : list of str
        e.g. `["data", "poisson_inhomogeneous", "poisson_inhomogeneous_weighted_trains"]`
        or `["data", "weibul_renewal_process", "weibul_renewal_process_weighted_trains"]`
    shaded_regions: bool,
        settings this to `True` disables the shaded area highlighting the areas of 3days infectioustness after tlat=2 and tlat=4, and sets different xlabels
    plot_kwargs: dict of dicts
        keys matching the strings in `which`
        passed to the plotting functions

    """

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    ax.set_rasterization_zorder(0)

    if not isinstance(which, list):
        which = list(which)

    norm_rate = 1 / 60 / 60 / 24
    e_step = 25

    for wdx, w in enumerate(which):
        log.info(f"conditional encounter rate for {w}")
        file = file_path_shorthand(w)
        dset = "conditional_encounter_rate"

        log.info(file)

        try:
            data = bnb.hi5.load(file, dset, raise_ex=True, keepdim=True)
        except:
            # w might be a loaded array
            data = np.copy(w)
            w = f"custom {wdx}"

        r_time = data[0, :] * norm_rate
        r_full = data[1, :] / norm_rate
        try:
            r_errs = data[1, :] / norm_rate
        except:
            pass

        try:
            kwargs = plot_kwargs[w].copy()
        except:
            # no customizations set, use the defaults
            kwargs = default_plot_kwargs(w)

        ax.plot(r_time, r_full, **kwargs)

        if w == "data" and shaded_regions:
            # shaded regions for examples: 2,3 and 6,3
            idx = np.where((r_time > 6) & (r_time < 9))
            ax.fill_between(
                r_time[idx],
                y1=np.zeros(len(idx)),
                y2=r_full[idx],
                color="#faad7c",
                alpha=0.4,
                lw=0,
                zorder=-3,
            )

            idx = np.where((r_time > 2) & (r_time < 5))
            ax.fill_between(
                r_time[idx],
                y1=np.zeros(len(idx)),
                y2=r_full[idx],
                color="#49737a",
                alpha=0.3,
                lw=0,
                zorder=-3,
            )

    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(0, 60)

    if shaded_regions:
        # in the first figure, where we show regions,
        # we want a bigger panel with more detailed ticks
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        for label in ax.xaxis.get_ticklabels()[::2]:
            label.set_visible(False)
        ax.set_xlim(0.0, 10.49)
    else:
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.yaxis.set_minor_locator(MultipleLocator(10))

    ax.margins(x=0, y=0)

    if show_xlabel:
        ax.set_xlabel(r"Time (days)")
    if show_ylabel:
        ax.set_ylabel(r"Rate (1/day)")
    if show_title:
        ax.set_title(r"Conditional encounter rate", fontsize=8)
    if show_legend:
        ax.legend()
    if show_legend_in_extra_panel:
        _legend_into_new_axes(ax)

    fig.tight_layout()
    _set_size(ax, 3.0 * cm, 1.8 * cm)

    return ax


# Fig 2f, Fig 3a, SM
@warntry
def plot_disease_mean_number_of_infectious_encounter_2d(
    ax=None, which="data", relative_to=None, control_plot=False
):
    """
    # Parameters
    which : str, "data" or samples from h5f["sample/"], e.g. "poisson_inhomogeneous"
    how : str, "relative" or "absolute"
    control_plot : bool, set to `True` for use in Fig. 4 for slightly different styling
    """

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    ax.set_rasterization_zorder(0)

    dset = "disease/delta/scan_mean_number_infectious_encounter"

    def get_2d(w, dset):
        file = file_path_shorthand(w)
        h5f = bnb.hi5.recursive_load(
            file,
            dtype=bdict,
            keepdim=True,
            skip=["trains"],
        )
        h5f = h5f[dset]

        data_2d = h5f["mean"][:]
        range_inf = h5f["range_infectious"][:]
        range_lat = h5f["range_latent"][:]

        return data_2d, range_inf, range_lat

    data_2d, range_inf, range_lat = get_2d(which, dset)

    # maybe we normalize
    if relative_to is not None:
        norm_2d, norm_range_inf, norm_range_lat = get_2d(relative_to, dset)
        assert np.all(range_lat == norm_range_lat)
        assert np.all(range_inf == norm_range_inf)
        data_2d /= norm_2d
        data_2d *= 100  # in percent

    # customize style
    # custom color maps
    palette = [
        (0, "#C31B2B"),
        (0.25, "#ffad7e"),
        (0.5, "#E7E7B6"),
        (0.85, "#195571"),
        (1, "#011A39"),
    ]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", palette, N=512)

    if relative_to is None:
        kwargs = dict(
            vmin=0,
            vmax=None,
            cmap=cmap.reversed(),
        )

    else:
        kwargs = dict(
            vmin=50,
            vmax=150,
            center=100,
            cmap=cmap.reversed(),
        )

    if control_plot:
        kwargs["cbar"] = False

    # draw!
    sns.heatmap(
        data_2d,
        ax=ax,
        square=True,
        xticklabels=False,
        yticklabels=False,
        zorder=-5,
        **kwargs,
    )

    xticklabels = []
    xticks = []
    for idx, x in enumerate(range_lat):
        if x.is_integer() and x > 0 and x < 8:
            xticks.append(idx)
            if control_plot and x % 2 != 1:
                xticklabels.append("")
            else:
                xticklabels.append(str(int(x)))

    yticklabels = []
    yticks = []
    for idy, y in enumerate(range_inf):
        if y.is_integer() and y > 0 and y < 8:
            yticks.append(idy)
            if control_plot and y % 2 != 1:
                yticklabels.append("")
            else:
                yticklabels.append(str(int(y)))

    # ax.set_xlabel("Latent period (days)")
    # ax.set_ylabel("Infectious period (days)")
    # ax.set_title(which)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.invert_yaxis()

    if not control_plot:
        if show_xlabel:
            ax.set_xlabel(r"Latent period (days)")
        if show_ylabel:
            ax.set_ylabel(r"Infectious period (days)")

    if show_title:
        ax.set_title(f"{which}", fontsize=8)

    fig.tight_layout()

    if control_plot:
        ax.tick_params(
            axis="both",
            which="major",
            length=1.5,
        )

    return ax


# new Fig 2
def plot_extinction_probability(
    h5f=None, ax=None, apply_formatting=True, which="analytic"
):
    """
    needs a different h5f than most plot functions:
    the branchin process one, by default `branching_process_Copenhagen_filtered_15min`
    """

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # TODO make consistent
    if h5f is None and (which == "analytic"):
        h5f = bnb.hi5.recursive_load(
            "./out/analytic_survival_Copenhagen_filtered_15min.h5",
            dtype=bdict,
            keepdim=True,
        )
    elif h5f is None and which == "branching_process":
        h5f = bnb.hi5.recursive_load(
            "./out/branching_process_Copenhagen_filtered_15min.h5",
            dtype=bdict,
            keepdim=True,
        )

    if which == "analytic":
        data_2 = "data/infectious_3.00_latent_2.00/survival_probability_p"
        data_6 = "data/infectious_3.00_latent_6.00/survival_probability_p"
        rand_2 = "rand/infectious_3.00_latent_2.00/survival_probability_p"
        rand_6 = "rand/infectious_3.00_latent_6.00/survival_probability_p"
        rand_all_2 = "rand_all/infectious_3.00_latent_2.00/survival_probability_p"
        rand_all_6 = "rand_all/infectious_3.00_latent_6.00/survival_probability_p"
    elif which == "branching_process":
        data_2 = "data/infectious_3.00_latent_2.00/survival_probability_p/N0=1/100000"
        data_6 = "data/infectious_3.00_latent_6.00/survival_probability_p/N0=1/100000"
        rand_2 = "rand/infectious_3.00_latent_2.00/survival_probability_p/N0=1/100000"
        rand_6 = "rand/infectious_3.00_latent_6.00/survival_probability_p/N0=1/100000"
        rand_all_2 = None
        rand_all_6 = None

    # for path in [data_2, data_6, rand_2, rand_6, rand_all_2, rand_all_6]:
    for path in [
        rand_all_2,
        rand_2,
        data_6,
        data_2,
    ]:

        plot_kwargs = dict()
        # customize plot style depending on data to plot
        if path is None:
            # not all defined for branching process
            continue
        if path[0:8] == "rand_all":
            # randomized across trains and individuals
            plot_kwargs = dict(linestyle=(0, (2.0, 2.0)), dash_capstyle="round")
            color = clrs["data_rand_all"]
            label = "randomized all"
            if "_2" not in path:
                color = _alpha_to_solid_on_bg(color, alpha=0.9, bg="black")
        elif path[0:4] == "rand":
            # randomized by train
            plot_kwargs = dict(linestyle=(0, (2.0, 2.0)), dash_capstyle="round")
            color = clrs["n_psn"]
            label = "randomized"
            if "_2" not in path:
                color = _alpha_to_solid_on_bg(color, alpha=0.9, bg="black")

        elif "latent_2.0" in path:
            color = clrs["n_low"]
            label = "data"
        elif "latent_6.0" in path:
            color = clrs["n_high"]
            label = "data"
        lat = re.search("latent_(\d+)", path, re.IGNORECASE).group(1)
        label += f" Tlat={lat}"

        data = h5f[path]
        x_prob = data[0, :]
        x_repr = data[1, :]
        y_surv = data[2, :]

        # we load survivial probability but decided to plot extinction probability
        y_ext = 1 - y_surv

        # limit xrange here, so we can keep clip_on=False
        idx = np.where((x_repr > 0.7) & (x_repr <= 5))[0]
        ax.plot(
            x_repr[idx],
            y_ext[idx],
            color=color,
            label=label,
            clip_on=False,
            **plot_kwargs,
        )

    if apply_formatting:
        ax.set_xlim(0.7, 5)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        # ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.set_xticks([1, 3, 5])
        ax.xaxis.set_minor_locator(MultipleLocator(1))

        sns.despine(ax=ax, trim=False, offset=3)

        if show_xlabel:
            ax.set_xlabel(r"Reproduction number $R_0$")
        if show_ylabel:
            ax.set_ylabel("Extinction probability")
        # if show_title:
        # ax.set_title(f"{periods}", fontsize=8)
        if show_legend:
            ax.legend()
        if show_legend_in_extra_panel:
            _legend_into_new_axes(ax)

        fig.tight_layout()
        # _set_size(ax, 5.5*cm, 3.5*cm)

    return ax


# Fig 4b
@warntry
def plot_gamma_distribution(ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    ax.set_rasterization_zorder(0)

    mean = 2
    x = np.arange(0, mean * 4, 0.01)  # days, delta peak for k-> inf at `loc`

    colors = dict()
    for idx, k in enumerate([1, 10, 100, 1e8]):
        colors[k] = pt.scientific.sequential.Tokyo_4_r.mpl_colors[idx]
        colors[1] = "#E0E3B2"

    k_vals = [1e8, 100, 10, 1]
    for idx, k in enumerate(k_vals):
        y = scipy.stats.gamma.pdf(x, a=k, loc=0, scale=mean / k)
        k_str = f"$k={k}$" if k != 1e8 else r"$k\to\infty$"
        ax.plot(x, y, color=colors[k], label=k_str)

    ax.set_ylim(1e-2, 5)
    ax.set_xlim(0, 11.49)
    ax.set_yscale("log")
    ax.set_xlabel("Period (days)")
    ax.set_ylabel("Distribtuion")

    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.xaxis.set_major_locator(MultipleLocator(2))

    _legend_into_new_axes(ax)
    _fix_log_ticks(ax.yaxis, every=1, hide_label_condition=lambda idx: False)

    fig.tight_layout()
    _set_size(ax, 3.3 * cm, 1.7 * cm)
    return ax


# Fig 4d, SM
@warntry
def plot_dispersion_scan_k(h5f, ax=None, periods="slow"):

    assert periods in ["fast", "slow"]
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    def local_plot(data, color, fmt, zorder=0, label=None, hline=None):

        kval = data[0, :]
        full = data[1, :]
        errs = data[3, :]

        ax.errorbar(
            x=kval,
            y=full,
            yerr=errs,
            fmt=fmt,
            markersize=ms_default,
            markerfacecolor="white",
            color=color,
            ecolor=color,
            alpha=1,
            elinewidth=0.5,
            capsize=1,
            zorder=zorder,
            label=label,
        )

        if hline is not None:
            ax.axhline(
                hline,
                0,
                1,
                ls=":",
                zorder=zorder - 5,
                color=_alpha_to_solid_on_bg(color, alpha=0.5),
            )

    p_todo = []
    if periods == "slow":
        p_todo.append("2_3")  # blue
        p_todo.append("6_3")  # red
        p_todo.append("2_3_surrogate")
        p_todo.append("6_3_surrogate")
    elif periods == "fast":
        p_todo.append("1_0.5")  # blue
        p_todo.append("1.5_0.5")  # red
        p_todo.append("1_0.5_surrogate")
        p_todo.append("1.5_0.5_surrogate")

    c_todo = []  # colors
    c_todo.append(clrs["n_low"])
    c_todo.append(clrs["n_high"])
    c_todo.append(clrs["n_psn"])
    c_todo.append(clrs["n_psn"])

    m_todo = []  # markers
    m_todo.append("s")
    m_todo.append("o")
    m_todo.append("s")
    m_todo.append("o")

    # iterate over all periods and chosen colors
    for period, color, fmt in zip(p_todo, c_todo, m_todo):

        data = h5f[f"disease/gamma_{period}/scan_k"][:]

        # calculate the k->inf limit from delta disease
        hline = h5f[f"disease/delta_{period}/mean_number_infectious_encounter"][0]

        log.info(f"{period}:\t{hline}")

        zorder = 0
        if "surrogate" in period:
            zorder = -5

        local_plot(data, color, fmt, zorder=zorder, label=period, hline=hline)

    ax.set_xscale("log")
    ax.set_xlim(0.1, 1.5e5)
    # ax.legend()
    _fix_log_ticks(ax.xaxis)
    # ax.yaxis.set_minor_locator(MultipleLocator(1))
    # ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.margins(x=0, y=0)
    # ax.set_ylim(21, 31)

    if show_xlabel:
        ax.set_xlabel("Dispersion $k$")
    if show_ylabel:
        ax.set_ylabel("Pot. inf. encounters")
    if show_title:
        ax.set_title(f"{periods}", fontsize=8)
    if show_legend:
        ax.legend()
    if show_legend_in_extra_panel:
        _legend_into_new_axes(ax)

    fig.tight_layout()
    _set_size(ax, 4.3 * cm, 3.3 * cm)
    # _set_size(ax, 4.3 * cm, 2.0 * cm)

    return ax


# Fig 4a
@warntry
def plot_disease_viral_load_examples():
    fig, axes = plt.subplots(
        nrows=3,
        sharex=True,
        sharey=True,
    )

    t_max = 14  # days
    hist_sample = 100000
    exsample = 1000
    exsample2 = 3

    np.random.seed(817)

    def disease_progression(k, mean_latent, mean_infectious, size=100):
        t_start = scipy.stats.gamma.rvs(a=k, loc=0, scale=mean_latent / k, size=size)
        t_end = t_start + scipy.stats.gamma.rvs(
            a=k, loc=0, scale=mean_infectious / k, size=size
        )

        # naive histogram, hour resolution
        hist = np.zeros(int(t_max * 24 * 60 + 1))

        for tdx in range(0, len(t_start)):
            t0 = int(t_start[tdx] * 24 * 60)
            t1 = int(t_end[tdx] * 24 * 60)
            if t1 >= len(hist):
                t1 = int(len(hist) - 1)
            hist[t0:t1] += 1

        # return hist, t_start, t_end
        return hist, t_start, t_end

    for idx, k in enumerate([1e8, 10, 1]):
        ax = axes[idx]
        ax.set_rasterization_zorder(0)

        hist, t_start, t_end = disease_progression(k, 2, 3, size=hist_sample)
        x = np.arange(len(hist)) / 24 / 60

        ax.plot(x, hist, zorder=5, ls="--", color=clr_dispersion[k])
        ax.set_xlim(0, 11.49)

        for tdx in range(0, exsample):
            y = np.zeros(len(hist))
            t0 = int(t_start[tdx] * 24 * 60)
            t1 = int(t_end[tdx] * 24 * 60)
            if t1 >= len(hist):
                t1 = int(len(hist) - 1)
            y[t0:t1] += hist_sample
            if tdx < exsample2:
                ax.plot(x, y, alpha=1, zorder=2, color="white", lw=1.5)
                ax.plot(x, y, alpha=1, zorder=2, color=clr_dispersion[k], lw=0.5)
            else:
                ax.plot(x, y, alpha=0.02, zorder=-1, color=clr_dispersion[k], lw=0.5)

    ax = axes[-1]
    ax.set_xlabel("Time (in days)")
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.xaxis.set_major_locator(MultipleLocator(2))
    # for idx, lab in enumerate(ax.xaxis.get_ticklabels()):
    #     if (idx) % 2 == 0:
    #         lab.set_visible(False)
    fig.tight_layout()


@warntry
def plot_disease_dist_infectious_encounters(
    which=["data", "data_rand", "data_rand_all"],
    ax=None,
    k="k_inf",
    periods="slow",
    plot_kwargs=None,
    control=None,
    annotate=True,
):

    assert k in ["k_inf", "k_10.0", "k_1.0"]
    assert periods in ["fast", "slow", "2_3", "6_3", "1_0.5", "1.5_0.5"]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    p_todo = []
    if periods == "slow":
        p_todo.append("2_3")  # bias towards low
        p_todo.append("6_3")  # bias towards high
    elif periods == "fast":
        p_todo.append("1_0.5")  # low
        p_todo.append("1.5_0.5")  # high
    else:
        p_todo.append(periods)

    for wdx, w in enumerate(which):
        file = file_path_shorthand(w)

        # iterate over all periods and chosen colors
        for pdx, period in enumerate(p_todo):

            # stitch together dset
            # depending on k, the path may differ (for k_inf we have delta disase)
            if k == "k_inf":
                dset = f"disease/delta_{period}"
            else:
                dset = f"disease/gamma_{period}/{k}"

            if control is not None:
                dset += f"/control_random_disease_{control}"
            dset += "/distribution_infectious_encounter"

            try:

                data = bnb.hi5.load(file, dset, raise_ex=True, keepdim=True)
                num_encounter = data[0, :]
                p_full = data[1, :]
                p_jack = data[2, :]
                p_errs = data[3, :]

                # here we hardcode the styles, instead of using defaults.
                if plot_kwargs is None:
                    plot_kwargs = dict()

                # comparing data vs randomized, fig 2
                if w == "data":
                    if pdx == 0:
                        color = clrs["n_low"]
                        zorder = 3
                    elif pdx == 1:
                        color = clrs["n_high"]
                        zorder = 2
                elif w == "data_rand":
                    if pdx == 0:
                        color = clrs["data_rand"]
                    elif pdx == 1:
                        color = _alpha_to_solid_on_bg(clrs["data_rand"], alpha=0.8)
                    zorder = 1
                    # plot_kwargs.setdefault("linestyle", (0, (0.01, 1.5)))
                    plot_kwargs.setdefault("linestyle", (0, (2.0, 2.0)))
                    plot_kwargs.setdefault("dash_capstyle", "round")
                    # plot_kwargs.setdefault("lw", 2)
                elif w == "data_rand_all":
                    if pdx == 0:
                        color = clrs["data_rand_all"]
                    elif pdx == 1:
                        color = _alpha_to_solid_on_bg(
                            clrs["data_rand_all"], alpha=0.9, bg="black"
                        )
                    zorder = 0
                    plot_kwargs.setdefault("linestyle", (0, (2.0, 2.0)))
                    plot_kwargs.setdefault("dash_capstyle", "round")

                ax.plot(num_encounter, p_full, color=color, zorder=zorder, **plot_kwargs)
                ref = _ev(num_encounter, p_full)
                ax.axvline(
                    ref,
                    0,
                    0.01,
                    linestyle=(0, (1, 2)),
                    dash_capstyle="round",
                    lw=0.8,
                    # color=_alpha_to_solid_on_bg(color, 0.5),
                    color=_alpha_to_solid_on_bg(color, 1.0),
                    zorder=zorder - 10,
                )

                log.info(f"{w}\t{period}:\t{ref:.2f}")
            except Exception as e:
                log.warning(f"Failed to plot {w} {dset} {period}")
                log.error(e)

    ax.set_xlim(-5, 100)
    ax.set_yscale("log")
    if periods == "slow":
        ax.set_ylim(1e-6, 1)
        # ax.set_ylim(1e-3, 1)
    elif periods == "fast":
        ax.set_ylim(1e-6, 1)

    _fix_log_ticks(ax.yaxis, every=1)
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_minor_locator(MultipleLocator(10))

    # ax.set_xlabel(r"Pot. inf. encounters ei"))
    # ax.set_ylabel(r"Probability P(ei)")

    if k == "k_inf":
        title = r"$k\to\infty$"
    else:
        title = f"$k = {float(k[2:]):.0f}$"

    title += f"    {periods}"
    if control is not None:
        title += f" {control}"

    if annotate:
        if show_xlabel:
            ax.set_xlabel(r"Pot. inf. encounters")
        if show_ylabel:
            ax.set_ylabel(r"Distribution")
        if show_title:
            ax.set_title(title, fontsize=8)
        if show_legend:
            ax.legend()
        if show_legend_in_extra_panel:
            _legend_into_new_axes(ax)

    return ax


@warntry
def plot_disease_dist_secondary_infections(
    which=["data", "data_rand", "data_rand_all"],
    R=3.0,
    ax=None,
    k="k_inf",
    periods="slow",
    control=None,
    annotate=True,
    plot_kwargs=None,
):
    """
    In the branching process approximation, we can analytically calculate the number of secondary infections, given the empirical distribution of pot. infectious encounters.

    # Parameters
    which : list of str
        datasets to plot, the usual ["data", "data_rand", "data_rand_all"]
    R : float
        for which R to plot the dist.
        we select the constant p_inf in Eq1 from R = _ev(n_inf) * p_inf
    k : str
        usually "k_inf" to use delta disease.
    periods : str
        the combintations of T_lat and T_inf
        "slow" for T_lat = 2 | 6 with T_inf = 3
        "fast" fot T_lat = 1 | 1.5 with T_inf = 0.5
        "2_3" or "6_3" to skip the comparison and only plot one
    """

    assert k in ["k_inf", "k_10.0", "k_1.0"]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    p_todo = []
    if periods == "slow":
        p_todo.append("2_3")  # bias towards low
        p_todo.append("6_3")  # bias towards high
    elif periods == "fast":
        p_todo.append("1_0.5")  # low
        p_todo.append("1.5_0.5")  # high
    else:
        # we may want to pass manually
        p_todo = [periods]

    for wdx, w in enumerate(which):
        file = file_path_shorthand(w)

        # iterate over all periods
        for pdx, period in enumerate(p_todo):

            log.info(f"{w}\t{period}")

            # stitch together dset
            # depending on k, the path may differ (for k_inf we have delta disase)
            if k == "k_inf":
                dset = f"disease/delta_{period}"
            else:
                dset = f"disease/gamma_{period}/{k}"

            if control is not None:
                dset += f"/control_random_disease_{control}"
            dset += "/distribution_infectious_encounter"

            try:

                data = bnb.hi5.load(file, dset, raise_ex=True, keepdim=True)
                num_encounter = data[0, :]
                p_full = data[1, :]
                p_jack = data[2, :]
                p_errs = data[3, :]

                # here we hardcode the styles, instead of using defaults.

                # comparing data vs randomized, fig 2
                if w == "data":
                    mrk = "s"
                    if period == "2_3" or period == "1_0.5":
                        color = clrs["n_low"]
                    elif period == "6_3" or period == "1.5_0.5":
                        color = clrs["n_high"]
                    zorder = 2
                elif w == "data_rand":
                    mrk = "o"
                    if pdx == 0:
                        color = clrs["data_rand"]
                    elif pdx == 1:
                        color = _alpha_to_solid_on_bg(clrs["data_rand"], alpha=0.8)
                    zorder = 6
                elif w == "data_rand_all":
                    mrk = "^"
                    if pdx == 0:
                        color = clrs["data_rand_all"]
                    elif pdx == 1:
                        color = _alpha_to_solid_on_bg(
                            clrs["data_rand_all"], alpha=0.9, bg="black"
                        )
                    zorder = 4

                kwargs = plot_kwargs.copy() if plot_kwargs is not None else {}
                kwargs.setdefault("color", color)
                kwargs.setdefault("zorder", zorder)
                kwargs.setdefault("marker", mrk)
                kwargs.setdefault("alpha", 1)
                kwargs.setdefault("label", f"{w} {period}")
                kwargs.setdefault("s", 1.3 if w == "data_rand" else ms_default)

                # convert R to p_inf, depending on the expected number of pot inf enc
                mean_n_inf = _ev(num_encounter, p_full)
                p_inf = R / mean_n_inf

                # scatter of the data
                x_vals, p_x = _offspring_dist(num_encounter, p_full, p_inf)
                ax.scatter(x_vals, p_x, clip_on=False, **kwargs)

                # plot the negative binomial fit
                # instead of just connecting the scattered data points
                kwargs["color"] = _alpha_to_solid_on_bg(kwargs["color"], 0.8)
                kwargs["zorder"] -= 1
                kwargs.pop("marker")
                kwargs.pop("s")
                kwargs.setdefault("lw", 0.8)
                # ax.plot(x_vals, p_x, **kwargs)

                try:
                    # retreive fit values from 2d sweep.
                    tlat, tift = period.split("_")  # "2_3" etc
                    coords = dict(latent=float(tlat), infectious=float(tift), R0=float(R))
                    r = float(_dispersion_data_prep(coords=coords, par="r", which=w))
                    p = float(_dispersion_data_prep(coords=coords, par="p", which=w))
                    log.info(f"negative binomial for {coords}: r: {r:.3f}, p: {p:.3f}")

                    # negative binomial from scipy matches julia pretty close
                    # p = p, r = n, but we converted p of success to failure in _data_prep
                    p = 1 - p
                    x_vals = np.arange(0, 26)
                    y_vals = scipy.stats.nbinom.pmf(x_vals, r, p)

                    # mean = p * r / (1 - p)
                    # var = p * r / (1 - p) ** 2
                    # log.info(f"mean: {mean:.3f}, var: {var:.3f}, r: {r:.3f} p: {p:.3f}")
                    # log.info(x_vals[0:10])
                    # log.info(y_vals[0:10])

                    ax.plot(x_vals, y_vals, clip_on=False, **kwargs)

                except Exception as e:
                    log.error(f"Failed to retreive fit parameters: {e}")

            except Exception as e:
                log.warning(f"Failed to plot {w} {dset} {period}")
                log.exception(e)

    # _fix_log_ticks(ax.yaxis, every=1)
    ax.set_xlim(0, 25)
    ax.set_ylim(0, None)
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_major_locator(MultipleLocator(20))
    sns.despine(ax=ax, trim=False, offset=3)
    # we probably want to compare with Lloyd Smith

    if k == "k_inf":
        title = r"$k\to\infty$"
    else:
        title = f"$k = {float(k[2:]):.0f}$"

    title += f"    {periods}"
    if control is not None:
        title += f" {control}"

    if annotate:
        if show_xlabel:
            ax.set_xlabel(r"Secondary infections")
        if show_ylabel:
            ax.set_ylabel(r"Distribution")
        if show_title:
            ax.set_title(title, fontsize=8)
        if show_legend:
            ax.legend()
        if show_legend_in_extra_panel:
            _legend_into_new_axes(ax)

    fig.tight_layout()

    return ax


def _offspring_dist(n_infs, p_n_infs, p, x_max=25):
    """
    Eq 1 of our paper,
    offspring distribution from empirical P(n_inf) in the branching approximation
    """

    n_max = n_infs[-1]
    x_vals = np.arange(0, x_max + 1)
    p_x = np.zeros(x_vals.shape)

    for xdr, x in enumerate(x_vals):
        n_beg = x
        n_end = n_max
        if n_beg > n_end:
            continue

        n_vals = np.arange(n_beg, n_end + 1)
        n_idxs = n_infs.searchsorted(n_vals)

        p_x[xdr] = np.sum(
            p_n_infs[n_idxs]
            * comb(n_vals, x)
            * np.power(p, x)
            * np.power((1 - p), (n_vals - x))
        )

    return x_vals, p_x


def sm_moments():

    which = ["data_rand_all", "data_rand", "data"]
    moments = [1, 2, 3, 4]
    axes = []
    for mdx, m in enumerate(moments):
        _, ax = plt.subplots()
        axes.append(ax)
        for wdx, w in enumerate(which):
            plot_offspring_moments(
                ax=ax,
                which=w,
                moment=m,
                period="2_3",
                normalize=True,
                plot_kwargs=dict(
                    color=_alpha_to_solid_on_bg(f"C{mdx}", bnb.plt.fade(wdx, len(which))),
                    label=f"{w} {m}",
                ),
            )
            if which == "data":
                # also plot another period?
                plot_offspring_moments(
                    ax=ax,
                    which=w,
                    moment=m,
                    normalize=True,
                    period="6_3",
                    plot_kwargs=dict(
                        color=_alpha_to_solid_on_bg(
                            f"C{mdx}", bnb.plt.fade(wdx, len(which))
                        ),
                        label=f"{w} {m} 6_3",
                    ),
                )
        ax.legend()
    return axes


# TODO: add which = "analytical poisson"
@warntry
def plot_offspring_moments(
    which="data",
    moment=1,
    ax=None,
    k="k_inf",
    period="2_3",
    control=None,
    annotate=True,
    normalize=False,
    plot_kwargs=None,
):
    """
    This is a lower-level function. no iteration over different data types,
    etc. one `which` one `period`

    plot_kwargs is passed to ax.plot

    # Parameters
    which : str
        datasets to plot, the usual "data", "data_rand", "data_rand_all"
    R : float
        for which R to plot the dist.
        we select the constant p_inf in Eq1 from R = _ev(n_inf) * p_inf
    k : str
        usually "k_inf" to use delta disease.
    period : str
        "2_3", "6_3", "1.5_0.5", "0.5_0.5"
    """

    assert k in ["k_inf", "k_10.0", "k_1.0"]
    assert period in ["2_3", "6_3", "1.5_0.5", "0.5_0.5"]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    w = which
    file = file_path_shorthand(w)

    # stitch together dset
    # depending on k, the path may differ (for k_inf we have delta disase)
    if k == "k_inf":
        dset = f"disease/delta_{period}"
    else:
        dset = f"disease/gamma_{period}/{k}"

    if control is not None:
        dset += f"/control_random_disease_{control}"
    dset += "/distribution_infectious_encounter"

    try:
        data = bnb.hi5.load(file, dset, raise_ex=True, keepdim=True)
        num_encounter = data[0, :]
        p_full = data[1, :]
        p_jack = data[2, :]
        p_errs = data[3, :]

        R_vals = np.arange(0.1, 3.1, 0.1)
        y_vals = np.ones(R_vals.shape) * np.nan

        for rdx, R in enumerate(R_vals):
            # convert R to p_inf
            mean_n_inf = _ev(num_encounter, p_full)
            p_inf = R / mean_n_inf

            # get offspring distribution from dist of pot inf encounters
            x_vals, p_x = _offspring_dist(num_encounter, p_full, p_inf)

            # get the moment
            moments = _stat_measures(x_vals, p_x)
            y_vals[rdx] = moments[moment]

        # we could normalize by what would be constant for the poisson process
        if normalize:
            if moment == 1:
                y_vals = y_vals / np.power(R_vals, 1.0)
            elif moment == 2:
                y_vals = y_vals / np.power(R_vals, 1.0)
            elif moment == 3:
                y_vals = y_vals / np.power(R_vals, -0.5)
            elif moment == 4:
                y_vals = (y_vals - 3) / np.power(R_vals, -1.0)

        plot_kwargs = plot_kwargs.copy() if plot_kwargs is not None else {}
        plot_kwargs.setdefault("label", f"{w} {period} m{moment}")

        ax.plot(R_vals, y_vals, **plot_kwargs)
        ax.set_ylim(1e-6, 12)

        log.info(f"{w}\t{dset}\t{period}")
    except Exception as e:
        log.warning(f"Failed to plot {w} {dset} {period}")
        log.exception(e)

    # _fix_log_ticks(ax.yaxis, every=1)
    # ax.xaxis.set_major_locator(MultipleLocator(50))
    # ax.xaxis.set_minor_locator(MultipleLocator(10))

    # ax.set_xlabel(r"Pot. inf. encounters ei"))
    # ax.set_ylabel(r"Probability P(ei)")

    if annotate:
        if show_xlabel:
            ax.set_xlabel(r"Reproduction number $R_0$")
        if show_ylabel:
            ax.set_ylabel(r"Statistics")
        if show_legend:
            ax.legend()
        if show_legend_in_extra_panel:
            _legend_into_new_axes(ax)

    fig.tight_layout()

    return ax


def plot_dispersion_cutplane(
    coords, x_dim=None, par="r", which="data", ax=None, **plot_kwargs
):
    """
    Plot of the Maximumlikelihood fits of a Negative Binomial to Offspring distributions.

    # Parameters
    par : str
        "r" or "p", which fit parameter to plot. The "r" parameter is what Lloyd-Smith
        commonly denotes with "k" the dispersion parameter. Lower "k" -> more dispersion.
    coords : dict with keys
        "R0", "infectious", "latent"
        mapping to selection in the ndim array
    """
    plot_kwargs = plot_kwargs.copy()

    # dimensionality checks and prep
    dims = ["R0", "infectious", "latent"]
    if x_dim is None:
        try:
            # use the one thats not specified
            x_dim = [d for d in dims if d not in coords.keys()][0]
        except:
            raise ValueError(
                "`x_dim` must be specified if you specify all 3 keys in `coords`"
            )

    # get the data
    ndim_data = _dispersion_data_prep(coords, par, which)

    # we allow one dim to iterate over to get multiple, fading lines
    dim_lens = [len(ndim_data[k]) for k in dims]
    assert 1 in dim_lens, f"Need at least one dimension to be of length 1"
    noniterdim = dims[dim_lens.index(1)]
    # the iterdim might be of len 1, too, but thats okay.
    iterdim = [d for d in dims if d not in [x_dim, noniterdim]][0]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # put noniterdim value into legend below
    nk_val = ndim_data[noniterdim].values[0]

    # iterate over the remaining dim
    for kdx, k_val in enumerate(ndim_data[iterdim].to_numpy()):
        this_ndim_data = ndim_data.sel({iterdim: k_val})
        # sanity check, only have one dimension remaining at this point
        assert len([d for d in this_ndim_data.shape if d > 1]) <= 1

        this_ndim_data = this_ndim_data.squeeze()

        kwargs = plot_kwargs.copy()
        kwargs.setdefault("label", f"{iterdim}={k_val}, {noniterdim}={nk_val}")
        kwargs.setdefault(
            "alpha", bnb.plt.fade(kdx, len(ndim_data[iterdim]), invert=True)
        )

        # this is horribly redundant code with dist_secondary_infections...

        if which == "data":
            mrk = "s"
            zorder = 2
        elif which == "data_rand":
            mrk = "o"
            zorder = 6
        elif which == "data_rand_all":
            mrk = "^"
            zorder = 4

        kwargs.setdefault("color", clrs.get(which, "black"))
        kwargs.setdefault("marker", mrk)
        kwargs.setdefault("zorder", zorder)
        kwargs.setdefault("s", ms_default)

        ax.scatter(
            this_ndim_data[x_dim],
            this_ndim_data,
            clip_on=False,
            **kwargs,
        )

    # customization for our paper case where tlat is the x axis
    if x_dim == "latent":
        ax.set_xlim(0.01, 7.99)
        ax.set_ylim(0.0, 0.8)
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.xaxis.set_ticks([1, 7])
        ax.yaxis.set_minor_locator(MultipleLocator(0.25))
    elif x_dim == "R0":
        # match y, we will hack this together in post
        ax.set_ylim(0.0, 0.8)
        ax.set_xlim(1, 5)
        ax.xaxis.set_ticks([1, 5])
        ax.yaxis.set_minor_locator(MultipleLocator(0.25))
        ax.yaxis.set_major_formatter(NullFormatter())

    xlabels = dict(
        R0="Reproduction number $R_0$",
        infectious="Infectious period",
        latent="Latent period",
    )
    ylabels = dict(
        r="Dispersion parameter $r$", p="Probability of success $p$", disp="Dispersion"
    )

    if show_xlabel:
        ax.set_xlabel(xlabels.get(x_dim, x_dim))
    if show_ylabel:
        ax.set_ylabel(ylabels.get(par, par))
    if show_title:
        ax.set_title(f"{which}")
    if show_legend:
        ax.legend()

    return ax


def plot_dispersion_2d(
    x_dim, y_dim, off_dim_coord, par="r", which="data", ax=None, **plot_kwargs
):
    """
    2D Plot of the Maximumlikelihood fits of a Negative Binomial to Offspring
    distributions.

    # Parameters
    par : str
        "r" or "p", which fit parameter to plot. The "r" parameter is what Lloyd-Smith
        commonly denotes with "k" the dispersion parameter. Lower "k" -> more dispersion.
    x_dim, y_dim : str
        "R0", "infectious", "latent"
        what to put where
    off_dim_coord : float
        where to cut the remaining dimension
    """
    plot_kwargs = plot_kwargs.copy()

    dims = ["R0", "infectious", "latent"]
    off_dim = [d for d in dims if d not in [x_dim, y_dim]][0]

    # get the data
    ndim_data = _dispersion_data_prep({off_dim: off_dim_coord}, par, which)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # lets use xarrays 2d plotting for now. Wraps matplotlib.pyplot.pcolormesh()
    # https://docs.xarray.dev/en/stable/generated/xarray.plot.pcolormesh.html#xarray.plot.pcolormesh
    ndim_data.plot(
        ax=ax, x=x_dim, y=y_dim, cbar_kwargs={"label": f"{par}"}, **plot_kwargs
    )

    ax.set_title(f"{which} {off_dim}={off_dim_coord}")

    return ax


def _dispersion_data_prep(coords, par, which):
    """
    # Parameters
    coords: dict with "latent", "infectious", "R0" specifying the coordinate _labels_
        (not indices)
    par: str
        "r" number of successes,
        "p" probability of success
        "disp" dispersion, 1/r
        "mean"
        "var"
    """

    dims = ["R0", "infectious", "latent"]
    coords = coords.copy()

    file = file_path_shorthand(which)
    group = "/disease/delta/scan_offspring_as_negative_binomial"
    data = bnb.hi5.recursive_load(file, group, keepdim=True)
    # ├── NB_p .................... ndarray  (5, 16, 17)
    # ├── NB_r .................... ndarray  (5, 16, 17)
    # ├── range_R0 ................ ndarray  (5,)
    # ├── range_infectious ........ ndarray  (16,)
    # └── range_latent ............ ndarray  (17,)

    r = data["NB_r"]
    p = data["NB_p"]
    # convert from julia to wiki definition
    p = 1 - p
    # now, p is the probability of _failure_, r is the number of successes
    if par == "r":
        ndim_data = r
    elif par == "p":
        ndim_data = p
    elif par == "disp":
        ndim_data = 1 / r
    elif par == "mean":
        ndim_data = r * p / (1 - p)
    elif par == "var":
        ndim_data = r * p / (1 - p) ** 2
    elif par == "jz":
        mean = r * p / (1 - p)
        var = r * p / (1 - p) ** 2
        ndim_data = var / mean - 1
    else:
        raise KeyError(f"unknow parameter {par}. known: r, p, mean, var")

    # we want lists in the coordinates for our xarray selection to work
    for k, v in coords.items():
        if not isinstance(v, (list, np.ndarray, tuple)):
            coords[k] = [v]

    # make our life easier by accessing dimensions by name
    ndim_data = xr.DataArray(
        data=ndim_data,
        dims=dims,
        coords={k: data[f"range_{k}"] for k in dims},
    )

    # select the subset
    ndim_data = ndim_data.sel(coords, method="nearest")

    return ndim_data


@warntry
def compare_disease_dist_encounters_generative(
    ax=None,
    which=["data"],
    periods=["2_3"],
    set_size=True,
    annotate=True,
):
    """

    here we hardcoded plot kwargs

    similar to above, but instead of comparing to randomized, we compare to generative
    process

    # Parameters
    process : "psn", "wbl" or "tlrd" corresponding to usual shorthands


    """

    control = None  # for the si, we have other datasets that e.g. exclude some trains

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    def local_plot(data, color, zorder=0, **kwargs):
        num_encounter = data[0, :]
        p_full = data[1, :]
        p_jack = data[2, :]
        p_errs = data[3, :]
        ax.plot(num_encounter, p_full, color=color, zorder=zorder, **kwargs)
        ref = _ev(num_encounter, p_full)
        return ref

    # original data is stored in /disease
    # poisson is stored in /sample/psn_inh.../disease

    # iterate over all periods and chosen colors
    # periods are saved as dset path
    for period in periods:
        if period == "2_3" or period == "1_0.5":
            period_color = clrs["n_low"]  # blue
        elif period == "6_3" or period == "1.5_0.5":
            period_color = clrs["n_high"]  # red

        dset = f"disease/delta_{period}"

        if control is not None:
            dset += f"/control_random_disease_{control}"
        dset += "/distribution_infectious_encounter"

        # iterate over models/data. different files
        for wdx, w in enumerate(which):

            try:
                file = file_path_shorthand(w)

                data = bnb.hi5.load(file, dset, raise_ex=True)

                zorder = 0
                if "surrogate_" in file:
                    zorder = 2

                kwargs = dict()
                if w == "data":
                    color = _alpha_to_solid_on_bg(period_color, 0.3)
                    kwargs["label"] = f"data {period}"
                else:
                    color = _alpha_to_solid_on_bg(period_color, 1.0)
                    kwargs["lw"] = 1.2
                    kwargs["label"] = f"{w} {period}"

                ref = local_plot(data, color, zorder, **kwargs)
                log.info(f"{w}\t{period}:\t{ref:.2f}")
            except Exception as e:
                log.warning(f"Failed to plot {file} {dset}")
                raise (e)

    ax.set_xlim(-5, 150)
    ax.set_yscale("log")
    if "2_3" in periods or "6_3" in periods:
        ax.set_ylim(1e-4, 1e-1)
        # ax.set_ylim(1e-3, 1)
    else:
        ax.set_ylim(1e-6, 1)

    _fix_log_ticks(ax.yaxis, every=1, hide_label_condition=lambda idx: idx % 2 == 1)
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_minor_locator(MultipleLocator(10))

    if "wbl" in w:
        title = f"Weibull renewal"
    elif "psn" in w:
        title = f"Inh. Poisson"
    elif "tlrd" in w:
        title = f"Tailored Weibull"
    else:
        title = "custom"

    if control is not None:
        title += f" {control}"

    if annotate:
        if show_xlabel:
            ax.set_xlabel(r"Pot. inf. encounters")
        if show_ylabel:
            ax.set_ylabel(r"Distribution")
        if show_title:
            ax.set_title(title, fontsize=8)
        if show_legend:
            ax.legend()
        if show_legend_in_extra_panel:
            _legend_into_new_axes(ax)

    if set_size:
        fig.tight_layout()
        _set_size(ax, 3.3 * cm, 1.7 * cm)

    return ax


# SM
@warntry
def all_dist_infectious_encounters(h5f):
    for p in ["fast", "slow"]:
        for k in ["k_inf", "k_1.0", "k_10.0"]:
            try:
                plot_disease_dist_infectious_encounters(h5f, k=k, periods=p)
            except Exception as e:
                log.info(f"failed for {k} {p}")
                log.info(e)


# SM
@warntry
def plot_controls_distributions_infectious_encounters(
    h5f,
    which="gamma_6_3",
    k_sel="k_10.0",
    inset=True,
):

    fig, ax = plt.subplots()

    if "delta" in which:
        data = h5f["disease"][which]
        log.info("ignoring `k_sel` because delta disease")
    else:
        assert f"disease/{which}/{k_sel}" in h5f.keypaths()
        data = h5f[f"disease/{which}/{k_sel}"]

    ax.set_xlim(-5, 100)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1)
    _fix_log_ticks(ax.yaxis, every=1)
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.margins(y=0, x=5)

    if inset:
        # inset with box-like plot of the means
        # axins = ax.inset_axes([0.27, 0.4, 0.6, 0.4])
        axins = inset_axes(ax, width="50%", height="25%", loc=1, borderpad=1)

        axins.xaxis.set_visible(False)
        # axins.set_ylim(15, 30)
        # axins.spines["bottom"].set_visible(False)

    for cdx, control in enumerate(
        [
            "data",
            "onset_wtrain_wtime",
            "onset_wtrain",
            "onset_wtime",
            "onset",
        ]
    ):
        log.info(f"{control}...")

        if control == "data":
            ctrl_data = data
        else:
            ctrl_data = data[f"control_random_disease_{control}"]

        ctrl_dist = ctrl_data["distribution_infectious_encounter"]
        ctrl_mean = ctrl_data["mean_number_infectious_encounter"]

        num_encounter = ctrl_dist[0, :]
        p_full = ctrl_dist[1, :]
        p_jack = ctrl_dist[2, :]
        p_errs = ctrl_dist[3, :]

        try:
            mean = ctrl_mean[0][0]
            errs = ctrl_mean[2][0]
            try:
                log.info(
                    f"`{control}` has mean: {mean:.1f} and ev:"
                    f" {_ev(num_encounter, p_full):.1f}"
                )
                log.info(f"`{control}` errors: {errs}")
            except:
                pass
        except:
            # dont exist for "data"
            mean = _ev(num_encounter, p_full)
            errs = np.nan

        ax.plot(
            num_encounter, p_full, zorder=3 if control == "data" else 2, label=control
        )
        # log.info([cdx], [mean], [errs])

        if inset:
            axins.errorbar(
                x=[cdx],
                y=[mean],
                yerr=[errs],
                fmt=".",
                markersize=2,
                elinewidth=0.5,
                capsize=5,
                zorder=0,
            )
            axins.set_ylim(10, 30)

    fig.tight_layout()

    if show_xlabel:
        ax.set_xlabel("Pot. inf. encounters ei")
    if show_ylabel:
        ax.set_ylabel("Probability P(ei)")
    if show_title:
        ax.set_title(f"{which} {k_sel}", fontsize=8)
    if show_legend:
        ax.legend(loc="lower left")
    if show_legend_in_extra_panel:
        _legend_into_new_axes(ax)

    # _set_size(ax, 6.5 * cm, 4.5 * cm)
    _set_size(ax, 3.5 * cm, 2.5 * cm)

    return ax


# SM
# we decided to put the mean value joint in their own panel, reusing a lot from above
def plot_controls_means_infectious_encounters(
    h5f, which_list=["gamma_6_3", "gamma_2_3"], k_sel="k_10.0"
):
    fig, ax = plt.subplots()

    for cdx, control in enumerate(
        [
            "data",
            "onset_wtrain_wtime",
            "onset_wtrain",
            "onset_wtime",
            "onset",
        ]
    ):
        log.info(f"{control}...")

        for wdx, which in enumerate(which_list):

            if "delta" in which:
                data = h5f["disease"][which]
                log.info("ignoring `k_sel` because delta disease")
            else:
                assert f"disease/{which}/{k_sel}" in h5f.keypaths()
                data = h5f[f"disease/{which}/{k_sel}"]

            if control == "data":
                ctrl_data = data
            else:
                ctrl_data = data[f"control_random_disease_{control}"]

            ctrl_mean = ctrl_data["mean_number_infectious_encounter"]

            try:
                mean = ctrl_mean[0][0]
                errs = ctrl_mean[2][0]
                try:
                    log.info(
                        f"`{control}` has mean: {mean:.1f} and ev:"
                        f" {_ev(num_encounter, p_full):.1f}"
                    )
                    log.info(f"`{control}` errors: {errs}")
                except:
                    pass
            except:
                # dont exist for "data"
                mean = _ev(num_encounter, p_full)
                errs = np.nan

            clr = f"C{cdx}"
            if wdx == 1:
                clr = _alpha_to_solid_on_bg(clr, 0.3)

            # we needed another ordering but dont want to change colors.
            idx = [
                "data",
                "onset",
                "onset_wtrain_wtime",
                "onset_wtime",
                "onset_wtrain",
            ].index(control)

            ax.errorbar(
                x=[(len(which_list) + 1) * idx + wdx],
                y=[mean],
                yerr=[errs],
                fmt=".",
                color=clr,
                markersize=2,
                elinewidth=0.5,
                capsize=5,
                zorder=0,
                label=f"{which} {control}",
            )

    ax.set_ylim(10, 30)
    ax.xaxis.set_visible(False)
    if show_ylabel:
        ax.set_ylabel("Pot. inf. encounters ei")
    if show_legend:
        ax.legend(loc="lower left")
    if show_legend_in_extra_panel:
        _legend_into_new_axes(ax)

    _set_size(ax, 3.5 * cm, 2.5 * cm)

    return ax


# Fig 5
# needs different h5f, not part of main file yet. usually in 'out_mf'
@warntry
def plot_case_numbers(
    h5f=None,
    which_latent=["latent_1.00", "latent_2.00", "latent_6.00"],
    average_over_rep=True,
    apply_formatting=True,
):
    fig, ax = plt.subplots()

    if h5f is None:
        h5f = bnb.hi5.recursive_load(
            "./out/sample_continuous_branching_Copenhagen_filtered_15min.h5",
            dtype=bdict,
            keepdim=True,
        )

    def plot_cases(cases, color, **kwargs):
        num_rep = len(cases.keys())
        k0 = list(cases.keys())[0]
        x = cases[k0][0, :] / 60 / 60 / 24 / 7
        y_all = np.zeros((len(x), num_rep))
        for idx, key in enumerate(cases.keys()):
            y_all[:, idx] = cases[key][1, :]
        y_all[y_all == 0] = np.nan

        if not average_over_rep:
            selected_rep = 3
            y_mean = y_all[:, selected_rep]
            y_err = np.ones(len(y_mean)) * np.nan
        else:
            y_mean = np.nanmean(y_all, axis=-1)
            y_err = np.nanstd(y_all, axis=-1) / np.sqrt(num_rep)

        ax.plot(x, y_mean, color=color, **kwargs)

    for wdx, w in enumerate(which_latent):
        log.info(w)
        real = h5f["measurements"]["cases"][w]
        surr = h5f["measurements_randomized_per_train"]["cases"][w]
        base_color = f"C{wdx}"
        if "1.00" in w:
            base_color = "#868686"
        elif "2.00" in w:
            base_color = clrs["n_low"]
        elif "6.00" in w:
            base_color = clrs["n_high"]
        plot_cases(surr, color=_alpha_to_solid_on_bg(base_color, 0.3), label=f"surr {w}")
        plot_cases(real, color=base_color, label=f"real {w}")

    if apply_formatting:
        # ax.axhline(1e4, 0, 1, color="gray", ls="--")
        # ax.set_ylim(1, 1.1e6)
        # ax.set_xlim(0, 14.75)
        ax.xaxis.set_major_locator(MultipleLocator(4))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        # for label in ax.xaxis.get_ticklabels()[::2]:
        #     label.set_visible(False)
        ax.set_yscale("log")
        _fix_log_ticks(ax.yaxis, hide_label_condition=lambda idx: idx % 2 == 1)

    if show_ylabel:
        ax.set_ylabel("Daily new cases")
    if show_xlabel:
        ax.set_xlabel("Time (weeks)")
    if show_legend:
        ax.legend(loc="lower left")
    if show_legend_in_extra_panel:
        _legend_into_new_axes(ax)

    if apply_formatting:
        fig.tight_layout()
        _set_size(ax, 3.0 * cm, 2.3 * cm)

    return ax


# essentially, this is a carbon copy of `plot_r4`
def plot_growth_rate():
    fig, ax = plt.subplots()

    rand = np.loadtxt(
        "./out/analysis_continuous_branching_measurements_randomized_per_train.dat",
        unpack=True,
    )
    data = np.loadtxt(
        "./out/analysis_continuous_branching_measurements.dat",
        unpack=True,
    )
    # time_x = rand[0, :]
    time_x = data[0, :]
    data_y = data[1, :]
    rand_y = rand[1, :]

    # convert to 1/days
    rand_y[:] *= 60 * 60 * 24
    data_y[:] *= 60 * 60 * 24

    # Analytic solution
    # Johannes derived a closed formula for R(lambda, t_lat) but its not
    # possible to solve for lambda(R, t_lat) so we do it numerically.
    from scipy.optimize import minimize

    t_ift = 3

    def func(lam):
        return lam * t_ift * np.exp(lam * t_lat) / (1.0 - np.exp(-lam * t_ift))

    def delta(lam, R):
        yt = func(lam)
        return (yt - R) ** 2

    # We have an estimate of R from our measured eift * pift: 25 * 0.12 ~= 3
    # as we do in `plot_disease_mean_number_of_infectious_encounter_cutplane`
    target_R = 2.9483780
    first_guess = 0.5
    lam_res = np.ones(len(time_x)) * np.nan
    for idx, t_lat in enumerate(time_x):
        res = minimize(
            delta, first_guess, args=(target_R), method="Nelder-Mead", tol=1e-6
        )
        lam_res[idx] = res.x[0]

    ax.plot(time_x, lam_res, lw=1, label="_analytic_solution", color=clrs["data_rand"])

    ax.errorbar(
        time_x[:],
        rand_y[:],
        yerr=rand[2, :],
        label="rand",
        color=clrs["data_rand"],
        fmt="o",
        markersize=ms_default,
        alpha=1,
        elinewidth=0.5,
        capsize=1,
        clip_on=False,
    )
    ax.errorbar(
        time_x[:],
        data_y[:],
        yerr=data[2, :],
        label="data",
        color=clrs["data"],
        fmt="o",
        markersize=ms_default,
        alpha=1,
        elinewidth=0.5,
        capsize=1,
        clip_on=False,
    )

    # shaded regions
    # for the crossings of colors that do not fall on sampled data points, we interpolate
    # and manually ad the points
    ins_x = 0.82
    ins_y = 0.5124
    idx = np.argwhere(time_x[:] > ins_x)[0][0]
    time_x = np.insert(time_x, idx, ins_x)
    data_y = np.insert(data_y, idx, ins_y)
    rand_y = np.insert(rand_y, idx, ins_y)

    idx = np.where((time_x[:] >= 0) & (time_x[:] <= ins_x))[0]
    ax.fill_between(
        time_x[idx],
        y1=rand_y[idx],
        y2=data_y[idx],
        color="#faad7c",
        alpha=0.4,
        lw=0,
        zorder=1,
    )
    idx = np.where((time_x[:] >= ins_x) & (time_x[:] <= 4))[0]
    ax.fill_between(
        time_x[idx],
        y1=rand_y[idx],
        y2=data_y[idx],
        color="#49737a",
        alpha=0.4,
        lw=0,
        zorder=1,
    )

    # insert another point
    ins_x = 7.32
    ins_y = 0.123
    idx = np.argwhere(time_x[:] > ins_x)[0][0]
    time_x = np.insert(time_x, idx, ins_x)
    data_y = np.insert(data_y, idx, ins_y)
    rand_y = np.insert(rand_y, idx, ins_y)

    idx = np.where((time_x[:] >= 4) & (time_x[:] <= ins_x))[0]
    ax.fill_between(
        time_x[idx],
        y1=rand_y[idx],
        y2=data_y[idx],
        color="#faad7c",
        alpha=0.4,
        lw=0,
        zorder=1,
    )

    idx = np.where((time_x[:] >= ins_x) & (time_x[:] <= 8))[0]
    ax.fill_between(
        time_x[idx],
        y1=rand_y[idx],
        y2=data_y[idx],
        color="#49737a",
        alpha=0.4,
        lw=0,
        zorder=1,
    )

    # ax.legend()
    ax.set_yscale("log")
    ax.set_ylim(1e-1, 2.0e-0)
    _fix_log_ticks(ax.yaxis, every=1)
    ax.set_xlim(0, 8)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    # ax.yaxis.set_major_locator(MultipleLocator(2))
    # ax.yaxis.set_minor_locator(MultipleLocator(1))

    sns.despine(ax=ax, right=True, top=True, trim=False, offset=3)

    if show_ylabel:
        ax.set_ylabel(r"Spreading rate $\lambda$\\n(1 / day)")
    if show_xlabel:
        ax.set_xlabel("Latent period (days)")
    if show_legend:
        ax.legend(loc="lower left")
    if show_legend_in_extra_panel:
        _legend_into_new_axes(ax)

    fig.tight_layout()
    _set_size(ax, 5.0 * cm, 2.8 * cm)

    return ax


# Fig 5
@warntry
def plot_r4():
    fig, ax = plt.subplots()

    rand = np.loadtxt(
        "./out_mf/analysis_mean-field_R_measurements_randomized_per_train.dat",
        unpack=True,
    )
    data = np.loadtxt(
        "./out_mf/analysis_mean-field_R_measurements.dat",
        unpack=True,
    )

    ax.errorbar(
        rand[0, :],
        rand[1, :],
        yerr=rand[2, :],
        label="r4 rand",
        color=clrs.data_randomized,
        fmt="o",
        markersize=ms_default,
        alpha=1,
        elinewidth=0.5,
        capsize=1,
    )
    ax.errorbar(
        data[0, :],
        data[1, :],
        yerr=data[2, :],
        label="r4 data",
        color=clrs["data"],
        fmt="o",
        markersize=ms_default,
        alpha=1,
        elinewidth=0.5,
        capsize=1,
    )

    # shaded regions
    idx = np.where((rand[0, :] >= 1) & (rand[0, :] <= 4))[0]
    ax.fill_between(
        rand[0, idx],
        y1=rand[1, idx],
        y2=data[1, idx],
        color="#49737a",
        alpha=0.4,
        lw=0,
        zorder=1,
    )

    idx = np.where((rand[0, :] >= 4) & (rand[0, :] <= 7))[0]
    ax.fill_between(
        rand[0, idx],
        y1=rand[1, idx],
        y2=data[1, idx],
        color="#faad7c",
        alpha=0.4,
        lw=0,
        zorder=1,
    )

    # ax.legend()
    ax.set_ylim(1.5, 4)
    ax.set_xlim(0, 8)
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_minor_locator(MultipleLocator(1))

    if show_ylabel:
        ax.set_ylabel("Rep. num. R")
    if show_xlabel:
        ax.set_xlabel("Latent period")
    if show_legend:
        ax.legend(loc="lower left")
    if show_legend_in_extra_panel:
        _legend_into_new_axes(ax)

    fig.tight_layout()
    _set_size(ax, 3.0 * cm, 1.375 * cm)

    return ax


# Fig 5
@warntry
def plot_r0(h5f):

    fig, ax = plt.subplots()

    # copy from figure 2e
    target_inf = 3.0  # days
    data = h5f["disease/delta/scan_mean_number_infectious_encounter"]
    range_inf = data["range_infectious"][:]
    range_lat = data["range_latent"][:]

    rdx = np.where(range_inf == target_inf)[0][0]
    data_2d = data["mean"][:]
    data_1d = data_2d[rdx, :] * 0.12
    ax.plot(
        range_lat,
        data_1d,
        color=clrs["data"],
        ls="-",
    )

    norm = data["mean_relative_to_poisson"][:][rdx, :]
    data_1d_norm = data_1d / norm
    ax.plot(range_lat, data_1d_norm, color=clrs.data_randomized, ls="-")

    # shaded regions
    idx = np.where((range_lat >= 1) & (range_lat <= 4))[0]
    ax.fill_between(
        range_lat[idx],
        y1=data_1d[idx],
        y2=data_1d_norm[idx],
        color="#49737a",
        alpha=0.4,
        lw=0,
        zorder=1,
    )

    idx = np.where((range_lat >= 4) & (range_lat <= 7))[0]
    ax.fill_between(
        range_lat[idx],
        y1=data_1d_norm[idx],
        y2=data_1d[idx],
        color="#faad7c",
        alpha=0.4,
        lw=0,
        zorder=1,
    )

    data = np.loadtxt(
        "./out_mf/analysis_mean-field_R_measurements.dat",
        unpack=True,
    )
    rand = np.loadtxt(
        "./out_mf/analysis_mean-field_R_measurements_randomized_per_train.dat",
        unpack=True,
    )

    ax.errorbar(
        rand[0, :],
        rand[5, :],
        yerr=rand[6, :],
        label="r0 rand",
        color=clrs.data_randomized,
        fmt="o",
        markersize=ms_default,
        alpha=1,
        elinewidth=0.5,
        capsize=1,
        zorder=2,
    )
    ax.errorbar(
        data[0, :],
        data[5, :],
        yerr=data[6, :],
        label="r0 data",
        color=clrs["data"],
        fmt="o",
        markersize=ms_default,
        alpha=1,
        elinewidth=0.5,
        capsize=1,
        zorder=2,
    )

    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.set_xlim(0, 8)
    ax.set_ylim(2, 4)

    if show_ylabel:
        ax.set_ylabel("Rep. num. R")
    if show_xlabel:
        ax.set_xlabel("Latent period")
    if show_legend:
        ax.legend(loc="lower left")
    if show_legend_in_extra_panel:
        _legend_into_new_axes(ax)

    fig.tight_layout()
    _set_size(ax, 3.0 * cm, 1.1 * cm)

    return ax


# ------------------------------------------------------------------------------ #
# style helpers
# ------------------------------------------------------------------------------ #


def save_ax(ax, fname, ensure_dpi_hack=True):

    if ensure_dpi_hack:
        # make sure the right dpi gets embedded so graphics programs will recognice it.
        # seems like this needs a tiny patch that is rasterized
        ax.set_rasterization_zorder(-50)
        x = ax.get_xlim()[0]
        x = [x, x]
        y = ax.get_ylim()
        ax.plot(x, y, lw=0.001, color="white", zorder=-51)

    ax.get_figure().savefig(fname, dpi=300, transparent=True)

    if figures_only_to_disk:
        plt.close(ax.get_figure())


def save_all_figures(path, fmt="pdf", save_pickle=False, **kwargs):
    """
    saves all open figures as pdfs and pickle. to load an existing figure:
    ```
    import pickle
    with open('/path/to/fig.pkl','rb') as fid:
        fig = pickle.load(fid)
    ```
    """
    import os

    path = os.path.expanduser(path)
    assert os.path.isdir(path)

    try:
        import pickle
    except ImportError:
        if pickle:
            log.info("Failed to import pickle")
            save_pickle = False

    try:
        from tqdm import tqdm
    except ImportError:

        def tqdm(*args):
            return iter(*args)

    if "dpi" not in kwargs:
        kwargs["dpi"] = 300
    if "transparent" not in kwargs:
        kwargs["transparent"] = True

    for i in tqdm(plt.get_fignums()):
        fig = plt.figure(i)
        fig.savefig(f"{path}/figure_{i}.{fmt}", **kwargs)
        if save_pickle:
            try:
                os.makedirs(f"{path}/pickle/", exist_ok=True)
                with open(f"{path}/pickle/figure_{i}.pkl", "wb") as fid:
                    pickle.dump(fig, fid)
            except Exception as e:
                print(e)


def _alpha_to_solid_on_bg(base, alpha, bg="white"):
    """
    Probide a color to start from `base`, and give it opacity `alpha` on
    the background color `bg`
    """

    def rgba_to_rgb(c, bg):
        bg = matplotlib.colors.to_rgb(bg)
        alpha = c[-1]

        res = (
            (1 - alpha) * bg[0] + alpha * c[0],
            (1 - alpha) * bg[1] + alpha * c[1],
            (1 - alpha) * bg[2] + alpha * c[2],
        )
        return res

    new_base = list(matplotlib.colors.to_rgba(base))
    new_base[3] = alpha
    return matplotlib.colors.to_hex(rgba_to_rgb(new_base, bg))


def _fix_log_ticks(ax_el, every=1, hide_label_condition=lambda idx: False):
    """
    # Parameters
    ax_el: usually `ax.yaxis`
    every: 1 or 2
    hide_label_condition : function e.g. `lambda idx: idx % 2 == 0`
    """
    ax_el.set_major_locator(LogLocator(base=10, numticks=10))
    ax_el.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(0, 1.05, every / 10), numticks=10)
    )
    ax_el.set_minor_formatter(matplotlib.ticker.NullFormatter())
    for idx, lab in enumerate(ax_el.get_ticklabels()):
        if hide_label_condition(idx):
            lab.set_visible(False)


def _pretty_log_ticks(ax_el, prec=2):
    """
    Example
    ```
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(_pretty_log_formatter()))
    ```

    """
    # https://stackoverflow.com/questions/21920233/matplotlib-log-scale-tick-label-number-formatting

    def myLogFormat(y, pos, prec=prec):
        if y > np.power(10.0, prec) or y < np.power(10.0, -prec):
            return r"$10^{{{:d}}}$".format(int(np.log10(y)))

        else:
            # Find the number of decimal places required
            decimalplaces = int(np.maximum(-np.log10(y), 0))
            # Insert that number into a format string
            formatstring = "{{:.{:1d}f}}".format(decimalplaces)
            # Return the formatted tick label
        return formatstring.format(y)

    ax_el.set_major_formatter(matplotlib.ticker.FuncFormatter(myLogFormat))


def _legend_into_new_axes(ax):
    fig, ax_leg = plt.subplots(figsize=(6 * cm, 6 * cm))
    h, l = ax.get_legend_handles_labels()
    ax_leg.axis("off")
    ax_leg.legend(h, l, loc="upper left")


def _detick(axis, keep_labels=False, keep_ticks=False):
    """
    ```
    _detick(ax.xaxis)
    _detick([ax.xaxis, ax.yaxis])
    ```
    """
    if not isinstance(axis, list):
        axis = [axis]
    for a in axis:
        if not keep_labels and not keep_ticks:
            a.set_ticks_position("none")
            a.set_ticks([])
        elif not keep_labels and keep_ticks:
            a.set_ticklabels([])


# def _set_size(ax, w, h):
#     """w, h: width, height in inches"""
#     # https://stackoverflow.com/questions/44970010/axes-class-set-explicitly-size-width-height-of-axes-in-given-units
#     if not use_compact_size:
#         return
#     l = ax.figure.subplotpars.left
#     r = ax.figure.subplotpars.right
#     t = ax.figure.subplotpars.top
#     b = ax.figure.subplotpars.bottom
#     figw = float(w) / (r - l)
#     figh = float(h) / (t - b)
#     ax.figure.set_size_inches(figw, figh)


def _set_size(ax, w, h):
    """w, h: width, height in inches"""
    # https://newbedev.com/axes-class-set-explicitly-size-width-height-of-axes-in-given-units
    from mpl_toolkits.axes_grid1 import Divider, Size

    axew = w
    axeh = h

    # lets use the tight layout function to get a good padding size for our axes labels.
    # fig = plt.gcf()
    # ax = plt.gca()
    fig = ax.get_figure()
    # fig.tight_layout()
    # obtain the current ratio values for padding and fix size
    oldw, oldh = fig.get_size_inches()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom

    # work out what the new  ratio values for padding are, and the new fig size.
    # ps: adding a bit to neww and newh gives more padding
    # the axis size is set from axew and axeh
    neww = axew + oldw * (1 - r + l) + 0.1
    newh = axeh + oldh * (1 - t + b) + 0.1
    newr = r * oldw / neww - 0.1
    newl = l * oldw / neww + 0.1
    newt = t * oldh / newh - 0.1
    newb = b * oldh / newh + 0.1

    # right(top) padding, fixed axes size, left(bottom) pading
    hori = [Size.Scaled(newr), Size.Fixed(axew), Size.Scaled(newl)]
    vert = [Size.Scaled(newt), Size.Fixed(axeh), Size.Scaled(newb)]

    divider = Divider(fig, (0.0, 0.0, 1.0, 1.0), hori, vert, aspect=False)
    # the width and height of the rectangle is ignored.

    ax.set_axes_locator(divider.new_locator(nx=1, ny=1))

    # we need to resize the figure now, as we have may have made our axes bigger than in.
    fig.set_size_inches(neww, newh)


# ------------------------------------------------------------------------------ #
# tech helpers
# ------------------------------------------------------------------------------ #


def _ev(x, pdist):
    """
    expectation value via x, p(x)
    """
    ev = 0
    for idx, p in enumerate(pdist):
        ev += x[idx] * pdist[idx]
    return ev


def _stat_measures(x, pdist):
    """
    get mean, variance and skewness etc.
    think moments.

    returns a dict: 1->mean, 2->variance, ...
    """
    mean = _ev(x, pdist)
    variance = _ev(np.power(x - mean, 2.0), pdist)
    skew = _ev(np.power(x - mean, 3.0), pdist) / np.power(variance, 3.0 / 2.0)
    kurtosis = _ev(np.power(x - mean, 4.0), pdist) / np.power(variance, 2.0)

    return {1: mean, 2: variance, 3: skew, 4: kurtosis}


def _poisson_limit(h5f, target_inf):
    data = h5f["disease/delta/scan_mean_number_infectious_encounter"]
    range_inf = data["range_infectious"][:]
    rdx = np.where(range_inf == target_inf)[0][0]
    norm = data["mean_relative_to_poisson"][:][rdx, :]
    res = data["mean"][:][rdx, :]
    return res / norm


def _fit_weibull_to_data(dist):

    dat = dist["lin"]
    # avoid x -> 0, weibull diverges and fits will have trouble.
    x = dat[0][1:]
    y = dat[1][1:]
    fitfunc = _weibull_dt_mean_with_amplitude

    # try a bunch of start values for the fits
    p0_k = np.logspace(np.log10(1e-4), np.log10(1e4), base=10, num=10)
    p0_mean = np.logspace(np.log10(1e-4), np.log10(1e4), base=10, num=20)
    p0_amp = [1]
    fitpars = np.array(list(product(p0_k, p0_mean, p0_amp)))

    # we have some insight into plausible starting values
    bounds = np.array(
        [
            [0, np.inf],  # k
            [0, 60 * 60 * 24],  # mean dt (in seconds)
            [0, np.inf],  # amplitude
        ]
    ).T

    # consider error estimates in the fit, but the sqrt(N) argument is hard to evaluate
    # when we already have probability density
    # idx = np.where(y > 0)[0]
    # x = x[idx]
    # y = y[idx]
    # err = np.sqrt(y)

    # do a fit for every set of starting points and return the ones that
    # have lowest sum of residuals
    def fitloop():
        ssresmin = np.inf
        fulpopt = None
        fulpcov = None

        # silence numpy a bit
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for idx, pars in enumerate(tqdm(fitpars, desc="Fits")):
                try:
                    popt, pcov = curve_fit(
                        fitfunc,
                        xdata=x,
                        ydata=y,
                        p0=pars,
                        bounds=bounds,
                        # sigma = err
                    )

                    residuals = y - fitfunc(x, *popt)
                    ssres = np.sum(residuals**2)

                except Exception as e:
                    ssres = np.inf
                    popt = None
                    pcov = None
                    # log.debug('Fit %d did not converge. Ignoring this fit', idx+1)

                if ssres < ssresmin:
                    ssresmin = ssres
                    fulpopt = popt
                    fulpcov = pcov

        return fulpopt, fulpcov, ssresmin

    fulpopt, fulpcov, ssresmin = fitloop()

    log.info(f"popt: {fulpopt} with ssres: {ssresmin:.2e}")

    # compare with dt direclty from data. do this on lin scale!
    # beware, this has a finite-sample bias
    ev_x = np.sum(x * y) * (x[1] - x[0]) / 2
    log.info(f"<dt> = {ev_x}")

    def func_with_popt(x):
        return fitfunc(x, *fulpopt)

    return func_with_popt


def _weibull(x, k, l):
    return k / l * np.power(x / l, k - 1) * np.exp(-np.power(x / l, k))


def _weibull_dt_mean_with_amplitude(x, k, mean, amp=1.0):
    l = mean / gamma(1 + 1 / k)
    # return amp*_weibull(x, k, l)
    # the two are equivalent, manual implementation is not well defined for x<=0)
    return amp * scipy.stats.exponweib.pdf(x, l, k)


if __name__ == "__main__":
    pass
