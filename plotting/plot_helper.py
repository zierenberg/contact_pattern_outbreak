# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2021-02-09 18:58:52
# @Last Modified: 2021-11-04 11:10:58
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
# h5f = h5.recursive_load(
#    "./out/results_Copenhagen_filtered_15min.h5", dtype=bdict, keepdim=True
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

# fmt: off
import os
import sys
import glob
import argparse
import inspect
import logging
import h5py as h5
import numpy as np
import pandas as pd
import scipy.stats
from scipy.special import gamma
from scipy.optimize import curve_fit
from itertools import product
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
matplotlib.rcParams["xtick.labelsize"]=8
matplotlib.rcParams["ytick.labelsize"]=8
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
from matplotlib.ticker import MultipleLocator, LogLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

log = logging.getLogger(__name__)
import seaborn as sns
import palettable as pt

import functools
from benedict import benedict
# we want "/" as keypath separator instead of ".", set it as default argument
bdict = functools.partial(benedict, keypath_separator="/")
# now we can call bdict() to get a dict that can be accessed as foo["some/level/bar"]

from addict import Dict
# these guys can be accessed like foo.some.level.bar

import h5_helper as h5

# enable code formatting with black
# fmt: on

clrs = Dict()
clrs.n_high = "#C31B2B"  # "#F83546"
clrs.n_low = "#5295C8"
clrs.n_psn = "#E7E7B6"

clrs.weekday = "#143163"
clrs.weekend = "#2E72A8"
clrs.medium = "#46718C"
clrs.weekday_psn = "#E3B63F"
clrs.weekend_psn = "#F1CD79"
clrs.medium_psn = "#F3C755"

clrs.weibull = "#ea5e48"
clrs.weibull_weekday = "#C6543A"
clrs.weibull_weekend = "#FA8267"

clrs.viral_load = "#1e7d72"
clrs.activity = "#6EB517"

clrs.cond_enc_rate = "#233954"
clrs.disease_psn_norm = "#e3e3ad"

clrs.dispersion = Dict()
clrs.dispersion.k_vals = [1, 10, 100, 1e8]
for idx, k in enumerate(clrs.dispersion.k_vals):
    # clrs.dispersion[k] = pt.cartocolors.sequential.BrwnYl_4.mpl_colors[idx]
    # clrs.dispersion[k] = pt.cartocolors.sequential.Emrld_4.mpl_colors[idx]
    # clrs.dispersion[k] = pt.cmocean.sequential.Oxy_4_r.mpl_colors[idx]
    # clrs.dispersion[k] = pt.scientific.sequential.Bamako_4_r.mpl_colors[idx]
    clrs.dispersion[k] = pt.scientific.sequential.Tokyo_4_r.mpl_colors[idx]
    clrs.dispersion[1] = "#E0E3B2"  # "#E8E8C0"

# custom color maps
clrs.palettes = Dict()
clrs.palettes["pastel_1"] = [
    (0, "#E7E7B6"),
    (0.25, "#ffad7e"),
    (0.5, "#cd6772"),
    (0.75, "#195571"),
    (1, "#011A39"),
]
clrs.palettes["div_pastel_1"] = [
    (0, "#C31B2B"),
    (0.25, "#ffad7e"),
    (0.5, "#E7E7B6"),
    (0.85, "#195571"),
    (1, "#011A39"),
]
clrs.ccmap = Dict()
for key in clrs.palettes.keys():
    clrs.ccmap[key] = LinearSegmentedColormap.from_list(key, clrs.palettes[key], N=512)

# cm to inch
cm = 0.3937

# select things to draw for every panel for every panel
show_title = True
show_xlabel = True
show_ylabel = True
show_legend = False
show_legend_in_extra_panel = True
use_compact_size = True  # this recreates the small panel size of the manuscript

# default marker size
ms_default = 2


def main_manuscript():
    h5f = h5.recursive_load(
        "./out/results_Copenhagen_filtered_15min.h5", dtype=bdict, keepdim=True
    )
    figure_1(h5f)
    figure_2(h5f)
    figure_3(h5f)
    figure_4(h5f)
    figure_5(h5f)


# ------------------------------------------------------------------------------ #
# macro functions to create figures
# ------------------------------------------------------------------------------ #


def figure_1(h5f):
    log.info("Figure 1")
    plot_etrain_rasters(h5f)
    plot_etrain_rate(h5f)
    plot_dist_inter_encounter_interval(h5f)


def figure_2(h5f):
    log.info("Figure 2")
    plot_dist_encounters_per_train(h5f)
    plot_dist_encounters_per_day(h5f)
    plot_etrain_raster_example(h5f)
    plot_conditional_rate(h5f, which=["data"])
    plot_disease_mean_number_of_infectious_encounter_cutplane(
        h5f, ax=None, how="absolute", t_inf=3
    )
    plot_disease_mean_number_of_infectious_encounter_2d(
        h5f, which="data", how="relative", control_plot=False
    )


def figure_3(h5f):
    log.info("Figure 3")
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


def figure_4(h5f):
    log.info("Figure 4")
    plot_disease_viral_load_examples()
    plot_gamma_distribution()
    plot_disease_dist_infectious_encounters(h5f, k="k_inf", periods="slow")
    plot_disease_dist_infectious_encounters(h5f, k="k_10.0", periods="slow")
    plot_dispersion_scan_k(h5f, periods="slow")

    compare_disease_dist_infectious_encounters_to_psn(h5f, periods="slow")


# this needs a different input file than the others
# todo, paths are still hardcoded
def figure_5(h5f):
    # this guy loads other files from disk, too
    plot_r0(h5f)

    # this guy loads only other files from disk
    plot_r4()

    # this guy needs a completely different file
    r_h5f = h5.recursive_load(
        "./out_mf/mean_field_samples_Copenhagen_filtered_15min.h5",
        dtype=bdict,
        keepdim=True,
    )
    plot_r(r_h5f)


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
    load = functools.partial(h5.recursive_load, dtype=bdict, keepdim=True, skip=["trains"])

    plot_2d = functools.partial(plot_disease_mean_number_of_infectious_encounter_2d,
        which="data", how=how, control_plot=False)

    h5ref = load("./out/results_Copenhagen_filtered_15min.h5",)

    fig_kws = dict(dpi=300, transparent=True)

    # Contact duration
    fig, ax1d = plt.subplots(figsize=(8*cm, 5*cm))

    kwargs = dict(label="15min (main)")
    plot_conditional_rate(h5ref, ax=ax1d, which=["data"], control_plot=True, kwargs_overwrite = kwargs)
    ax = plot_2d(h5ref)
    ax.set_title("15min (main)")
    ax.get_figure().savefig(f"./figs/mins/2d_15min_{how}.pdf", **fig_kws)

    h5f = load("./out_min_sweep/results_Copenhagen_filtered_5min.h5")
    kwargs = dict(label="5min", lw=0.5)
    plot_conditional_rate(h5f, ax=ax1d, which=["data"], control_plot=True, kwargs_overwrite = kwargs)
    ax = plot_2d(h5f)
    ax.set_title("5min")
    ax.get_figure().savefig(f"./figs/mins/2d_5min_{how}.pdf", **fig_kws)

    h5f = load("./out_min_sweep/results_Copenhagen_filtered_10min.h5")
    kwargs = dict(label="10min", lw=0.5)
    plot_conditional_rate(h5f, ax=ax1d, which=["data"], control_plot=True, kwargs_overwrite = kwargs)
    ax = plot_2d(h5f)
    ax.set_title("10min")
    ax.get_figure().savefig(f"./figs/mins/2d_10min_{how}.pdf", **fig_kws)

    h5f = load("./out_min_sweep/results_Copenhagen_filtered_20min.h5")
    kwargs = dict(label="20min", lw=0.5)
    plot_conditional_rate(h5f, ax=ax1d, which=["data"], control_plot=True, kwargs_overwrite = kwargs)
    ax = plot_2d(h5f)
    ax.set_title("20min")
    ax.get_figure().savefig(f"./figs/mins/2d_20min_{how}.pdf", **fig_kws)

    h5f = load("./out_min_sweep/results_Copenhagen_filtered_30min.h5")
    kwargs = dict(label="30min", lw=0.5)
    plot_conditional_rate(h5f, ax=ax1d, which=["data"], control_plot=True, kwargs_overwrite = kwargs)
    ax = plot_2d(h5f)
    ax.set_title("30min")
    ax.get_figure().savefig(f"./figs/mins/2d_30min_{how}.pdf", **fig_kws)

    _set_size(fig.axes[0], 5.0 * cm, 3.5 * cm )
    fig.axes[0].set_ylim(0, 100)
    fig.savefig(f"./figs/mins/cer.pdf", **fig_kws)

    # RSSI conditional encounter rate
    fig, ax = plt.subplots(figsize=(8*cm, 5*cm))

    kwargs = dict(label="-80db (main)")
    plot_conditional_rate(h5ref, ax=ax, which=["data"], control_plot=True, kwargs_overwrite = kwargs)

    h5f = load("./out_rssi75/results_Copenhagen_filtered_15min.h5")
    kwargs = dict(label="-75db")
    plot_conditional_rate(h5f, ax=ax, which=["data"], control_plot=True, kwargs_overwrite = kwargs)


    h5f = load("./out_rssi95/results_Copenhagen_filtered_15min.h5")
    kwargs = dict(label="-95db")
    plot_conditional_rate(h5f, ax=ax, which=["data"], control_plot=True, kwargs_overwrite = kwargs)

    _set_size(fig.axes[0], 5.0 * cm, 3.5 * cm )
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
    _set_size(ax, 5.0 * cm, 3.5 * cm )
    ax.get_figure().savefig(f"./figs/sm_ecr.pdf", **fig_kws)

    ax = plot_dist_inter_encounter_interval(h5f, sm_generative_processes=True)
    _set_size(ax, 5.0 * cm, 3.5 * cm )
    ax.get_figure().savefig(f"./figs/sm_iei.pdf", **fig_kws)



# decorator for lower level plot functions to continue if subplot fails
def warntry(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            log.exception(f"{func.__name__}: {e}")

    return wrapper


# Fig 1b
@warntry
def plot_etrain_rasters(h5f):

    fig, ax = plt.subplots()
    ax.set_rasterization_zorder(0)

    # load the trains
    assert (
        "data/trains" in h5f.keypaths()
    ), "if you want to do the raster plot, do not skip the loading of trains"
    trains = h5f["data/trains"]

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
            color="#46718C",
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
def plot_etrain_rate(h5f, ax=None, sm_generative_processes = False):
    if ax is None:
        with plt.rc_context({"xtick.labelsize": 6, "ytick.labelsize": 6}):
            fig, ax = plt.subplots(figsize=(6.5 * cm, 4.5 * cm))

    else:
        fig = ax.get_figure()
    ax.set_rasterization_zorder(0)

    norm_rate = 1 / 60 / 60 / 24

    data = h5f["data/encounter_train/rate"]
    r_time = data[0, :] / 60 / 60 / 24
    r_full = data[1, :] / norm_rate
    r_jack = data[2, :] / norm_rate
    r_errs = data[3, :] / norm_rate

    try:
        r_psn = h5f["sample/poisson_inhomogeneous/rate"][1, :] / norm_rate
    except Exception as e:
        r_psn = np.ones(len(r_time)) * np.nan
        log.warning("Couldnt find Poisson encounter rate")
    try:
        r_wbl = h5f["sample/weibul_renewal_process/rate"][1, :] / norm_rate
    except Exception as e:
        r_wbl = np.ones(len(r_time)) * np.nan
        log.warning("Couldnt find Weibull encounter rate")

    # error bars
    ax.errorbar(
        x=r_time,
        y=r_full,
        yerr=r_errs,
        fmt="o",
        markersize=ms_default,
        color=clrs.cond_enc_rate,
        ecolor=clrs.cond_enc_rate,
        alpha=1,
        elinewidth=0.5,
        capsize=1,
        zorder=2,
        label="data",
    )

    # inh poisson, as continuous line. default, main manuscript
    ax.plot(
        r_time,
        r_psn,
        color=clrs.medium_psn,
        ls="--",
        alpha=1,
        zorder=0,
        label="inh. Poisson",
    )

    # for the sm, we want a more complete picture and different style.
    if sm_generative_processes:
        ax.clear()
        r_psn_errs = h5f["sample/poisson_inhomogeneous/rate"][3, :] / norm_rate
        r_wbl_errs = h5f["sample/weibul_renewal_process/rate"][3, :] / norm_rate

        # data
        ax.plot(r_time, r_full, color=clrs.cond_enc_rate, ls="--", lw=1, label="data",
            zorder=2)

        # inh_poisson
        ax.errorbar(
            x=r_time,
            y=r_psn,
            yerr=r_psn_errs,
            fmt="s",
            ls="-",
            markersize=ms_default,
            color=clrs.medium_psn,
            ecolor=clrs.medium_psn,
            mfc="white",
            alpha=1,
            # lw=0.5,
            elinewidth=0.5,
            capsize=0,
            zorder=0,
            label="inh. Poisson process",
        )

        # weibull renewal
        ax.errorbar(
            x=r_time,
            y=r_wbl,
            yerr=r_wbl_errs,
            fmt="o",
            ls="-",
            markersize=ms_default,
            color=clrs.weibull,
            ecolor=clrs.weibull,
            alpha=1,
            # ls="--",
            # lw=0.5,
            elinewidth=0.5,
            capsize=0,
            zorder=0,
            label="Weibull renewal process",
        )



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

    # weekdys on x axis
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

    fig.tight_layout()
    _set_size(ax, 3.1 * cm, 2.0 * cm)

    return ax


# Fig 1d, SM
@warntry
def plot_dist_inter_encounter_interval(h5f, ax=None, which="log", sm_generative_processes=False):
    with plt.rc_context({"xtick.labelsize": 6, "ytick.labelsize": 6}):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6.5 * cm, 4.5 * cm))
        else:
            fig = ax.get_figure()
        ax.set_rasterization_zorder(0)

    # data is in seconds
    iei_norm = 1 / 60 / 60 / 24

    assert which in ["log", "lin", "both"]

    if which == "log" or which == "both":
        # real data
        dat = h5f["data/encounter_train/distribution_inter_encounter_intervals_logbin"][
            :
        ]
        iei = dat[0, :] * iei_norm
        prob = dat[1, :]
        jack_iei = dat[2, :] * iei_norm
        jack_prob = dat[3, :]
        errs_iei = dat[4, :] * iei_norm
        errs_prob = dat[5, :]

        # surrogate data
        s_dat = h5f[
            "data_surrogate_randomize_per_train/encounter_train/distribution_inter_encounter_intervals_logbin"
        ][:]
        s_iei = s_dat[0, :] * iei_norm
        s_prob = s_dat[1, :]
        s_jack_iei = s_dat[2, :] * iei_norm
        s_jack_prob = s_dat[3, :]
        s_errs_iei = s_dat[4, :] * iei_norm
        s_errs_prob = s_dat[5, :]

        # poisson
        try:
            dat_psn = h5f[
                "sample/poisson_inhomogeneous/distribution_inter_encounter_intervals_logbin"
            ][:]
        except Exception as e:
            log.debug(e)
            dat_psn = np.ones(dat.shape) * np.nan
            dat_psn[0, :] = dat[0, :]
            log.warning("Could not load IEI dist for inhom. Poisson")

        iei_psn = dat_psn[0, :] * iei_norm
        prob_psn = dat_psn[1, :]
        jack_iei_psn = dat_psn[2, :] * iei_norm
        jack_prob_psn = dat_psn[3, :]
        errs_iei_psn = dat_psn[4, :] * iei_norm
        errs_prob_psn = dat_psn[5, :]

        # weibull
        try:
            dat_wbl = h5f[
                "sample/weibul_renewal_process/distribution_inter_encounter_intervals_logbin"
            ][:]
        except:
            dat_wbl = np.ones(dat.shape) * np.nan
            dat_wbl[0, :] = dat[0, :]
            log.warning("Could not load IEI dist for Weibull")

        iei_wbl = dat_wbl[0, :] * iei_norm
        prob_wbl = dat_wbl[1, :]
        jack_iei_wbl = dat_wbl[2, :] * iei_norm
        jack_prob_wbl = dat_wbl[3, :]
        errs_iei_wbl = dat_wbl[4, :] * iei_norm
        errs_prob_wbl = dat_wbl[5, :]

        # ------------------------------------------------------------------------------ #
        # plot
        # ------------------------------------------------------------------------------ #

        # main manuscript
        # data error bars in x and y
        e_step = 1
        ax.errorbar(
            x=iei[::e_step],
            y=prob[::e_step],
            xerr=errs_iei[::e_step],
            yerr=errs_prob[::e_step],
            fmt="o",
            markersize=ms_default,
            color=clrs.cond_enc_rate,
            ecolor=clrs.cond_enc_rate,
            alpha=1,
            elinewidth=1,
            capsize=0,
            zorder=5,
            label="data",
        )

        ax.plot(iei_wbl, prob_wbl, color=clrs.weibull, ls=(0, (1, 1)), label="Weibull")
        # ax.plot(iei_psn, prob_psn, color=clrs.medium_psn, ls=(0, (1, 1)), label="Poisson")

        # for the sm, we want a more complete picture and different style.
        if sm_generative_processes:
            ax.clear()

            # data
            ax.plot(iei, prob, color=clrs.cond_enc_rate, ls="--", lw=1,
                label="data", zorder=2)

            # inh poisson
            ax.errorbar(
                x=iei_psn[::e_step],
                y=prob_psn[::e_step],
                xerr=errs_iei_psn[::e_step],
                yerr=errs_prob_psn[::e_step],
                fmt="s",
                ls="-",
                markersize=ms_default,
                color=clrs.medium_psn,
                ecolor=clrs.medium_psn,
                mfc="white",
                alpha=1,
                elinewidth=0.5,
                capsize=0,
                zorder=0,
                label="inh. Poisson process",
            )

            # weibull
            ax.errorbar(
                x=iei_wbl[::e_step],
                y=prob_wbl[::e_step],
                xerr=errs_iei_wbl[::e_step],
                yerr=errs_prob_wbl[::e_step],
                fmt="o",
                ls="-",
                markersize=ms_default,
                color=clrs.weibull,
                ecolor=clrs.weibull,
                alpha=1,
                elinewidth=0.5,
                capsize=0,
                zorder=0,
                label="Weibull renewal process",
            )


    if which == "lin" or which == "both":
        dat = h5f["data/encounter_train/distribution_inter_encounter_intervals"][:]

        iei = dat[0, :] * iei_norm
        prob = dat[1, :]
        jack_prob = dat[2, :]
        errs_prob = dat[3, :]

        # simple estimate
        ax.plot(
            iei, prob, color=clrs.weekday, alpha=0.2, label="lin simple", zorder=0,
        )

    # annotations
    ax.axvline(1 / 24 / 60 * 5, 0, 1, color="gray", ls=":")  # 5 min
    ax.axvline(1 / 24, 0, 1, color="gray", ls=":")  # hour
    ax.axvline(1, 0, 1, color="gray", ls=":")  # day
    ax.axvline(7, 0, 1, color="gray", ls=":")  # week

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

    # show less ticks in main manuscript
    _fix_log_ticks(
        ax.yaxis, every=2, hide_label_condition=lambda idx: not (idx + 2) % 4 == 0
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
def plot_dist_encounters_per_day(h5f, ax=None):

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    ax.set_rasterization_zorder(0)

    # real data, no differentiation of weekday
    data = h5f["data/encounter_train/distribution_daily_number_encounters"]
    num_contacts_merged = data[0, :]
    p_full_merged = data[1, :]
    p_jack_merged = data[2, :]
    p_errs_merged = data[3, :]

    # surrogate data
    data = h5f[
        "data_surrogate_randomize_per_train/encounter_train/distribution_daily_number_encounters"
    ]
    s_num_contacts_merged = data[0, :]
    s_p_full_merged = data[1, :]
    s_p_jack_merged = data[2, :]
    s_p_errs_merged = data[3, :]

    # fit exponential
    def fitfunc(x, offset, slope):
        return offset + slope * x

    fitstart = 1
    fitend = None
    np.random.seed(815)
    y = np.log(p_full_merged[fitstart:fitend])
    valid_idx = np.isfinite(y)
    x = num_contacts_merged[fitstart:fitend]
    popt, pcov = curve_fit(fitfunc, xdata=x[valid_idx], ydata=y[valid_idx],)

    # offset
    popt[0] += 2
    p_full_merged_fit = np.exp(fitfunc(num_contacts_merged, *popt))
    fitstart = 15
    fitend = 51

    ax.errorbar(
        x=num_contacts_merged,
        y=p_full_merged,
        yerr=p_errs_merged,
        fmt="o",
        markersize=ms_default,
        # markerfacecolor="white",
        color=clrs.cond_enc_rate,
        ecolor=clrs.cond_enc_rate,
        alpha=1,
        elinewidth=1,
        capsize=1,
        zorder=3,
        label=f"data",
    )

    ax.plot(
        s_num_contacts_merged,
        s_p_full_merged,
        color=clrs.disease_psn_norm,
        alpha=1,
        zorder=2,
    )

    ax.plot(
        num_contacts_merged[fitstart:fitend],
        p_full_merged_fit[fitstart:fitend],
        color=clrs.cond_enc_rate,
        alpha=0.2,
        zorder=2,
        label="fit",
    )

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
def plot_dist_encounters_per_train(h5f, ax=None):

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    ax.set_rasterization_zorder(0)

    # d1: <number>, d2: <P>, d3: <number>(jackknife), d4: <P>(jackknife), d5: err(<number>), d6: err(<P>)

    # real data
    data = h5f["data/encounter_train/distribution_total_number_encounter_linbin"]
    num_contacts = data[0, :]
    p_full = data[1, :]
    num_errs = data[4, :]
    p_errs = data[5, :]

    # surrogate
    data = h5f[
        "data_surrogate_randomize_per_train/encounter_train/distribution_total_number_encounter_linbin"
    ]
    s_num_contacts = data[0, :]
    s_p_full = data[1, :]
    s_num_errs = data[4, :]
    s_p_errs = data[5, :]

    log.info(s_num_errs)
    log.info(s_p_errs)

    # this is a number, fit result
    exp_scale = h5f["data/encounter_train/distribution_total_number_encounter_fit_exp"]

    # color = clrs.medium
    color = clrs.cond_enc_rate

    ax.errorbar(
        x=num_contacts,
        y=p_full,
        xerr=num_errs,
        yerr=p_errs,
        fmt="o",
        markersize=ms_default,
        color=color,
        ecolor=color,
        alpha=1,
        elinewidth=1,
        capsize=0,
        zorder=1,
        label="data",
    )
    num_c_max = np.nanmax(num_contacts) * 1.1
    ax.plot(
        np.arange(num_c_max),
        scipy.stats.expon.pdf(np.arange(num_c_max), loc=0, scale=exp_scale),
        color=_alpha_to_solid_on_bg(color, 0.5),
        zorder=2,
        ls="-",
        label="fit",
    )

    ax.plot(
        s_num_contacts,
        s_p_full,
        color=clrs.disease_psn_norm,
        zorder=0,
        ls="-",
        label="surrogate",
    )

    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1.1e-2)
    ax.xaxis.set_major_locator(MultipleLocator(200))
    ax.xaxis.set_minor_locator(MultipleLocator(50))
    # for idx, lab in enumerate(ax.xaxis.get_ticklabels()):
    # if idx % 2 == 0:
    # lab.set_visible(False)

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
    real_train = h5.load(h5f["h5/filename"], f"/data/trains/train_{tid}") / 60 / 60 / 24
    surr_train = (
        h5.load(
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
        color=clrs.disease_psn_norm,
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
    h5f, ax=None, how="absolute", t_inf=3,
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
    ax.set_rasterization_zorder(0)

    assert how in ["relative", "absolute"]

    data = h5f["disease/delta/scan_mean_number_infectious_encounter"]
    range_inf = data["range_infectious"][:]
    range_lat = data["range_latent"][:]

    rdx = np.where(range_inf == t_inf)[0][0]

    if how == "absolute":
        data_2d = data["mean"][:]
        data_1d = data_2d[rdx, :]
        ax.plot(
            range_lat, data_1d, color="black", ls=":",
        )

        norm = data["mean_relative_to_poisson"][:][rdx, :]
        ax.plot(range_lat, data_1d / norm, color=clrs.disease_psn_norm)

    elif how == "relative":
        data_2d = data["mean_relative_to_poisson"][:] * 100
        data_1d = data_2d[rdx, :]
        ax.plot(range_lat, data_1d)

    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.margins(x=0, y=0)
    ax.set_ylim(22, 30)
    # ax.set_xlabel("Latent period (days)")
    # ax.set_ylabel(f"Encounter ({how})")
    fig.tight_layout()

    return ax


# Fig 2d, Fig 3b, SM
@warntry
def plot_conditional_rate(h5f, ax=None, which=["data"], control_plot=False, kwargs_overwrite=None):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    ax.set_rasterization_zorder(0)

    if not isinstance(which, list):
        which = list(which)

    norm_rate = 1 / 60 / 60 / 24
    e_step = 25

    # reuse for inset
    def local_plot(ax):
        for wdx, w in enumerate(which):
            log.info(w)
            if "data" in w:
                data = h5f[f"{w}/encounter_train/conditional_encounter_rate"]
            else:
                data = h5f[f"sample/{w}/conditional_encounter_rate"]

            r_time = data[0, :] * norm_rate
            r_full = data[1, :] / norm_rate
            try:
                r_errs = data[1, :] / norm_rate
            except:
                pass

            if kwargs_overwrite is None:
                kwargs = dict(alpha=1, label=w, zorder=0, color=f"C{wdx}")
            else:
                kwargs = kwargs_overwrite

            if kwargs_overwrite is not None:
                pass
            elif w == "data":
                kwargs["zorder"] = 2
                kwargs["color"] = clrs.cond_enc_rate
                if control_plot:
                    kwargs["alpha"] = 1.0
                    kwargs["lw"] = 0.75
            elif w == "data_surrogate_randomize_per_train":
                kwargs["color"] = clrs.disease_psn_norm
            elif control_plot:
                kwargs["zorder"] = 1

            # weighted -> thicker, more saturated
            if kwargs_overwrite is not None:
                pass
            elif w == "weibul_renewal_process":
                kwargs["color"] = _alpha_to_solid_on_bg(clrs.weibull, 0.5)
                kwargs["lw"] = 0.75
            elif w == "weibul_renewal_process_weighted_trains":
                kwargs["color"] = clrs.weibull
                kwargs["lw"] = 2

            elif w == "poisson_inhomogeneous":
                kwargs["color"] = _alpha_to_solid_on_bg(clrs.medium_psn, 0.4)
                kwargs["lw"] = 0.75
                kwargs["zorder"] = 4
            elif w == "poisson_inhomogeneous_weighted_trains":
                kwargs["color"] = clrs.medium_psn
                kwargs["lw"] = 2

            ax.plot(r_time, r_full, **kwargs)

            if w == "data" and not control_plot and kwargs_overwrite is None:
                # shaded regions for examples: 2,3 and 6,3
                idx = np.where((r_time > 6) & (r_time < 9))
                ax.fill_between(
                    r_time[idx],
                    y1=np.zeros(len(idx)),
                    y2=r_full[idx],
                    color="#faad7c",
                    alpha=0.4,
                    lw=0,
                    zorder=1,
                )

                idx = np.where((r_time > 2) & (r_time < 5))
                ax.fill_between(
                    r_time[idx],
                    y1=np.zeros(len(idx)),
                    y2=r_full[idx],
                    color="#49737a",
                    alpha=0.3,
                    lw=0,
                    zorder=1,
                )

    local_plot(ax)

    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(0, 60)

    if control_plot:
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.yaxis.set_minor_locator(MultipleLocator(10))
    else:
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
        for label in ax.xaxis.get_ticklabels()[::2]:
            label.set_visible(False)

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
    h5f, ax=None, which="data", how="relative", control_plot=False
):
    """
        # Parameters
        which : str, "data" or samples from h5f["sample/"], e.g. "poisson_inhomogeneous"
        how : str, "relative" or "absolute"
        control_plot : bool, set to `True` for use in Fig. 4 for slightly different styling
    """

    assert how in ["relative", "absolute"]
    samples = []
    try:
        samples = list(h5f["sample"].keys())
    except Exception as e:
        log.warning(e)
    assert which in ["data"] + samples, f"did you sample {which}?"

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    ax.set_rasterization_zorder(0)

    if which == "data":
        data = h5f["disease/delta/scan_mean_number_infectious_encounter"]
    else:
        # which == "poisson_homogeneous_weighted_train"
        data = h5f[
            f"sample/{which}/disease/delta/scan_mean_number_infectious_encounter"
        ]

    range_inf = data["range_infectious"][:]
    range_lat = data["range_latent"][:]

    if how == "absolute_old":
        data_2d = data["mean"][:]
        kwargs = dict(
            vmin=0,
            vmax=None,
            # cbar_kws={"label": "Number of infectious encounter"},
            cmap=clrs.ccmap["pastel_1"],
        )
    elif how == "absolute":
        data_2d = data["mean"][:]
        kwargs = dict(
            vmin=0,
            vmax=None,
            # cbar_kws={"label": "Number of infectious encounter"},
            cmap=clrs.ccmap["div_pastel_1"].reversed(),
        )

    elif how == "relative":
        data_2d = data["mean_relative_to_poisson"][:] * 100
        kwargs = dict(
            vmin=50, vmax=150, center=100, cmap=clrs.ccmap["div_pastel_1"].reversed(),
        )
    if control_plot:
        kwargs["cbar"] = False
    else:
        if show_xlabel:
            ax.set_xlabel(r"Latent period (days)")
        if show_ylabel:
            ax.set_ylabel(r"Infectious period (days)")

    xticklabels = []
    xticks = []
    for idx, x in enumerate(range_lat):
        if x.is_integer() and x > 0 and x < 8:
            xticklabels.append(str(int(x)))
            xticks.append(idx)

    yticklabels = []
    yticks = []
    for idy, y in enumerate(range_inf):
        if y.is_integer() and y < 8:
            yticklabels.append(str(int(y)))
            yticks.append(idy)

    sns.heatmap(
        data_2d,
        ax=ax,
        # linewidth=0.01,
        square=True,
        xticklabels=False,
        yticklabels=False,
        zorder=-5,
        **kwargs,
    )

    # ax.set_xlabel("Latent period (days)")
    # ax.set_ylabel("Infectious period (days)")
    # ax.set_title(which)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.invert_yaxis()

    if show_title:
        ax.set_title(f"{which}", fontsize=8)

    fig.tight_layout()

    # if control_plot:
    #     ax.tick_params(
    #         top=True,
    #         bottom=True,
    #         left=True,
    #         right=True,
    #         labeltop=False,
    #         labelbottom=False,
    #         labelright=False,
    #         labelleft=False,
    #     )
        # _set_size(ax, 2.1 * cm, 2.1 * cm)

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

    k_vals = [1e8, 100, 10, 1]
    for idx, k in enumerate(k_vals):
        y = scipy.stats.gamma.pdf(x, a=k, loc=0, scale=mean / k)
        k_str = f"$k={k}$" if k != 1e8 else r"$k\to\infty$"
        ax.plot(x, y, color=clrs.dispersion[k], label=k_str)

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
    c_todo.append(clrs.n_low)
    c_todo.append(clrs.n_high)
    c_todo.append(clrs.n_psn)
    c_todo.append(clrs.n_psn)

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
    fig, axes = plt.subplots(nrows=3, sharex=True, sharey=True,)

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

        ax.plot(x, hist, zorder=5, ls="--", color=clrs.dispersion[k])
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
                ax.plot(x, y, alpha=1, zorder=2, color=clrs.dispersion[k], lw=0.5)
            else:
                ax.plot(x, y, alpha=0.02, zorder=-1, color=clrs.dispersion[k], lw=0.5)

    ax = axes[-1]
    ax.set_xlabel("Time (in days)")
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.xaxis.set_major_locator(MultipleLocator(2))
    # for idx, lab in enumerate(ax.xaxis.get_ticklabels()):
    #     if (idx) % 2 == 0:
    #         lab.set_visible(False)
    fig.tight_layout()


# Fig 4c
@warntry
def plot_disease_dist_infectious_encounters(h5f, ax=None, k="k_inf", periods="slow"):

    assert k in ["k_inf", "k_10.0", "k_1.0"]
    assert periods in ["fast", "slow"]
    control = None  # onset_train or sth?

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    def local_plot(data, color, zorder=0):
        num_encounter = data[0, :]
        p_full = data[1, :]
        p_jack = data[2, :]
        p_errs = data[3, :]
        ax.plot(num_encounter, p_full, color=color, zorder=zorder)
        ref = _ev(num_encounter, p_full)
        ax.axvline(
            ref,
            0,
            1,
            color=_alpha_to_solid_on_bg(color, 0.5),
            ls=":",
            zorder=zorder - 10,
        )
        return ref

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

    c_todo = []
    c_todo.append(clrs.n_low)
    c_todo.append(clrs.n_high)
    c_todo.append(clrs.n_psn)
    c_todo.append(clrs.n_psn)

    # iterate over all periods and chosen colors
    for period, color in zip(p_todo, c_todo):

        # depending on k, the path may differ (for k_inf we have delta disase)
        if k == "k_inf":
            path = f"disease/delta_{period}"
        else:
            path = f"disease/gamma_{period}/{k}"

        if control is not None:
            path += f"/control_random_disease_{control}"
        path += "/distribution_infectious_encounter"

        try:
            assert path in h5f.keypaths()
            data = h5f[path]

            zorder = 2
            if "_surrogate" in period:
                zorder = 0

            ref = local_plot(data, color, zorder)
            log.info(f"{k}\t{period}:\t{ref:.2f}")
        except Exception as e:
            log.warning(f"Failed to plot {path}")

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

    if show_xlabel:
        ax.set_xlabel(r"Pot. inf. encounters")
    if show_ylabel:
        ax.set_ylabel(r"Distribution")
    if show_title:
        ax.set_title(title, fontsize=8)
    # if show_legend:
    #     ax.legend()
    # if show_legend_in_extra_panel:
    #     _legend_into_new_axes(ax)

    fig.tight_layout()
    _set_size(ax, 3.3 * cm, 1.7 * cm)

def compare_disease_dist_infectious_encounters_to_psn(h5f, ax=None, periods="slow"):
    """
    similar to above, but instead of comparing to randomized, we compare to the
    train weighted inhom. psn. process
    """

    k = "k_inf" # we only do this for delta disease model
    assert periods in ["fast", "slow"]
    control = None  # onset_train or sth?

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
        ax.axvline(
            ref,
            0,
            1,
            color=_alpha_to_solid_on_bg(color, 0.5),
            ls=":",
            zorder=zorder - 10,
        )
        return ref

    # original data is stored in /disease
    # poisson is stored in /sample/psn_inh.../disease

    for path_prefix in ["", "sample/poisson_inhomogeneous_weighted_trains/"]:

        p_todo = []
        if periods == "slow":
            p_todo.append("2_3")  # blue
            p_todo.append("6_3")  # red
        elif periods == "fast":
            p_todo.append("1_0.5")  # blue
            p_todo.append("1.5_0.5")  # red

        c_todo = []
        c_todo.append(clrs.n_low)
        c_todo.append(clrs.n_high)

        # iterate over all periods and chosen colors
        for period, color in zip(p_todo, c_todo):

            path = f"{path_prefix}disease/delta_{period}"

            if control is not None:
                path += f"/control_random_disease_{control}"
            path += "/distribution_infectious_encounter"

            try:
                assert path in h5f.keypaths()
                data = h5f[path]

                zorder = 2
                if "_surrogate" in period:
                    zorder = 0

                kwargs = dict()
                if path_prefix == "":
                    color="gray"
                    # kwargs["ls"] = "-"
                    # kwargs["alpha"] = 1.0
                # else:
                    # kwargs["ls"] = ":"
                    # kwargs["alpha"] = 0.5


                ref = local_plot(data, color, zorder, **kwargs)
                log.info(f"{k}\t{period}:\t{ref:.2f}")
            except Exception as e:
                log.warning(f"Failed to plot {path}")
                raise(e)

    ax.set_xlim(-5, 150)
    ax.set_yscale("log")
    if periods == "slow":
        ax.set_ylim(1e-6, 1)
        # ax.set_ylim(1e-3, 1)
    elif periods == "fast":
        ax.set_ylim(1e-6, 1)

    _fix_log_ticks(ax.yaxis, every=1)
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_minor_locator(MultipleLocator(10))


    if k == "k_inf":
        title = r"$k\to\infty$"
    else:
        title = f"$k = {float(k[2:]):.0f}$"

    title += f"    {periods}"
    if control is not None:
        title += f" {control}"

    if show_xlabel:
        ax.set_xlabel(r"Pot. inf. encounters")
    if show_ylabel:
        ax.set_ylabel(r"Distribution")
    if show_title:
        ax.set_title(title, fontsize=8)
    # if show_legend:
    #     ax.legend()
    # if show_legend_in_extra_panel:
    #     _legend_into_new_axes(ax)

    fig.tight_layout()
    _set_size(ax, 3.3 * cm, 1.7 * cm)



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
    h5f, which="gamma_6_3", k_sel="k_10.0", inset=True,
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
        ["data", "onset_wtrain_wtime", "onset_wtrain", "onset_wtime", "onset",]
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
                    f"`{control}` has mean: {mean:.1f} and ev: {_ev(num_encounter, p_full):.1f}"
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
        ["data", "onset_wtrain_wtime", "onset_wtrain", "onset_wtime", "onset",]
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
                        f"`{control}` has mean: {mean:.1f} and ev: {_ev(num_encounter, p_full):.1f}"
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
def plot_r(h5f, which=["latent_1.00", "latent_2.00", "latent_6.00"]):
    fig, ax = plt.subplots()

    def plot_cases(cases, color, **kwargs):
        num_rep = len(cases.keys())
        k0 = list(cases.keys())[0]
        x = cases[k0][0, :] / 60 / 60 / 24 / 7
        y_all = np.zeros((len(x), num_rep))
        for idx, key in enumerate(cases.keys()):
            y_all[:, idx] = cases[key][1, :]
        y_all[y_all == 0] = np.nan
        y_mean = np.nanmean(y_all, axis=-1)
        y_err = np.nanstd(y_all, axis=-1) / np.sqrt(num_rep)

        ax.plot(x, y_mean, color=color, **kwargs)

    for wdx, w in enumerate(which):
        log.info(w)
        real = h5f["measurements"]["cases"][w]
        surr = h5f["measurements_randomized_per_train"]["cases"][w]
        base_color = f"C{wdx}"
        if "1.00" in w:
            base_color = "#868686"
        elif "2.00" in w:
            base_color = clrs.n_low
        elif "6.00" in w:
            base_color = clrs.n_high
        plot_cases(
            surr, color=_alpha_to_solid_on_bg(base_color, 0.3), label=f"surr {w}"
        )
        plot_cases(real, color=base_color, label=f"real {w}")

    ax.axhline(1e4, 0, 1, color="gray", ls="--")
    ax.set_ylim(1, 1.1e6)
    ax.set_xlim(0, 14.75)
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

    fig.tight_layout()
    _set_size(ax, 3.0 * cm, 2.3 * cm)

    return ax


# Fig 5
@warntry
def plot_r4():
    fig, ax = plt.subplots()

    rand = np.loadtxt(
        "./out_mf/analysis_mean-field_R_measurements_randomized_per_train.dat",
        unpack=True,
    )
    data = np.loadtxt("./out_mf/analysis_mean-field_R_measurements.dat", unpack=True,)

    ax.errorbar(
        rand[0, :],
        rand[1, :],
        yerr=rand[2, :],
        label="r4 rand",
        color=clrs.disease_psn_norm,
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
        color=clrs.cond_enc_rate,
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
        range_lat, data_1d, color=clrs.cond_enc_rate, ls="-",
    )

    norm = data["mean_relative_to_poisson"][:][rdx, :]
    data_1d_norm = data_1d / norm
    ax.plot(range_lat, data_1d_norm, color=clrs.disease_psn_norm, ls="-")

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

    data = np.loadtxt("./out_mf/analysis_mean-field_R_measurements.dat", unpack=True,)
    rand = np.loadtxt(
        "./out_mf/analysis_mean-field_R_measurements_randomized_per_train.dat",
        unpack=True,
    )

    ax.errorbar(
        rand[0, :],
        rand[5, :],
        yerr=rand[6, :],
        label="r0 rand",
        color=clrs.disease_psn_norm,
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
        color=clrs.cond_enc_rate,
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


def _legend_into_new_axes(ax):
    fig, ax_leg = plt.subplots(figsize=(6 * cm, 6 * cm))
    h, l = ax.get_legend_handles_labels()
    ax_leg.axis("off")
    ax_leg.legend(h, l, loc="upper left")


def _set_size(ax, w, h):
    """ w, h: width, height in inches """
    # https://stackoverflow.com/questions/44970010/axes-class-set-explicitly-size-width-height-of-axes-in-given-units
    if not use_compact_size:
        return
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w) / (r - l)
    figh = float(h) / (t - b)
    ax.figure.set_size_inches(figw, figh)


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
                    ssres = np.sum(residuals ** 2)

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
