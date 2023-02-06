#using Pkg
#Pkg.add(["DelimitedFiles", "ProgressMeter", "StatsBase", "Roots", "LinearAlgebra", "Printf", "HDF5", "Distributions", "SpecialFunctions", "Parameters"])
#Pkg.add(url="https://github.com/JuliaIO/HDF5.jl")
#
using HDF5
using Printf
using Statistics
using StatsBase
using LinearAlgebra
using Roots
using IterTools

using ProgressMeter

include("utils.jl")
include("load_data.jl")
include("write_files.jl")
include("temporal_features.jl")
include("disease_spread.jl")
include("error.jl")
include("sampling.jl")
include("surrogate.jl")

default_range_latent     = 0:0.05:8
default_range_infectious = 0.05:0.05:8
default_range_R0 = 1.0:1.0:5.0
default_support_crate = 0:300:seconds_from_days(7*1.5)

"""
# when using only intermediate functions:
    cas, ets, list_contacts = load_processed_data()
    filename="./test.h5"
    experiment=Copenhagen()
    skip_jackknife=true
"""
function analyse_all(
        experiment::ContactData;
        # optional
        minimum_duration = 15*60,
        path_dat = "./dat",
        path_out = "./out",
        filter_out_incomplete=false,
        ct_range_times = 0:60*60:10*24*60*60,
        disease_model = DiseaseDefault(),
        seed=1000,
        # 0: essentials, 1: essentials and fast extras, 2: most, 3: everything
        # to skip jackknife analysis set gloibal variable `skip_jackknife = true`
        level_of_details = 2,
    )
    lod = level_of_details
    mkpath(path_out)
    filename_data = @sprintf("%s/data_%s_filtered_%dmin.h5",
        path_out, label(experiment), minimum_duration/60)
    filename_rand = @sprintf("%s/data_randomized_per_train_%s_filtered_%dmin.h5",
        path_out, label(experiment), minimum_duration/60)
    filename_rand_all = @sprintf("%s/data_randomized_all_%s_filtered_%dmin.h5",
        path_out, label(experiment), minimum_duration/60)

    if filter_out_incomplete
        filename_data = replace(filename_data, ".h5" => "_filterOutIncomplete.h5")
        filename_rand = replace(filename_rand, ".h5" => "_filterOutIncomplete.h5")
        filename_rand_all = replace(filename_rand_all, ".h5" => "_filterOutIncomplete.h5")
    end

    ###########################################################################
    # data
    # experiment=Copenhagen(); minimum_duration=15*60; path_dat="./dat";filter_out_incomplete=false;
    cas, ets, list_contacts = load_processed_data(experiment, minimum_duration, path_dat, filter_out_incomplete=filter_out_incomplete);
    # write encounter train
    lod >= 1 && myh5write(filename_data,"/trains/",ets)

    if experiment == InVS15()
        support_crate=0:300:seconds_from_days(7*1.5)
    else
        support_crate=0:timestep(ets):seconds_from_days(7*1.5)
    end

    ###########################################################################
    # temporal features

    lod >= 0 && analyse_temporal_features_of_encounter_train(ets, filename_data, "/", support_crate=support_crate)
    lod >= 2 && analyse_contact_duration(list_contacts, experiment, filename_data, "/extra/contacts")
    lod >= 2 && analyse_temporal_features_of_contact_activity(cas, filename_data, "/extra/contact_activity")


    ###########################################################################
    # surrogate data

    # disease periods
    range_latent = default_range_latent
    range_infectious = default_range_infectious

    if lod >= 1
        root="/"
        sur = surrogate_randomize_per_train(ets, seed)
        myh5write(filename_rand,@sprintf("%s/trains/",root), sur)
        analyse_temporal_features_of_encounter_train(sur, filename_rand, root, support_crate=support_crate)
        # disease
        analyse_infectious_encounter_scan_delta(range_latent, range_infectious, sur, filename_rand, @sprintf("%s/disease/delta", root))
        analyse_infectious_encounter_detail(DeltaDiseaseModel(seconds_from_days(2), seconds_from_days(3)), sur, filename_rand, "/disease/delta_2_3")
        analyse_infectious_encounter_detail(DeltaDiseaseModel(seconds_from_days(6), seconds_from_days(3)), sur, filename_rand, "/disease/delta_6_3")
        analyse_infectious_encounter_detail(DeltaDiseaseModel(seconds_from_days(1), seconds_from_days(0.5)), sur, filename_rand, "/disease/delta_1_0.5")
        analyse_infectious_encounter_detail(DeltaDiseaseModel(seconds_from_days(1.5), seconds_from_days(0.5)), sur, filename_rand, "/disease/delta_1.5_0.5")
    end

    if lod >= 2
        # surrogate version 2, where randimization is not constrained within trains
        root="/"
        sur_rand_all = surrogate_randomize_all(ets, seed)
        myh5write(filename_rand_all,@sprintf("%s/trains/",root), sur_rand_all)
        analyse_temporal_features_of_encounter_train(sur_rand_all, filename_rand_all, root, support_crate=support_crate)
        analyse_infectious_encounter_scan_delta(range_latent, range_infectious, sur_rand_all, filename_rand_all, @sprintf("%s/disease/delta", root))
        analyse_infectious_encounter_detail(DeltaDiseaseModel(seconds_from_days(2), seconds_from_days(3)), sur_rand_all, filename_rand_all, "/disease/delta_2_3")
        analyse_infectious_encounter_detail(DeltaDiseaseModel(seconds_from_days(6), seconds_from_days(3)), sur_rand_all, filename_rand_all, "/disease/delta_6_3")
        analyse_infectious_encounter_detail(DeltaDiseaseModel(seconds_from_days(1), seconds_from_days(0.5)), sur_rand_all, filename_rand_all, "/disease/delta_1_0.5")
        analyse_infectious_encounter_detail(DeltaDiseaseModel(seconds_from_days(1.5), seconds_from_days(0.5)), sur_rand_all, filename_rand_all, "/disease/delta_1.5_0.5")
    end


    ###########################################################################
    # disease spread
    # only for data `ets` and surrogate version 1 `sur`
    if lod >= 1
        analyse_infectious_encounter_scan_delta(range_latent, range_infectious, ets, filename_data, "/disease/delta")
        #details
        analyse_infectious_encounter_detail(DeltaDiseaseModel(seconds_from_days(2), seconds_from_days(3)), ets, filename_data, "/disease/delta_2_3")
        analyse_infectious_encounter_detail(DeltaDiseaseModel(seconds_from_days(6), seconds_from_days(3)), ets, filename_data, "/disease/delta_6_3")
        # details at different points
        analyse_infectious_encounter_detail(DeltaDiseaseModel(seconds_from_days(1), seconds_from_days(0.5)), ets, filename_data, "/disease/delta_1_0.5")
        analyse_infectious_encounter_detail(DeltaDiseaseModel(seconds_from_days(1.5), seconds_from_days(0.5)), ets, filename_data, "/disease/delta_1.5_0.5")

        #gamma - scan
        range_k = 10 .^ collect(0:0.2:5)
        analyse_infectious_encounter_scan_gamma((2,3), range_k, ets, filename_data, "/disease/gamma_2_3")
        analyse_infectious_encounter_scan_gamma((6,3), range_k, ets, filename_data, "/disease/gamma_6_3")
        analyse_infectious_encounter_scan_gamma((2,3), range_k, sur, filename_rand, "/disease/gamma_2_3")
        analyse_infectious_encounter_scan_gamma((6,3), range_k, sur, filename_rand, "/disease/gamma_6_3")
        #gamma - detail
        analyse_infectious_encounter_detail(GammaDiseaseModel(seconds_from_days(2), seconds_from_days(3), 1), ets, filename_data, "/disease/gamma_2_3/k_1.0")
        analyse_infectious_encounter_detail(GammaDiseaseModel(seconds_from_days(6), seconds_from_days(3), 1), ets, filename_data, "/disease/gamma_6_3/k_1.0")
        analyse_infectious_encounter_detail(GammaDiseaseModel(seconds_from_days(2), seconds_from_days(3), 10), ets, filename_data, "/disease/gamma_2_3/k_10.0")
        analyse_infectious_encounter_detail(GammaDiseaseModel(seconds_from_days(6), seconds_from_days(3), 10), ets, filename_data, "/disease/gamma_6_3/k_10.0")
        # gamma - detail, surrogate
        analyse_infectious_encounter_detail(GammaDiseaseModel(seconds_from_days(2), seconds_from_days(3), 1), sur, filename_rand, "/disease/gamma_2_3/k_1.0")
        analyse_infectious_encounter_detail(GammaDiseaseModel(seconds_from_days(6), seconds_from_days(3), 1), sur, filename_rand, "/disease/gamma_6_3/k_1.0")
        analyse_infectious_encounter_detail(GammaDiseaseModel(seconds_from_days(2), seconds_from_days(3), 10), sur, filename_rand, "/disease/gamma_2_3/k_10.0")
        analyse_infectious_encounter_detail(GammaDiseaseModel(seconds_from_days(6), seconds_from_days(3), 10), sur, filename_rand, "/disease/gamma_6_3/k_10.0")

        #gamma - scan, less realistic points
        range_k = 10 .^ collect(0:0.2:5)
        analyse_infectious_encounter_scan_gamma((1,0.5), range_k, ets, filename_data, "/disease/gamma_1_0.5")
        analyse_infectious_encounter_scan_gamma((1.5,0.5), range_k, ets, filename_data, "/disease/gamma_1.5_0.5")
        analyse_infectious_encounter_scan_gamma((1,0.5), range_k, sur, filename_rand, "/disease/gamma_1_0.5")
        analyse_infectious_encounter_scan_gamma((1.5,0.5), range_k, sur, filename_rand, "/disease/gamma_1.5_0.5")
        #gamma - detail
        analyse_infectious_encounter_detail(GammaDiseaseModel(seconds_from_days(1), seconds_from_days(0.5), 1), ets, filename_data, "/disease/gamma_1_0.5/k_1.0")
        analyse_infectious_encounter_detail(GammaDiseaseModel(seconds_from_days(1.5), seconds_from_days(0.5), 1), ets, filename_data, "/disease/gamma_1.5_0.5/k_1.0")
        analyse_infectious_encounter_detail(GammaDiseaseModel(seconds_from_days(1), seconds_from_days(0.5), 10), ets, filename_data, "/disease/gamma_1_0.5/k_10.0")
        analyse_infectious_encounter_detail(GammaDiseaseModel(seconds_from_days(1.5), seconds_from_days(0.5), 10), ets, filename_data, "/disease/gamma_1.5_0.5/k_10.0")
        # gamma - detail, surrogate
        analyse_infectious_encounter_detail(GammaDiseaseModel(seconds_from_days(1), seconds_from_days(0.5), 1), sur, filename_rand, "/disease/gamma_1_0.5/k_1.0")
        analyse_infectious_encounter_detail(GammaDiseaseModel(seconds_from_days(1.5), seconds_from_days(0.5), 1), sur, filename_rand, "/disease/gamma_1.5_0.5/k_1.0")
        analyse_infectious_encounter_detail(GammaDiseaseModel(seconds_from_days(1), seconds_from_days(0.5), 10), sur, filename_rand, "/disease/gamma_1_0.5/k_10.0")
        analyse_infectious_encounter_detail(GammaDiseaseModel(seconds_from_days(1.5), seconds_from_days(0.5), 10), sur, filename_rand, "/disease/gamma_1.5_0.5/k_10.0")
    end

    ##########################################################################
    # dispersion

    return
end


###############################################################################
###############################################################################
### analyse contact activitity
function analyse_temporal_features_of_contact_activity(
        cas::contact_activities{I,T},
        filename::String,
        root::String;
        ############ optional
        lags=0:1:Int(seconds_from_days(7)/step(times(cas))),    # 1-week autocorrelation function
        times=0:60*60:seconds_from_days(7)           # 1-week contact rate (taken directly from first argument, seems to work)

    ) where {I,T}
    println("writing to ", filename, " dset ", root)
    println("... coefficient of variation")
    cv, cvj, cverr = coefficient_variation(cas, jackknife_error=true)
    myh5write(filename, @sprintf("%s/coefficient_variation", root), hcat(cv, cvj, cverr))
    myh5desc(filename, @sprintf("%s/coefficient_variation", root),
             "Coefficient of variation (std/mean of contact per timestep), d1: cv(full), d2: cv(jackknife), d3: error")

    #println("... autocorrelation function")
    #c, cj, cerr = jackknife(x->autocorrelation_function(x, lags), cas)
    #myh5write(filename, @sprintf("%s/autocorrelation", root), hcat(lags, c, cj, cerr))
    #myh5desc(filename, @sprintf("%s/autocorrelation", root),
    #         "autocorrelation of individual contacts over time, d1: timetep-lag, d2: C(full), d3: C(jackknife), d4: error")

    println("... contact rate")
    r, rj, rerr = jackknife(x->rate(x, times).weights, cas)
    myh5write(filename, @sprintf("%s/rate", root), hcat(times[1:end-1], r, rj, rerr))
    myh5desc(filename, @sprintf("%s/rate", root),
             "contact rate averaged across indivdials and across repetition of `times` in experiment, d1: times, d2: rate(full), d3: rate(jackknife), d4: error")

    println("... contact rate (high precision)")
    times_highres = 0:300:seconds_from_days(7)
    r, rj, rerr = jackknife(x->rate(x, times_highres).weights, cas)
    myh5write(filename, @sprintf("%s/rate_5min", root), hcat(times_highres[1:end-1], r, rj, rerr))
    myh5desc(filename, @sprintf("%s/rate_5min", root),
             "same as rate but at 5 min timesteps")

    println("... distribution of cluster sizes")
    samples_cluster_sizes = sample_cluster_sizes(cas)
    samples_cluster_sizes = samples_cluster_sizes[length.(samples_cluster_sizes) .> 0]
    full_dist = distribution(samples_cluster_sizes);
    weights_cluster(x) = distribution(x, edges=full_dist.edges[1]).weights
    P, Pj, Perr = jackknife_mj(weights_cluster, samples_cluster_sizes, naive=full_dist.weights);
    myh5write(filename, @sprintf("%s/distribution_encounter_cluster", root), hcat(full_dist.edges[1][1:end-1], P, Pj, Perr))
    myh5desc(filename, @sprintf("%s/distribution_encounter_cluster", root),
             "distribution of contact cluster (total sum of successive activity separated by zero activity), d1: size, d2: P(full), d3: P(jackknife), d4: error")

    xbin, Pbin, = logbin(full_dist)
    array_cluster_logbin(x) = hcat(logbin(distribution(x, edges=full_dist.edges[1]))...)[:,1:2]
    xP, xP_J, xP_err = jackknife_mj(array_cluster_logbin, samples_cluster_sizes, naive=hcat(xbin,Pbin));
    myh5write(filename, @sprintf("%s/distribution_encounter_cluster_logbin", root), hcat(xP, xP_J, xP_err))
    myh5desc(filename, @sprintf("%s/distribution_encounter_cluster_logbin", root),
             "distribution of inter-encounter intervals with log binning, d1: <duration>, d2: <P>, d3: <duration>(jackknife), d4: <P>(jackknife), d5: err(<duration>), d6: err(<P>)")
end

###############################################################################
###############################################################################
### analyse contact stuff
function  analyse_contact_duration(list_contacts, experiment::ContactData, filename::String, root::String)
    println("writing to ", filename, " dset ", root)
    list_durations = [getindex.(contacts, 2) for contacts in list_contacts];
    # remove lists with zero elements because they 1) do not contribute to the
    # statistics of the durations and 2) thereby break the jackknife routine
    list_durations = list_durations[length.(list_durations) .> 0]

    println("... distribution of contact durations")
    # full distribution
    full_dist = distribution_durations(list_durations, timestep=timestep(experiment));
    weights(x) = distribution_durations(x, edges=full_dist.edges[1]).weights
    P, Pj, Perr = jackknife_mj(weights, list_durations, naive=full_dist.weights);
    myh5write(filename, @sprintf("%s/distribution_contact_durations", root), hcat(full_dist.edges[1][1:end-1], P, Pj, Perr))
    myh5desc(filename, @sprintf("%s/distribution_contact_durations", root),
             "distribution of contact durations, d1: duration, d2: P(full), d3: P(jackknife), d4: error")

    # log-binning: remove initial range of short durations which are trivially excluded by our filter
    mask = trues(length(full_dist.edges[1]))
    mask[1:findfirst(full_dist.weights .> 0)] .= false
    full_dist = distribution_durations(list_durations, edges=full_dist.edges[1][mask])
    xbin, Pbin = logbin(full_dist)
    array_xP_logbin(x) = hcat(logbin(distribution_durations(x, edges=full_dist.edges[1]))...)[:,1:2]
    xP, xP_J, xP_err = jackknife_mj(array_xP_logbin, list_durations, naive=hcat(xbin,Pbin));
    myh5write(filename, @sprintf("%s/distribution_contact_durations_logbin", root), hcat(xP, xP_J, xP_err))
    myh5desc(filename, @sprintf("%s/distribution_contact_durations_logbin", root),
             "distribution of contact durations with log binning, d1: <duration>, d2: <P>, d3: <duration>(jackknife), d4: <P>(jackknife), d5: err(<duration>), d6: err(<P>)")
end

###############################################################################
###############################################################################
### analyse contact trains
function analyse_temporal_features_of_encounter_train(
        ets::encounter_trains{I,T},
        filename::String,
        root::String;
        ############ optional/default
        # lags for autocorrelation
        lags = 0:1:200,
        # 1-week contact rate in steps of hours
        times_rate = 0:60*60:seconds_from_days(7),
        # 1.5-week conditional contact rate
        support_crate = 0:timestep(ets):seconds_from_days(7*1.5),
    ) where {I,T}
    println("writing to ", filename, " dset ", root)

    println("... distribution total number encounter per train")
    edges=0:1:700
    f_weights(x) = normalize!(fit(Histogram{Float64}, length.(trains(x)), edges)).weights
    w, wj, werr = jackknife(f_weights, ets)
    myh5write(filename, @sprintf("%s/distribution_total_number_encounter", root), hcat(edges[1:end-1], w, wj, werr))
    myh5desc(filename, @sprintf("%s/distribution_total_number_encounter", root),
             "distribution of total number of encounter across trains, d1: num encounter, d2: P(full), d3: P(jack), d4: error")
    fit_exp = fit(Exponential, length.(trains(ets)))
    myh5write(filename, @sprintf("%s/distribution_total_number_encounter_fit_exp", root), [mean(fit_exp)])
    myh5desc(filename, @sprintf("%s/distribution_total_number_encounter_fit_exp", root),
             "fit result for exponential distribution: scale (mean)")
    xbin, Pbin = logbin(edges[1:end-1], w, increment_factor=1, bin_width_start=10)
    # linear binning due to increment factor 1
    array_xP_linbin(x) = hcat(logbin(edges[1:end-1], f_weights(x), increment_factor=1, bin_width_start=10)...)[:,1:2]
    xP, xP_J, xP_err = jackknife(array_xP_linbin, ets, naive=hcat(xbin,Pbin));
    myh5write(filename, @sprintf("%s/distribution_total_number_encounter_linbin", root), hcat(xP, xP_J, xP_err))
    myh5desc(filename, @sprintf("%s/distribution_total_number_encounter_linbin", root),
             "distribution of inter-encounter intervals with log binning, d1: <number>, d2: <P>, d3: <number>(jackknife), d4: <P>(jackknife), d5: err(<number>), d6: err(<P>)")


    println("... daily number of encounter")
    # error is simple jackknife because the basic statistics is the number of
    # encounter per time window, which is the same across all trains (result is
    # a clear zero for empty trains)
    dist_daily = distribution_number_encounter(ets, seconds_from_days(1))[1]
    w, wj, werr = jackknife(x->distribution_number_encounter(x, seconds_from_days(1), edges=dist_daily.edges[1])[1].weights, ets);
    myh5write(filename, @sprintf("%s/distribution_daily_number_encounters", root), hcat(dist_daily.edges[1][1:end-1], w, wj, werr))
    myh5desc(filename, @sprintf("%s/distribution_daily_number_encounters", root),
             "distribution of daily number of encounter (weekend AND weekday), d1: num encounter, d2: P(full) , d3: P(jack), d4: error")

    # partition experiment into windows of size 1 day and sort according to
    # pattern to groups 1 (weekend) and 2 (weekday)
    pattern = [1,2*ones(5)...,1]
    # we want the results of distribution_number_contacts(ets, seconds_from_days(1), pattern=pattern, edges=dist_daily.edges[1])
    # but this is a dictionary of distributions. Instead, for the jackknife
    # error function we need a 2D Array to compute all errors consistently with
    # a single run -> hence we transform to an array of log-weights
    dict_dist(x, edges) = distribution_number_encounter(x, seconds_from_days(1), pattern, edges=edges)
    array_weights(x, edges) = hcat([dist.weights for dist in values(dict_dist(x, edges))]...)
    w, wj, werr = jackknife(x->array_weights(x, dist_daily.edges[1]), ets);
    myh5write(filename, @sprintf("%s/distribution_daily_number_encounters_sep", root), hcat(dist_daily.edges[1][1:end-1], w, wj, werr))
    myh5desc(filename, @sprintf("%s/distribution_daily_number_encounters_sep", root),
             "distribution of daily number of encounter (weekend VS weekday), d1: num encounter, d2: P(full) weekend, d3: P(full) weekday, d4: P(jack) weekend, d5: P(jack) weekday, d6: err weekend, d7: err weekday")


    println("... encounter rate")
    # error is simple jackknife because the basic statistics is number of
    # coontacts per time step, which is the same across all trains
    r, rj, rerr = jackknife(x->rate(x, times_rate).weights, ets)
    myh5write(filename, @sprintf("%s/rate", root), hcat(times_rate[1:end-1], r, rj, rerr))
    myh5desc(filename, @sprintf("%s/rate", root),
             "encounter rate averaged across indivdials and across repetition of `times` in experiment, d1: times, d2: rate(full), d3: rate(jackknife), d4: error")

    # we also need to write the encounter rate at the higher time-resolution because
    # the convolution with the duration needs to match time discretization
    times_rate_highres = 0:300:seconds_from_days(7)
    r, rj, rerr = jackknife(x->rate(x, times_rate_highres).weights, ets)
    myh5write(filename, @sprintf("%s/rate_5min", root), hcat(times_rate_highres[1:end-1], r, rj, rerr))
    myh5desc(filename, @sprintf("%s/rate_5min", root),
             "same as `rate` but at 5 min timesteps")

    println("... conditional encounter rate")
    # error is delete-mj jackknife because basic statistics is the average over encounter that we condition on. Hence, we need to remove the emtpy trains
    ets_noempty = ets[length.(ets).>0]
    r, rj, rerr = jackknife_mj(x->conditional_encounter_rate(x, support_crate).weights, ets_noempty)
    myh5write(filename, @sprintf("%s/conditional_encounter_rate", root), hcat(support_crate[1:end-1], r, rj, rerr))
    myh5desc(filename, @sprintf("%s/conditional_encounter_rate", root),
             "average encounter rate conditioned on having an encounter at time 0, averaged across all encounter in experiment, d1: times, d2: rate(full), d3: rate(jackknife), d4: error")


    println("... distribution of inter-encounter intervals")
    # error is delete-mj jackknife because basic statistics is the
    # inter-encounter interval, a random variable that occurs differenlty often
    # in different trains
    list_dts = inter_encounter_intervals(ets);
    list_dts = list_dts[length.(list_dts) .> 0]
    full_dist = distribution_durations(list_dts, timestep=timestep(ets));
    weights(x) = distribution_durations(x, edges=full_dist.edges[1]).weights
    P, Pj, Perr = jackknife_mj(weights, list_dts, naive=full_dist.weights);
    myh5write(filename, @sprintf("%s/distribution_inter_encounter_intervals", root), hcat(full_dist.edges[1][1:end-1], P, Pj, Perr))
    myh5desc(filename, @sprintf("%s/distribution_inter_encounter_intervals", root),
             "distribution of inter-encounter intervals, d1: duration, d2: P(full), d3: P(jackknife), d4: error")

    xbin, Pbin, = logbin(full_dist)
    array_xP_logbin(x) = hcat(logbin(distribution_durations(x, edges=full_dist.edges[1]))...)[:,1:2]
    xP, xP_J, xP_err = jackknife_mj(array_xP_logbin, list_dts, naive=hcat(xbin,Pbin));
    myh5write(filename, @sprintf("%s/distribution_inter_encounter_intervals_logbin", root), hcat(xP, xP_J, xP_err))
    myh5desc(filename, @sprintf("%s/distribution_inter_encounter_intervals_logbin", root),
             "distribution of inter-encounter intervals with log binning, d1: <duration>, d2: <P>, d3: <duration>(jackknife), d4: <P>(jackknife), d5: err(<duration>), d6: err(<P>)")


    println("... autocorrelation function of inter-encounter interval")
    c, cj, cerr = jackknife_mj(x->autocorrelation_function(x, lags), list_dts)
    myh5write(filename, @sprintf("%s/autocorrelation_inter_encounter_intervals", root), hcat(lags, c, cj, cerr))
    myh5desc(filename, @sprintf("%s/autocorrelation_inter_encounter_intervals", root),
             "autocorrelation of inter-encounter intervals, d1: k, d2: C(full), d3: C(jackknife), d4: error")


    println("... distribution of cluster sizes")
    minimum_separation = 8*300 # corresponds to roughly mean duration (0.655277 h ~ 8*300 s)
    samples_cluster_sizes = sample_cluster_sizes(list_dts, minimum_separation)
    samples_cluster_sizes = samples_cluster_sizes[length.(samples_cluster_sizes) .> 0]
    full_dist = distribution(samples_cluster_sizes);
    weights_cluster(x) = distribution(x, edges=full_dist.edges[1]).weights
    P, Pj, Perr = jackknife_mj(weights_cluster, samples_cluster_sizes, naive=full_dist.weights);
    myh5write(filename, @sprintf("%s/distribution_encounter_cluster", root), hcat(full_dist.edges[1][1:end-1], P, Pj, Perr))
    myh5desc(filename, @sprintf("%s/distribution_encounter_cluster", root),
             "distribution of contact cluster (total sum of successive activity separated by zero activity), d1: size, d2: P(full), d3: P(jackknife), d4: error")

    xbin, Pbin, = logbin(full_dist)
    array_cluster_logbin(x) = hcat(logbin(distribution(x, edges=full_dist.edges[1]))...)[:,1:2]
    xP, xP_J, xP_err = jackknife_mj(array_cluster_logbin, samples_cluster_sizes, naive=hcat(xbin,Pbin));
    myh5write(filename, @sprintf("%s/distribution_encounter_cluster_logbin", root), hcat(xP, xP_J, xP_err))
    myh5desc(filename, @sprintf("%s/distribution_encounter_cluster_logbin", root),
             "distribution of inter-encounter intervals with log binning, d1: <duration>, d2: <P>, d3: <duration>(jackknife), d4: <P>(jackknife), d5: err(<duration>), d6: err(<P>)")
end


###############################################################################
###############################################################################
### disease spread

function analyse_infectious_encounter_detail(
        disease_model::AbstractDiseaseModel,
        ets::encounter_trains{I,T},
        filename::String,
        root::String;
       ) where {I,T}
    println("... detailed analysis of infectious encounter for model ", disease_model)
    ets_noempty = ets[length.(ets) .> 0]

    # viral load
    println("... ... disease progression")
    time_range = 0:60*60:seconds_from_days(21)
    print("... ... evaluate viral load\n")
    load = viral_load(disease_model, time_range)
    myh5write(filename, @sprintf("%s/viral_load", root), load)
    myh5desc(filename, @sprintf("%s/viral_load", root),
        "probability to be infectious for current disease model, d1: time range, d2: probability")

    # distribution of infectious encounters
    println("... ... distribution infectious encounter")
    try
        data = samples_infectious_encounter(disease_model, ets)
        data_noempty = data[length.(data) .> 0]
        dist = distribution_from_samples_infectious_encounter(data)
        # jackknife_mj because relevant statistics is the number of encounters where infection could start
        weights(x) = distribution_from_samples_infectious_encounter(x, edges=dist.edges[1]).weights

        w, wj, werr = jackknife_mj(weights, data_noempty, naive=dist.weights);
        myh5write(filename, @sprintf("%s/distribution_infectious_encounter", root), hcat(dist.edges[1][1:end-1], w, wj, werr))
        myh5desc(filename, @sprintf("%s/distribution_infectious_encounter", root),
                 "distribution of number of encounters during infectious interval, d1: number of encounters, d2: P(full), d3: P(jack), d4:error")

        o,oj, oerr = jackknife_mj(x->mean_from_samples_infectious_encounter(x), data_noempty)
        myh5write(filename, @sprintf("%s/mean_number_infectious_encounter", root), hcat(o,oj,oerr))
        myh5desc(filename, @sprintf("%s/mean_number_infectious_encounter", root),
                 "mean number of infectious encounter , d1: full, d2: jack, d3: err")
    catch e
        println(e)
    end

    # controls
    println("... ... controls")
    try
        println("... ... ... random disase onset")
        data = samples_infectious_encounter_random_onset(disease_model, ets)
        data_noempty = data[length.(data) .> 0]
        dist = distribution_from_samples_infectious_encounter(data)
        # jackknife_mj because relevant statistics is the number of encounters where infection could start
        weights(x) = distribution_from_samples_infectious_encounter(x, edges=dist.edges[1]).weights
        w, wj, werr = jackknife_mj(weights, data_noempty, naive=dist.weights);
        myh5write(filename, @sprintf("%s/control_random_disease_onset/distribution_infectious_encounter", root), hcat(dist.edges[1][1:end-1], w, wj, werr))
        myh5desc(filename,  @sprintf("%s/control_random_disease_onset/distribution_infectious_encounter", root),
                 "distribution of number of encounters during infectious interval, d1: number of encounters, d2: P(full), d3: P(jack), d4:error")
        o,oj, oerr = jackknife_mj(x->mean_from_samples_infectious_encounter(x), data_noempty)
        myh5write(filename, @sprintf("%s/control_random_disease_onset/mean_number_infectious_encounter", root), hcat(o,oj,oerr))
        myh5desc(filename,  @sprintf("%s/control_random_disease_onset/mean_number_infectious_encounter", root),
                 "mean number of infectious encounter , d1: full, d2: jack, d3: err")
    catch e
        println(e)
    end
    try
        println("... ... ... random disase onset (weighted train)")
        data = samples_infectious_encounter_random_onset(disease_model, ets, weighted_train=true)
        data_noempty = data[length.(data) .> 0]
        dist = distribution_from_samples_infectious_encounter(data)
        # jackknife_mj because relevant statistics is the number of encounters where infection could start
        weights(x) = distribution_from_samples_infectious_encounter(x, edges=dist.edges[1]).weights
        w, wj, werr = jackknife_mj(weights, data_noempty, naive=dist.weights);
        myh5write(filename, @sprintf("%s/control_random_disease_onset_wtrain/distribution_infectious_encounter", root), hcat(dist.edges[1][1:end-1], w, wj, werr))
        myh5desc(filename,  @sprintf("%s/control_random_disease_onset_wtrain/distribution_infectious_encounter", root),
                 "distribution of number of encounters during infectious interval, d1: number of encounters, d2: P(full), d3: P(jack), d4:error")
        o,oj, oerr = jackknife_mj(x->mean_from_samples_infectious_encounter(x), data_noempty)
        myh5write(filename, @sprintf("%s/control_random_disease_onset_wtrain/mean_number_infectious_encounter", root), hcat(o,oj,oerr))
        myh5desc(filename,  @sprintf("%s/control_random_disease_onset_wtrain/mean_number_infectious_encounter", root),
                 "mean number of infectious encounter , d1: full, d2: jack, d3: err")
    catch e
        println(e)
    end
    try
        println("... ... ... random disase onset (weighted time)")
        data = samples_infectious_encounter_random_onset(disease_model, ets, weighted_time=true)
        data_noempty = data[length.(data) .> 0]
        dist = distribution_from_samples_infectious_encounter(data)
        # jackknife_mj because relevant statistics is the number of encounters where infection could start
        weights(x) = distribution_from_samples_infectious_encounter(x, edges=dist.edges[1]).weights
        w, wj, werr = jackknife_mj(weights, data_noempty, naive=dist.weights);
        myh5write(filename, @sprintf("%s/control_random_disease_onset_wtime/distribution_infectious_encounter", root), hcat(dist.edges[1][1:end-1], w, wj, werr))
        myh5desc(filename,  @sprintf("%s/control_random_disease_onset_wtime/distribution_infectious_encounter", root),
                 "distribution of number of encounters during infectious interval, d1: number of encounters, d2: P(full), d3: P(jack), d4:error")
        o,oj, oerr = jackknife_mj(x->mean_from_samples_infectious_encounter(x), data_noempty)
        myh5write(filename, @sprintf("%s/control_random_disease_onset_wtime/mean_number_infectious_encounter", root), hcat(o,oj,oerr))
        myh5desc(filename,  @sprintf("%s/control_random_disease_onset_wtime/mean_number_infectious_encounter", root),
                 "mean number of infectious encounter , d1: full, d2: jack, d3: err")
    catch e
        println(e)
    end
    try
        println("... ... ... random disase onset (weighted train/time)")
        data = samples_infectious_encounter_random_onset(disease_model, ets, weighted_time=true, weighted_train=true)
        data_noempty = data[length.(data) .> 0]
        dist = distribution_from_samples_infectious_encounter(data)
        # jackknife_mj because relevant statistics is the number of encounters where infection could start
        weights(x) = distribution_from_samples_infectious_encounter(x, edges=dist.edges[1]).weights
        w, wj, werr = jackknife_mj(weights, data_noempty, naive=dist.weights);
        myh5write(filename, @sprintf("%s/control_random_disease_onset_wtrain_wtime/distribution_infectious_encounter", root), hcat(dist.edges[1][1:end-1], w, wj, werr))
        myh5desc(filename,  @sprintf("%s/control_random_disease_onset_wtrain_wtime/distribution_infectious_encounter", root),
                 "distribution of number of encounters during infectious interval, d1: number of encounters, d2: P(full), d3: P(jack), d4:error")
        o,oj, oerr = jackknife_mj(x->mean_from_samples_infectious_encounter(x), data_noempty)
        myh5write(filename, @sprintf("%s/control_random_disease_onset_wtrain_wtime/mean_number_infectious_encounter", root),  hcat(o,oj,oerr))
        myh5desc(filename,  @sprintf("%s/control_random_disease_onset_wtrain_wtime/mean_number_infectious_encounter", root),
                 "mean number of infectious encounter , d1: full, d2: jack, d3: err")
    catch e
        println(e)
    end
    try
        println("... ... ... disease inset from encounters of first week only")
        data = samples_infectious_encounter_with_infection_in_early_interval(disease_model, ets, seconds_from_days(7))
        data_noempty = data[length.(data) .> 0]
        dist = distribution_from_samples_infectious_encounter(data)
        # jackknife_mj because relevant statistics is the number of encounters where infection could start
        weights(x) = distribution_from_samples_infectious_encounter(x, edges=dist.edges[1]).weights
        w, wj, werr = jackknife_mj(weights, data_noempty, naive=dist.weights);
        myh5write(filename, @sprintf("%s/control_sample_first_week_only/distribution_infectious_encounter", root), hcat(dist.edges[1][1:end-1], w, wj, werr))
        myh5desc(filename,  @sprintf("%s/control_sample_first_week_only/distribution_infectious_encounter", root),
                 "distribution of number of encounters during infectious interval, d1: number of encounters, d2: P(full), d3: P(jack), d4:error")
        o,oj, oerr = jackknife_mj(x->mean_from_samples_infectious_encounter(x), data_noempty)
        myh5write(filename, @sprintf("%s/control_sample_first_week_only/mean_number_infectious_encounter", root),  hcat(o,oj,oerr))
        myh5desc(filename,  @sprintf("%s/control_sample_first_week_only/mean_number_infectious_encounter", root),
                 "mean number of infectious encounter , d1: full, d2: jack, d3: err")
    catch e
        println(e)
    end
end

function analyse_infectious_encounter_scan_delta(
        range_latent_in_days,
        range_infectious_in_days,
        ets::encounter_trains{I,T1},
        filename::String,
        root::String;
        # optional
        seed=1000,
       ) where {I,T1, T2}
    print("... scan mean number infectious encounters for delta disease\n")

    # scan
    results  = Array{Float64,2}(undef,length(range_latent_in_days),length(range_infectious_in_days))
    @showprogress 1 for (i, latent) in enumerate(range_latent_in_days)
        for (j, infectious) in enumerate(range_infectious_in_days)
            mean_ets = mean_infectious_encounter(DeltaDiseaseModel(seconds_from_days(latent), seconds_from_days(infectious)), ets)
            results[i,j] = mean_ets
        end
    end

    myh5write(filename, @sprintf("%s/scan_mean_number_infectious_encounter/mean", root), results)
    myh5write(filename, @sprintf("%s/scan_mean_number_infectious_encounter/range_latent", root), collect(range_latent_in_days))
    myh5write(filename, @sprintf("%s/scan_mean_number_infectious_encounter/range_infectious", root), collect(range_infectious_in_days))
    myh5desc(filename, @sprintf("%s/scan_mean_number_infectious_encounter", root),
        "scan of the mean number of infectious encounters for different latent and infectious periods, d1: latent period, d2: infectious period")
end

function analyse_infectious_encounter_scan_gamma(
        params_in_days,
        range_k,
        ets::encounter_trains{I,T},
        filename::String,
        root::String;
        # optional
        samples = Int(1e5)
    ) where {I,T}
    println("... scan k-dependence for gamma disease model with ", params_in_days)
    latent=getindex(params_in_days, 1)
    infectious=getindex(params_in_days, 2)

    # scan
    o = Vector{Float64}(undef,length(range_k))
    oj = Vector{Float64}(undef,length(range_k))
    oerr  = Vector{Float64}(undef,length(range_k))
    p = Progress(length(range_k), 1, "k-values: ", offset=0)
    for (i, k) in enumerate(range_k)
        next!(p; showvalues = [(:i,i), (:k,k)])
        # next!(p)
        model = GammaDiseaseModel(seconds_from_days(latent), seconds_from_days(infectious), k)
        rng = MersenneTwister(1000)
        try
            #first need to generate data (not mean infectious encounter but all samples of infectious encounter)
            data = samples_infectious_encounter(model, ets)
            data_noempty = data[length.(data).>0]
            o[i], oj[i], oerr[i] = jackknife_mj(x->mean(vcat(x...)), data_noempty)
        catch
            o[i] = oj[i] = oerr[i] = NaN
        end
    end

    myh5write(filename, @sprintf("%s/scan_k", root), hcat(collect(range_k), o, oj, oerr))
    myh5desc(filename, @sprintf("%s/scan_k", root),
             "k dependence of the mean number of infectious encounters for gamma-distributied disease models (parameters in dsets), d1: k, d2: mean(full), d3: mean(jack), d4: mean(err)")
end


function analyse_dispersion_scan_delta(
        range_latent_in_days,
        range_infectious_in_days,
        range_R0,
        ets::encounter_trains{I,T1},
        filename::String,
        root::String;
        # optional
        seed_samples=1000,
        num_samples=Int(1e3)
    ) where {I,T1, T2}
    result_r  = Array{Float64,3}(undef,
                                 length(range_latent_in_days),
                                 length(range_infectious_in_days),
                                 length(range_R0)
                                )
    result_p  = Array{Float64,3}(undef,
                                 length(range_latent_in_days),
                                 length(range_infectious_in_days),
                                 length(range_R0)
                                )
    print("... scan offpsring dispersion for delta disease\n")
    @showprogress 1 for (i, latent) in enumerate(range_latent_in_days)
        for (j, infectious) in enumerate(range_infectious_in_days)
            disease_model = DeltaDiseaseModel(seconds_from_days(latent),
                                              seconds_from_days(infectious));
            edist = EmpiricalDistribution(
                distribution_from_samples_infectious_encounter(
                    samples_infectious_encounter(disease_model, ets)
                )
            );
            for (k, R0) in enumerate(range_R0)
                # println(" ... ... ", latent, " ", infectious, " ", R0)
                p_inf = R0/expectation(edist)
                if p_inf > 1
                    result_r[i,j,k], result_p[i,j,k] = (NaN,NaN)
                else
                    offspring_dist = offspring_distribution(edist,p_inf);
                    try
                        NB = fit_mle_negative_binomial(MersenneTwister(seed_samples), offspring_dist);
                        result_r[i,j,k], result_p[i,j,k] = params(NB)
                    catch e
                        result_r[i,j,k], result_p[i,j,k] = (NaN,NaN)
                    end
                end
            end
        end
    end

    myh5write(filename, @sprintf("%s/scan_offspring_as_negative_binomial/NB_r", root), result_r)
    myh5write(filename, @sprintf("%s/scan_offspring_as_negative_binomial/NB_p", root), result_p)
    myh5write(filename, @sprintf("%s/scan_offspring_as_negative_binomial/range_R0", root), collect(range_R0))
    myh5write(filename, @sprintf("%s/scan_offspring_as_negative_binomial/range_latent", root), collect(range_latent_in_days))
    myh5write(filename, @sprintf("%s/scan_offspring_as_negative_binomial/range_infectious", root), collect(range_infectious_in_days))
    myh5desc(filename, @sprintf("%s/scan_offspring_as_negative_binomial", root),
             "scan over different latent periods, nfectious periods, and R0 to fit
             the offspring distribution with a negative biomial (r,p), d1: latent period, d2: infectious period, d3: R0")
    return
end


function analyse_survival_scan_delta(
        range_latent_in_days,
        range_infectious_in_days,
        range_R0,
        ets::encounter_trains{I,T1},
        filename::String,
        root::String;
    ) where {I,T1, T2}
    result_r  = Array{Float64,3}(undef,
                                 length(range_latent_in_days),
                                 length(range_infectious_in_days),
                                 length(range_R0)
                                )
    result  = Array{Float64,3}(undef,
                                 length(range_latent_in_days),
                                 length(range_infectious_in_days),
                                 length(range_R0)
                                )
    print("... scan survival probabilities for delta disease (this may take a while)\n")
    @showprogress 1 for (i, latent) in enumerate(range_latent_in_days)
        for (j, infectious) in enumerate(range_infectious_in_days)
            disease_model = DeltaDiseaseModel(seconds_from_days(latent),
                                              seconds_from_days(infectious));
            edist = EmpiricalDistribution(
                distribution_from_samples_infectious_encounter(
                    samples_infectious_encounter(disease_model, ets)
                )
            );
            for (k, R0) in enumerate(range_R0)
                p_inf = R0/expectation(edist)
                if p_inf > 1
                    result[i,j,k] = NaN
                else
                    try
                        result[i,j,k] = solve_survival_probability(edist, p_inf)
                    catch e
                        result[i,j,k] = NaN
                    end
                end
            end
        end
    end

    myh5write(filename, @sprintf("%s/scan_survival_probability/survival_probability", root), result)
    myh5write(filename, @sprintf("%s/scan_survival_probability/range_R0", root), collect(range_R0))
    myh5write(filename, @sprintf("%s/scan_survival_probability/range_latent", root), collect(range_latent_in_days))
    myh5write(filename, @sprintf("%s/scan_survival_probability/range_infectious", root), collect(range_infectious_in_days))
    myh5desc(filename, @sprintf("%s/scan_survival_probability", root),
             "scan over different latent periods, infectious periods, and R0.
             d1: latent period, d2: infectious period, d3: R0")
    return
end

###############################################################################
###############################################################################
### quick default
analyse_all() = analyse_all(Copenhagen())



