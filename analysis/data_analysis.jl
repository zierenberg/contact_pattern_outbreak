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
        cluster_minimum_separation = 15*60,
        disease_model = DiseaseDefault(),
        seed=1000,
        # 0: essentials, 1: essentials and fast extras, 2: most, 3: everything
        # to skip jackknife analysis set gloibal variable `skip_jackknife = true`
        level_of_details = 2,
    )
    lod = level_of_details
    if filter_out_incomplete
        filename = @sprintf("%s/results_%s_filtered_%dmin_filterOutIncomplete.h5", path_out, label(experiment), minimum_duration/60)
    else
        filename = @sprintf("%s/results_%s_filtered_%dmin.h5", path_out, label(experiment), minimum_duration/60)
    end

    ###########################################################################
    # data
    # experiment=Copenhagen(); minimum_duration=15*60; path_dat="./dat";filter_out_incomplete=false;
    cas, ets, list_contacts = load_processed_data(experiment, minimum_duration, path_dat, filter_out_incomplete=filter_out_incomplete);
    # write encounter train
    lod >= 1 && myh5write(filename,"/data/trains/",ets)

    if experiment == InVS15()
        support_crate=0:300:seconds_from_days(7*1.5)
    else
        support_crate=0:timestep(ets):seconds_from_days(7*1.5)
    end

    ###########################################################################
    # temporal features

    lod >= 2 && analyse_temporal_features_of_contact_activity(cas, filename, "/data/contact_activity")
    lod >= 1 && analyse_contact_duration(list_contacts, experiment, filename, "/data/contacts")
    lod >= 1 && analyse_temporal_features_of_encounter_train(ets, filename, "/data/encounter_train", support_crate=support_crate)

    ###########################################################################
    # surrogate data and sampling


    # disease periods
    range_latent = 0:0.05:8
    range_infectious = 0.05:0.05:8

    if lod >= 1
        root="/data_surrogate_randomize_per_train"
        sur_rand_train = surrogate_randomize_per_train(ets, seed)
        ref = sur_rand_train
        myh5write(filename,@sprintf("%s/trains/",root), sur_rand_train)
        analyse_temporal_features_of_encounter_train(sur_rand_train, filename, @sprintf("%s/encounter_train",root), support_crate=support_crate)
        analyse_infectious_encounter_scan_delta(range_latent, range_infectious, sur_rand_train, ref, filename, @sprintf("%s/disease/delta", root))
    end

    if lod >= 2
        # surrogate version 2
        root="/data_surrogate_randomize_all"
        sur_rand_all = surrogate_randomize_all(ets, seed)
        myh5write(filename,@sprintf("%s/trains/",root), sur_rand_all)
        analyse_temporal_features_of_encounter_train(sur_rand_all, filename, @sprintf("%s/encounter_train",root), support_crate=support_crate)
        analyse_infectious_encounter_scan_delta(range_latent, range_infectious, sur_rand_all, ref, filename, @sprintf("%s/disease/delta", root))
    end

    # surrogates from sampling
    if lod >= 3
        num_sample_trains = length(ets)
        # weights for trains
        train_weights = length.(ets) ./ mean(length.(ets))

        # (inhomogeneous) poisson processes
        contact_rate = rate(ets, 0:timestep(ets):seconds_from_days(7))
        mean_rate = mean(contact_rate.weights)

        root="sample/poisson_homogeneous/"
        samples = sample_encounter_trains_poisson(mean_rate, num_sample_trains, -seconds_from_days(7), (0, duration(ets)), MersenneTwister(seed), timestep=timestep(ets))
        analyse_temporal_features_of_encounter_train(samples, filename, root, support_crate=support_crate)
        analyse_infectious_encounter_scan_delta(range_latent, range_infectious, samples, ref, filename, @sprintf("%s/disease/delta", root))

        root="sample/poisson_homogeneous_weighted_trains/"
        samples = sample_encounter_trains_poisson(mean_rate, num_sample_trains, -seconds_from_days(7), (0, duration(ets)), MersenneTwister(seed), timestep=timestep(ets), weights=train_weights)
        analyse_temporal_features_of_encounter_train(samples, filename, root, support_crate=support_crate)
        analyse_infectious_encounter_scan_delta(range_latent, range_infectious, samples, ref, filename, @sprintf("%s/disease/delta", root))

        root="sample/poisson_inhomogeneous/"
        samples = sample_encounter_trains_poisson(contact_rate, num_sample_trains, -seconds_from_days(7), (0, duration(ets)), MersenneTwister(seed), timestep=timestep(ets))
        analyse_temporal_features_of_encounter_train(samples, filename, root, support_crate=support_crate)
        analyse_infectious_encounter_scan_delta(range_latent, range_infectious, samples, ref, filename, @sprintf("%s/disease/delta", root))

        root="sample/poisson_inhomogeneous_weighted_trains/"
        samples = sample_encounter_trains_poisson(contact_rate, num_sample_trains, -seconds_from_days(7), (0, duration(ets)), MersenneTwister(seed), timestep=timestep(ets), weights=train_weights)
        analyse_temporal_features_of_encounter_train(samples, filename, root, support_crate=support_crate)
        analyse_infectious_encounter_scan_delta(range_latent, range_infectious, samples, ref, filename, @sprintf("%s/disease/delta", root))

        # weibull renewal processes
        list_dts = inter_encounter_intervals(ets);
        list_dts = list_dts[length.(list_dts) .> 0];
        dist = distribution_durations(list_dts, timestep=timestep(ets));
        args_weibull = fit_Weibull(dist)

        # weibull renewal
        root="sample/weibul_renewal_process/"
        samples = sample_encounter_trains_weibull_renewal(args_weibull, num_sample_trains, -seconds_from_days(7), (0, duration(ets)), MersenneTwister(seed), timestep=timestep(ets))
        analyse_temporal_features_of_encounter_train(samples, filename, root, support_crate=support_crate)
        analyse_infectious_encounter_scan_delta(range_latent, range_infectious, samples, ref, filename, @sprintf("%s/disease/delta", root))

        # weibull renewal with weighted trains
        root="sample/weibul_renewal_process_weighted_trains/"
        samples = sample_encounter_trains_weibull_renewal(args_weibull, num_sample_trains, -seconds_from_days(7), (0, duration(ets)), MersenneTwister(seed), timestep=timestep(ets), weights=train_weights)
        analyse_temporal_features_of_encounter_train(samples, filename, root, support_crate=support_crate)
        analyse_infectious_encounter_scan_delta(range_latent, range_infectious, samples, ref, filename, @sprintf("%s/disease/delta", root))
    end


    ###########################################################################
    # disease spread
    if lod >= 1
        analyse_infectious_encounter_scan_delta(range_latent, range_infectious, ets, ref, filename, "/disease/delta")
        #details
        analyse_infectious_encounter_detail(DeltaDiseaseModel(seconds_from_days(2), seconds_from_days(3)), ets, filename, "/disease/delta_2_3")
        analyse_infectious_encounter_detail(DeltaDiseaseModel(seconds_from_days(6), seconds_from_days(3)), ets, filename, "/disease/delta_6_3")
        analyse_infectious_encounter_detail(DeltaDiseaseModel(seconds_from_days(2), seconds_from_days(3)), ref, filename, "/disease/delta_2_3_surrogate")
        analyse_infectious_encounter_detail(DeltaDiseaseModel(seconds_from_days(6), seconds_from_days(3)), ref, filename, "/disease/delta_6_3_surrogate")
        # details at different points
        analyse_infectious_encounter_detail(DeltaDiseaseModel(seconds_from_days(1), seconds_from_days(0.5)), ets, filename, "/disease/delta_1_0.5")
        analyse_infectious_encounter_detail(DeltaDiseaseModel(seconds_from_days(1.5), seconds_from_days(0.5)), ets, filename, "/disease/delta_1.5_0.5")
        analyse_infectious_encounter_detail(DeltaDiseaseModel(seconds_from_days(1), seconds_from_days(0.5)), ref, filename, "/disease/delta_1_0.5_surrogate")
        analyse_infectious_encounter_detail(DeltaDiseaseModel(seconds_from_days(1.5), seconds_from_days(0.5)), ref, filename, "/disease/delta_1.5_0.5_surrogate")
    end

    #gamma - scan
    if lod >= 1
        range_k = 10 .^ collect(0:0.2:5)
        analyse_infectious_encounter_scan_gamma((2,3), range_k, ets, filename, "/disease/gamma_2_3")
        analyse_infectious_encounter_scan_gamma((6,3), range_k, ets, filename, "/disease/gamma_6_3")
        analyse_infectious_encounter_scan_gamma((2,3), range_k, ref, filename, "/disease/gamma_2_3_surrogate")
        analyse_infectious_encounter_scan_gamma((6,3), range_k, ref, filename, "/disease/gamma_6_3_surrogate")
        #gamma - detail
        analyse_infectious_encounter_detail(GammaDiseaseModel(seconds_from_days(2), seconds_from_days(3), 1), ets, filename, "/disease/gamma_2_3/k_1.0")
        analyse_infectious_encounter_detail(GammaDiseaseModel(seconds_from_days(6), seconds_from_days(3), 1), ets, filename, "/disease/gamma_6_3/k_1.0")
        analyse_infectious_encounter_detail(GammaDiseaseModel(seconds_from_days(2), seconds_from_days(3), 10), ets, filename, "/disease/gamma_2_3/k_10.0")
        analyse_infectious_encounter_detail(GammaDiseaseModel(seconds_from_days(6), seconds_from_days(3), 10), ets, filename, "/disease/gamma_6_3/k_10.0")
        # gamma - detail, surrogate
        analyse_infectious_encounter_detail(GammaDiseaseModel(seconds_from_days(2), seconds_from_days(3), 1), ref, filename, "/disease/gamma_2_3_surrogate/k_1.0")
        analyse_infectious_encounter_detail(GammaDiseaseModel(seconds_from_days(6), seconds_from_days(3), 1), ref, filename, "/disease/gamma_6_3_surrogate/k_1.0")
        analyse_infectious_encounter_detail(GammaDiseaseModel(seconds_from_days(2), seconds_from_days(3), 10), ref, filename, "/disease/gamma_2_3_surrogate/k_10.0")
        analyse_infectious_encounter_detail(GammaDiseaseModel(seconds_from_days(6), seconds_from_days(3), 10), ref, filename, "/disease/gamma_6_3_surrogate/k_10.0")
    end

    #gamma - scan, more unrealistic points
    if lod >= 1
        range_k = 10 .^ collect(0:0.2:5)
        analyse_infectious_encounter_scan_gamma((1,0.5), range_k, ets, filename, "/disease/gamma_1_0.5")
        analyse_infectious_encounter_scan_gamma((1.5,0.5), range_k, ets, filename, "/disease/gamma_1.5_0.5")
        analyse_infectious_encounter_scan_gamma((1,0.5), range_k, ref, filename, "/disease/gamma_1_0.5_surrogate")
        analyse_infectious_encounter_scan_gamma((1.5,0.5), range_k, ref, filename, "/disease/gamma_1.5_0.5_surrogate")
        #gamma - detail
        analyse_infectious_encounter_detail(GammaDiseaseModel(seconds_from_days(1), seconds_from_days(0.5), 1), ets, filename, "/disease/gamma_1_0.5/k_1.0")
        analyse_infectious_encounter_detail(GammaDiseaseModel(seconds_from_days(1.5), seconds_from_days(0.5), 1), ets, filename, "/disease/gamma_1.5_0.5/k_1.0")
        analyse_infectious_encounter_detail(GammaDiseaseModel(seconds_from_days(1), seconds_from_days(0.5), 10), ets, filename, "/disease/gamma_1_0.5/k_10.0")
        analyse_infectious_encounter_detail(GammaDiseaseModel(seconds_from_days(1.5), seconds_from_days(0.5), 10), ets, filename, "/disease/gamma_1.5_0.5/k_10.0")
        # gamma - detail, surrogate
        analyse_infectious_encounter_detail(GammaDiseaseModel(seconds_from_days(1), seconds_from_days(0.5), 1), ref, filename, "/disease/gamma_1_0.5_surrogate/k_1.0")
        analyse_infectious_encounter_detail(GammaDiseaseModel(seconds_from_days(1.5), seconds_from_days(0.5), 1), ref, filename, "/disease/gamma_1.5_0.5_surrogate/k_1.0")
        analyse_infectious_encounter_detail(GammaDiseaseModel(seconds_from_days(1), seconds_from_days(0.5), 10), ref, filename, "/disease/gamma_1_0.5_surrogate/k_10.0")
        analyse_infectious_encounter_detail(GammaDiseaseModel(seconds_from_days(1.5), seconds_from_days(0.5), 10), ref, filename, "/disease/gamma_1.5_0.5_surrogate/k_10.0")
    end

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
    println("writing to ", root)
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
    println("writing to ", root)
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
        ############ optional
        lags=0:1:200,                                         # lags for autocorrelation
        times_rate=0:60*60:seconds_from_days(7),        # 1-week contact rate in steps of hours
        support_crate=0:timestep(ets):seconds_from_days(7*1.5),# 1.5-week conditional contact rate

    ) where {I,T}
    println("writing to ", root)

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
        ref::encounter_trains{I,T2},
        filename::String,
        root::String;
        # optional
        seed=1000,
       ) where {I,T1, T2}
    print("... scan mean number infectious encounters for delta disease\n")

    # scan
    results  = Array{Float64,2}(undef,length(range_latent_in_days),length(range_infectious_in_days))
    relative = Array{Float64,2}(undef,length(range_latent_in_days),length(range_infectious_in_days))
    @showprogress 1 for (i, latent) in enumerate(range_latent_in_days)
        for (j, infectious) in enumerate(range_infectious_in_days)
            mean_ets = mean_infectious_encounter(DeltaDiseaseModel(seconds_from_days(latent), seconds_from_days(infectious)), ets)
            mean_ref = mean_infectious_encounter(DeltaDiseaseModel(seconds_from_days(latent), seconds_from_days(infectious)), ref)
            results[i,j] = mean_ets
            relative[i,j] = mean_ets/mean_ref
        end
    end

    myh5write(filename, @sprintf("%s/scan_mean_number_infectious_encounter/mean", root), results)
    myh5write(filename, @sprintf("%s/scan_mean_number_infectious_encounter/mean_relative_to_poisson", root), relative)
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


###############################################################################
###############################################################################
### sample disease spread with generative model that captures some features of data
### and analyse effective R from the time course of new cases.

function analyse_effective_R(filename::String;
    path_out = "./out",
    dsetname::String="measurements")
    infectious = 3
    h5open(filename, "r") do file
        encounter_rate = read(file[dsetname], "encounter_rate")
        cases = file[dsetname]["cases"]
        keys_latent = keys(cases)
        keys_latent = keys_latent[occursin.("latent", keys_latent)]
        list_latent = parse.(Float64, last.(split.(keys_latent, "_")))
        # go through hdf5 file
        R4_avg = zeros(length(keys_latent))
        R4_std = zeros(length(keys_latent))
        Rg_avg = zeros(length(keys_latent))
        Rg_std = zeros(length(keys_latent))
        R0_avg = zeros(length(keys_latent))
        R0_std = zeros(length(keys_latent))
        Tg = zeros(length(keys_latent))
        for (l, key_latent) in enumerate(keys_latent)
            println("\n", key_latent, " ", list_latent[l])
            #todo: string manipulation
            disease_model = DeltaDiseaseModel(seconds_from_days(list_latent[l]), seconds_from_days(infectious))

            # calculate generation time as the expectation value over infection times
            interval_infectious  = infectious_interval(disease_model)
            mask_infectious = interval_infectious[1] .< encounter_rate[:,1] .< interval_infectious[2];
            time_generation = sum(encounter_rate[mask_infectious, 1].*encounter_rate[mask_infectious,2])/sum(encounter_rate[mask_infectious, 2])
            time_generation /= seconds_from_days(1)
            keys_seed = keys(cases[key_latent])
            mean_R4 = zeros(length(keys_seed))
            mean_Rg = zeros(length(keys_seed))
            mean_R0 = zeros(Union{Float64, Missing}, length(keys_seed))
            @showprogress 1 for (s, seed) in enumerate(keys_seed)
                println("...", seed)
                measurement = read(file, @sprintf("%s/cases/%s/%s/", dsetname, key_latent, seed))
                measurement = measurement[:, 2]
                r0, stat = read(file, @sprintf("%s/R0/%s/%s/", dsetname, key_latent, seed))
                mean_R0[s] = r0

                # find left border where cases >= some threshold
                large_weights = findall(measurement[:].>1000)
                if length(large_weights) < 1
                    mean_Rg[s] = NaN
                    mean_R4[s] = NaN
                    mean_R0[s] = missing
                    print("no data qualified, skipping ", r0, "\n")
                    continue
                end

                # include inbetween points where fluctuations drive case numbers below thresh
                valid_range = large_weights[1]:large_weights[end]
                # average R estimage across 3 days
                a = 3
                # time shift new vs old, first use estimated generation time from data
                g = round(Int64, time_generation)
                println("average over: ", a, " range from, to: ", valid_range, " time shift: ", g)

                if length(valid_range) <= g+a
                    mean_Rg[s] = NaN
                else
                    # calculate R with local (`a`-long) sums for every time point
                    R_g = zeros(length(valid_range)-g-a)
                    for (i,t) in enumerate(valid_range[1]:valid_range[end]-g-a)
                        R_g[i] = sum(measurement[t+g : t+g+a]) / sum(measurement[t : t+a])
                    end
                    # and average across all time points
                    mean_Rg[s] = mean(R_g)
                end

                # repeat for conventional g
                g = 4
                if length(valid_range) <= g+a
                    mean_R4[s] = NaN
                else
                    R_4 = zeros(length(valid_range)-g-a)
                    for (i,t) in enumerate(valid_range[1]:valid_range[end]-g-a)
                        R_4[i] = sum(measurement[t+g : t+g+a]) / sum(measurement[t : t+a])
                    end
                    mean_R4[s] = mean(R_4)
                end

                println("R0, Rg, R4: ", mean_R0[s], " ", mean_Rg[s], " ", mean_R4[s])
            end
            # l is latent period
            # mean_and_std returns nan if any element is nan.
            R4_avg[l], R4_std[l] = mean_and_std(mean_R4)
            Rg_avg[l], Rg_std[l] = mean_and_std(mean_Rg)
            R0_avg[l], R0_std[l] = mean_and_std(skipmissing(mean_R0))
            Tg[l] = time_generation
        end

        # store results somewhere
        open(@sprintf("%s/analysis_mean-field_R_%s.dat", path_out, dsetname), "w") do io
            write(io, "#latent\t avg(R4)\t std(R4)\t avg(Rg)\t std(Rg)\t avg(R0)\t std(R0)\t g\n")
            writedlm(io, zip(list_latent, R4_avg, R4_std, Rg_avg, Rg_std, R0_avg, R0_std, Tg))
        end
    end
end

function sample_mean_field_for_effective_R(;
        #optional
        experiment=Copenhagen(),
        minimum_duration = 15*60,
        path_dat = "./dat",
        path_out = "./out",
    )

    function do_it(ets, filename, dsetname)
        # encounter rate
        encounter_rate = conditional_encounter_rate(ets, 0:timestep(ets):seconds_from_days(8+4))
        myh5write(filename, @sprintf("/%s/encounter_rate", dsetname),
            hcat(encounter_rate.edges[1][1:end-1], encounter_rate.weights))
        myh5desc(filename, @sprintf("/%s/encounter_rate", dsetname),
            "average encounter rate conditioned on having an encounter at time 0, averaged across all encounter in experiment, d1: times, d2: rate(full)")

        probability_infection = 0.12 # heuristically chosen to match R approx 3.3 for infectious=3 and latent=4 when g=4
        infectious = 3 # days
        range_latent = 0.0:0.5:8
        for (l, latent) in enumerate(range_latent)
            println(latent)
            disease_model = DeltaDiseaseModel(seconds_from_days(latent), seconds_from_days(infectious))
            #dist_offspring = distribution_from_samples_infectious_encounter(samples_infectious_encounter(disease_model, ets))

            # sample
            rng = MersenneTwister(1000)
            range_seeds = 1000:1010
            @showprogress 1 for (s, seed) in enumerate(range_seeds)
                println("...", seed)
                I0 = floor(Int, 1 + latent)
                measurement, sum_offsprings, sum_samples = spread_mean_field(disease_model, encounter_rate, probability_infection, seconds_from_days.(0:120.), max_cases=5e6, seed=seed, initial_infection_times=rand(rng,I0)*infectious)

                myh5write(filename, @sprintf("/%s/cases/latent_%.2f/%d", dsetname, latent, seed), hcat(measurement.edges[1][1:end-1], measurement.weights))
                myh5write(filename, @sprintf("/%s/R0/latent_%.2f/%d", dsetname, latent, seed), [sum_offsprings/sum_samples, sum_samples])
            end
        end
    end

    # use real data
    filename = @sprintf("%s/mean_field_samples_%s_filtered_%dmin.h5", path_out, label(experiment), minimum_duration/60)
    cas, ets, list_contacts = load_processed_data(experiment, minimum_duration, path_dat);
    do_it(ets, filename, "measurements")

    # use randomized data
    ets = surrogate_randomize_per_train(ets, 1000)
    do_it(ets, filename, "measurements_randomized_per_train")
end


###############################################################################
###############################################################################
### quick default
analyse_all() = analyse_all(Copenhagen())



