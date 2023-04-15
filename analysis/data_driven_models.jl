using HDF5
using Printf
using Statistics
using StatsBase
using LinearAlgebra
using IterTools

using ProgressMeter

include("load_data.jl")
include("write_files.jl")
include("disease_spread.jl")
include("branching_process.jl")
include("surrogate.jl")
include("sampling.jl")
include("temporal_features.jl")
include("data_analysis.jl")

###############################################################################
###############################################################################
###############################################################################
# data-driven surrogate point processes

"""
    sample_surrogate_tailored()

generate samples of high-rate Weibull renewal processes, where encounter are
accepted with probability to match time-dependent rate. Weibull parameters are
obtained from minimal Kullback-Leibler divergence between empirical iei-dist
and sample iei-dist tails.

experiment=Copenhagen(); minimum_duration=15*60; path_dat="./dat"
"""
function sample_surrogate_tailored(
        experiment=Copenhagen(),
        minimum_duration = 15*60,
        path_dat = "./dat",
        path_out = "./out",
        num_samples = 20,
        support_crate=0:300:seconds_from_days(7*1.5),
        range_latent = default_range_latent,
        range_infectious = default_range_infectious,
    )
    filename = @sprintf("%s/surrogate_tailored_%s_filtered_%dmin.h5", path_out, label(experiment), minimum_duration/60)
    _,ets, _ = load_processed_data(experiment, minimum_duration, path_dat);
    filename_rand = @sprintf("%s/surrogate_tailored_randomized_per_train_%s_filtered_%dmin.h5",
        path_out, label(experiment), minimum_duration / 60)

    time_start = -seconds_from_days(7)
    interval_record = (0,seconds_from_days(28))

    # specificy features that need to be reproducde
    # number of trains:
    num_sample_trains = length(ets)
    # relative rate (prop. to number of encounters) of each train compared to mean:
    train_weights = length.(ets) ./ mean(length.(ets))
    # time-dependent global rate:
    ref_rate = rate(ets, 0:timestep(ets):seconds_from_days(7))
    # mean global rate:
    ref_rate_mean = mean(ref_rate.weights)
    # reference distribution of inter-encounter intervals
    edges = timestep(ets)*(150:1:1500)
    ref_dist = distribution_durations(inter_encounter_intervals(ets), edges=edges);

    # precompute internals for loop
    ref_rate_max = maximum(ref_rate.weights)
    time_prob = deepcopy(ref_rate)
    time_prob.weights /= ref_rate_max

    println("""
            Sweep over shape parameter to determine optimal parameters for
            small KL-divergence between inter-encounter-interval distributions
            from sample and data
            """)
    range_shape = collect(0.1:0.01:1.0)
    range_kld1 = zeros(length(range_shape))
    range_kld2 = zeros(length(range_shape))
    @showprogress 1 for (i, _shape) in enumerate(range_shape)
        for j in 1:num_samples
            #for reproducibilty, always start from the same seed for different
            #parameters that will immediately result in uncorrelated encounter
            #trains because the underlying distribution is different.
            rng = MersenneTwister(1000+j)
            test_ets,_ = sample_encounter_trains_tailored(rng, _shape,
                                                        ref_rate_max,
                                                        time_prob,
                                                        train_weights,
                                                        time_start,
                                                        interval_record,
                                                        timestep(ets))

            test_dist = distribution_durations(
                                    inter_encounter_intervals(test_ets),
                                    edges=ref_dist.edges[1]
                                    )
            kld = kldivergence(test_dist, ref_dist)
            range_kld1[i] += kld
            range_kld2[i] += kld^2
       end
       range_kld1[i] /= num_samples
       range_kld2[i] /= num_samples
    end
    range_err = sqrt.((range_kld2 - range_kld1.^2)/num_samples)
    datasetname=@sprintf("/scan_parameter_shape")
    myh5write(filename, datasetname, hcat(range_shape, range_kld1, range_err))
    myh5desc(filename, datasetname, @sprintf(
             "Average Kullback-Leibler divergence (kld) from %d samples as a function of the single
             free parameter (shape) ,
             d1: shape,
             d2: kld,
             d3: err(kld)",
             num_samples)
            )

    println("Find best shape parameter from the minimum kl-divergence")
    best_shape = range_shape[argmin(range_kld1)]
    rng = MersenneTwister(1000)
    samples, best_scale =  sample_encounter_trains_tailored(rng, best_shape,
                                                             ref_rate_max,
                                                             time_prob,
                                                             train_weights,
                                                             time_start,
                                                             interval_record,
                                                             timestep(ets))
    myh5write(filename, "/best_parameters/shape", best_shape)
    myh5write(filename, "/best_parameters/scale", best_scale)
    myh5write(filename, "/best_parameters/rate_max", ref_rate_max)
    myh5write(filename, "/best_parameters/time_prob", hcat(time_prob.edges[1][1:end-1], time_prob.weights))
    myh5write(filename, "/best_parameters/train_weights", train_weights)
    myh5write(filename, "/best_parameters/time_start", time_start)
    myh5write(filename, "/best_parameters/interval_record", [interval_record...])
    myh5write(filename, "/best_parameters/timestep", timestep(ets))

    root="/"
    println("Analyse temporal features")
    analyse_temporal_features_of_encounter_train(samples,
                                                 filename,
                                                 root,
                                                 support_crate=support_crate)

    println("Analyse disease-related features")
    analyse_infectious_encounter_scan_delta(range_latent, range_infectious,
                                            samples, filename,
                                            @sprintf("%s/disease/delta", root))
    analyse_infectious_encounter_detail(DeltaDiseaseModel(seconds_from_days(2),
                                                          seconds_from_days(3)),
                                                          samples, filename,
                                                          @sprintf("%s/disease/delta_2_3",
                                                                   root))
    analyse_infectious_encounter_detail(DeltaDiseaseModel(seconds_from_days(6),
                                                          seconds_from_days(3)),
                                                          samples, filename,
                                                          @sprintf("%s/disease/delta_6_3",
                                                                   root))

    # randomize, for now we only need this as a reference for the 2d plots
    println("Randomize encounter trains")
    seed = 815
    sur = surrogate_randomize_per_train(samples, seed)
    # myh5write(filename_rand,@sprintf("%s/trains/",root), sur)
    analyse_infectious_encounter_scan_delta(range_latent, range_infectious, sur, filename_rand, @sprintf("%s/disease/delta", root))
end

#############
## Poisson based

"""
    sample_surrogate_sample_surrogate_inhomogeneous_poisson_weighted()

generate samples of inhomogeneous Poisson processes with heterogeneous
(weighted) weights to match the time-dependent rates of the data.

experiment=Copenhagen(); minimum_duration=15*60; path_dat="./dat"
"""
function sample_surrogate_inhomogeneous_poisson_weighted(
        experiment = Copenhagen(),
        minimum_duration = 15*60,
        path_dat = "./dat",
        path_out = "./out",
        num_samples = 20,
        support_crate = default_support_crate,
        range_latent = default_range_latent,
        range_infectious = default_range_infectious,
        seed=1000,
    )
    filename = @sprintf("%s/surrogate_inhomogeneous_poisson_weighted_%s_filtered_%dmin.h5", path_out, label(experiment), minimum_duration/60)
    _,ets, _ = load_processed_data(experiment, minimum_duration, path_dat);
    filename_rand = @sprintf("%s/surrogate_inhomogeneous_poisson_weighted_randomized_per_train_%s_filtered_%dmin.h5",
        path_out, label(experiment), minimum_duration / 60)
    # specificy features that need to be reproducde
    # number of trains:
    num_sample_trains = length(ets)
    # relative rate (prop. to number of encounters) of each train compared to mean:
    train_weights = length.(ets) ./ mean(length.(ets))
    # time-dependent global rate:
    ref_rate = rate(ets, 0:timestep(ets):seconds_from_days(7))
    mean_rate = mean(ref_rate.weights)

    time_start = -seconds_from_days(7)
    interval_record = (0,seconds_from_days(28))
    samples = sample_encounter_trains_poisson(ref_rate,
                                              num_sample_trains,
                                              time_start,
                                              interval_record,
                                              MersenneTwister(seed),
                                              timestep=timestep(ets),
                                              weights=train_weights)

    root="/"
    println("Analyse temporal features")
    analyse_temporal_features_of_encounter_train(samples,
                                                 filename,
                                                 root,
                                                 support_crate=support_crate)

    println("Analyse disease-related features")
    analyse_infectious_encounter_scan_delta(range_latent, range_infectious,
                                            samples, filename,
                                            @sprintf("%s/disease/delta", root))
    analyse_infectious_encounter_detail(DeltaDiseaseModel(seconds_from_days(2),
                                                          seconds_from_days(3)),
                                                          samples, filename,
                                                          @sprintf("%s/disease/delta_2_3",
                                                                   root))
    analyse_infectious_encounter_detail(DeltaDiseaseModel(seconds_from_days(6),
                                                          seconds_from_days(3)),
                                                          samples, filename,
                                                          @sprintf("%s/disease/delta_6_3",
                                                                   root))

    # randomize, for now we only need this as a reference for the 2d plots
    println("Randomize encounter trains")
    seed = 815
    sur = surrogate_randomize_per_train(samples, seed)
    # myh5write(filename_rand,@sprintf("%s/trains/",root), sur)
    analyse_infectious_encounter_scan_delta(range_latent, range_infectious, sur, filename_rand, @sprintf("%s/disease/delta", root))
end


"""
    sample_surrogate_sample_surrogate_inhomogeneous_poisson()

generate samples of inhomogeneous Poisson processes with same mean rates that
match the time-dependent rate of the data.

experiment=Copenhagen(); minimum_duration=15*60; path_dat="./dat"
"""
function sample_surrogate_inhomogeneous_poisson(
        experiment = Copenhagen(),
        minimum_duration = 15*60,
        path_dat = "./dat",
        path_out = "./out",
        num_samples = 20,
        support_crate = default_support_crate,
        range_latent = default_range_latent,
        range_infectious = default_range_infectious,
        seed=1000,
    )
    filename = @sprintf("%s/surrogate_inhomogeneous_poisson_%s_filtered_%dmin.h5", path_out, label(experiment), minimum_duration/60)
    _,ets, _ = load_processed_data(experiment, minimum_duration, path_dat);
    filename_rand = @sprintf("%s/surrogate_inhomogeneous_poisson_randomized_per_train_%s_filtered_%dmin.h5",
        path_out, label(experiment), minimum_duration / 60)
    # specificy features that need to be reproducde
    # number of trains:
    num_sample_trains = length(ets)
    # time-dependent global rate:
    ref_rate = rate(ets, 0:timestep(ets):seconds_from_days(7))
    mean_rate = mean(ref_rate.weights)

    time_start = -seconds_from_days(7)
    interval_record = (0,seconds_from_days(28))
    samples = sample_encounter_trains_poisson(ref_rate,
                                              num_sample_trains,
                                              time_start,
                                              interval_record,
                                              MersenneTwister(seed),
                                              timestep=timestep(ets))

    root="/"
    println("Analyse temporal features")
    analyse_temporal_features_of_encounter_train(samples,
                                                 filename,
                                                 root,
                                                 support_crate=support_crate)

    println("Analyse disease-related features")
    analyse_infectious_encounter_scan_delta(range_latent, range_infectious,
                                            samples, filename,
                                            @sprintf("%s/disease/delta", root))
    analyse_infectious_encounter_detail(DeltaDiseaseModel(seconds_from_days(2),
                                                          seconds_from_days(3)),
                                                          samples, filename,
                                                          @sprintf("%s/disease/delta_2_3",
                                                                   root))
    analyse_infectious_encounter_detail(DeltaDiseaseModel(seconds_from_days(6),
                                                          seconds_from_days(3)),
                                                          samples, filename,
                                                          @sprintf("%s/disease/delta_6_3",
                                                                   root))

    # randomize, for now we only need this as a reference for the 2d plots
    println("Randomize encounter trains")
    seed = 815
    sur = surrogate_randomize_per_train(samples, seed)
    # myh5write(filename_rand,@sprintf("%s/trains/",root), sur)
    analyse_infectious_encounter_scan_delta(range_latent, range_infectious, sur, filename_rand, @sprintf("%s/disease/delta", root))
end


"""
    sample_surrogate_sample_surrogate_homogeneous_poisson_weighted()

generate samples of homogeneous Poisson processes with heterogeneous
weights to match the heterogeneous rates of the data.

experiment=Copenhagen(); minimum_duration=15*60; path_dat="./dat"
"""
function sample_surrogate_homogeneous_poisson_weighted(
        experiment = Copenhagen(),
        minimum_duration = 15*60,
        path_dat = "./dat",
        path_out = "./out",
        num_samples = 20,
        support_crate = default_support_crate,
        range_latent = default_range_latent,
        range_infectious = default_range_infectious,
        seed=1000,
    )
    filename = @sprintf("%s/surrogate_homogeneous_poisson_weighted_%s_filtered_%dmin.h5", path_out, label(experiment), minimum_duration/60)
    _,ets, _ = load_processed_data(experiment, minimum_duration, path_dat);
    filename_rand = @sprintf("%s/surrogate_homogeneous_poisson_weighted_randomized_per_train_%s_filtered_%dmin.h5",
        path_out, label(experiment), minimum_duration / 60)
    # specificy features that need to be reproducde
    # number of trains:
    num_sample_trains = length(ets)
    # relative rate (prop. to number of encounters) of each train compared to mean:
    train_weights = length.(ets) ./ mean(length.(ets))
    # time-dependent global rate:
    ref_rate = rate(ets, 0:timestep(ets):seconds_from_days(7))
    mean_rate = mean(ref_rate.weights)

    time_start = -seconds_from_days(7)
    interval_record = (0,seconds_from_days(28))
    samples = sample_encounter_trains_poisson(mean_rate,
                                              num_sample_trains,
                                              time_start,
                                              interval_record,
                                              MersenneTwister(seed),
                                              timestep=timestep(ets),
                                              weights=train_weights)

    root="/"
    println("Analyse temporal features")
    analyse_temporal_features_of_encounter_train(samples,
                                                 filename,
                                                 root,
                                                 support_crate=support_crate)

    println("Analyse disease-related features")
    analyse_infectious_encounter_scan_delta(range_latent, range_infectious,
                                            samples, filename,
                                            @sprintf("%s/disease/delta", root))
    analyse_infectious_encounter_detail(DeltaDiseaseModel(seconds_from_days(2),
                                                          seconds_from_days(3)),
                                                          samples, filename,
                                                          @sprintf("%s/disease/delta_2_3",
                                                                   root))
    analyse_infectious_encounter_detail(DeltaDiseaseModel(seconds_from_days(6),
                                                          seconds_from_days(3)),
                                                          samples, filename,
                                                          @sprintf("%s/disease/delta_6_3",
                                                                   root))

    # randomize, for now we only need this as a reference for the 2d plots
    println("Randomize encounter trains")
    seed = 815
    sur = surrogate_randomize_per_train(samples, seed)
    # myh5write(filename_rand,@sprintf("%s/trains/",root), sur)
    analyse_infectious_encounter_scan_delta(range_latent, range_infectious, sur, filename_rand, @sprintf("%s/disease/delta", root))
end


"""
    sample_surrogate_sample_surrogate_homogeneous_poisson()

generate samples of homogeneous Poisson processes with same mean rates that
match the mean rate of the data.

experiment=Copenhagen(); minimum_duration=15*60; path_dat="./dat"
"""
function sample_surrogate_homogeneous_poisson(
        experiment = Copenhagen(),
        minimum_duration = 15*60,
        path_dat = "./dat",
        path_out = "./out",
        num_samples = 20,
        support_crate = default_support_crate,
        range_latent = default_range_latent,
        range_infectious = default_range_infectious,
        seed=1000,
    )
    filename = @sprintf("%s/surrogate_homogeneous_poisson_%s_filtered_%dmin.h5", path_out, label(experiment), minimum_duration/60)
    _,ets, _ = load_processed_data(experiment, minimum_duration, path_dat);
    filename_rand = @sprintf("%s/surrogate_homogeneous_poisson_randomized_per_train_%s_filtered_%dmin.h5",
        path_out, label(experiment), minimum_duration / 60)
    # specificy features that need to be reproducde
    # number of trains:
    num_sample_trains = length(ets)
    # time-dependent global rate:
    ref_rate = rate(ets, 0:timestep(ets):seconds_from_days(7))
    mean_rate = mean(ref_rate.weights)

    time_start = -seconds_from_days(7)
    interval_record = (0,seconds_from_days(28))

    samples = sample_encounter_trains_poisson(mean_rate,
                                              num_sample_trains,
                                              time_start,
                                              interval_record,
                                              MersenneTwister(seed),
                                              timestep=timestep(ets))

    root="/"
    println("Analyse temporal features")
    analyse_temporal_features_of_encounter_train(samples,
                                                 filename,
                                                 root,
                                                 support_crate=support_crate)

    println("Analyse disease-related features")
    analyse_infectious_encounter_scan_delta(range_latent, range_infectious,
                                            samples, filename,
                                            @sprintf("%s/disease/delta", root))
    analyse_infectious_encounter_detail(DeltaDiseaseModel(seconds_from_days(2),
                                                          seconds_from_days(3)),
                                                          samples, filename,
                                                          @sprintf("%s/disease/delta_2_3",
                                                                   root))
    analyse_infectious_encounter_detail(DeltaDiseaseModel(seconds_from_days(6),
                                                          seconds_from_days(3)),
                                                          samples, filename,
                                                          @sprintf("%s/disease/delta_6_3",
                                                                   root))

    # randomize, for now we only need this as a reference for the 2d plots
    println("Randomize encounter trains")
    seed = 815
    sur = surrogate_randomize_per_train(samples, seed)
    # myh5write(filename_rand,@sprintf("%s/trains/",root), sur)
    analyse_infectious_encounter_scan_delta(range_latent, range_infectious, sur, filename_rand, @sprintf("%s/disease/delta", root))
end

#############
## Weibull based

"""
    sample_surrogate_sample_surrogate_weibull_weighted()

generate samples of Weibull renewal processes with heterogeneous (weighted)
weights to match the inter-encounter-interval distributino of the data.

experiment=Copenhagen(); minimum_duration=15*60; path_dat="./dat"
"""
function sample_surrogate_weibull_weighted(
        experiment = Copenhagen(),
        minimum_duration = 15*60,
        path_dat = "./dat",
        path_out = "./out",
        num_samples = 20,
        support_crate = default_support_crate,
        range_latent = default_range_latent,
        range_infectious = default_range_infectious,
        seed=1000,
    )
    filename = @sprintf("%s/surrogate_weibull_weighted_%s_filtered_%dmin.h5", path_out, label(experiment), minimum_duration/60)
    _,ets, _ = load_processed_data(experiment, minimum_duration, path_dat);
    filename_rand = @sprintf("%s/surrogate_weibull_weighted_randomized_per_train_%s_filtered_%dmin.h5",
        path_out, label(experiment), minimum_duration / 60)
    # specificy features that need to be reproducde
    # number of trains:
    num_sample_trains = length(ets)
    # relative rate (prop. to number of encounters) of each train compared to mean:
    train_weights = length.(ets) ./ mean(length.(ets))
    # relative rate (prop. to number of encounters) of each train compared to mean:
    train_weights = length.(ets) ./ mean(length.(ets))
    # reference distribution of inter-encounter intervals
    ref_dist = distribution_durations(inter_encounter_intervals(ets), timestep=timestep(ets));
    args_weibull = fit_Weibull(ref_dist)

    time_start = -seconds_from_days(7)
    interval_record = (0,seconds_from_days(28))

    samples = sample_encounter_trains_weibull_renewal(args_weibull,
                                              num_sample_trains,
                                              time_start,
                                              interval_record,
                                              MersenneTwister(seed),
                                              timestep=timestep(ets),
                                              weights=train_weights)

    root="/"
    println("Analyse temporal features")
    analyse_temporal_features_of_encounter_train(samples,
                                                 filename,
                                                 root,
                                                 support_crate=support_crate)

    println("Analyse disease-related features")
    analyse_infectious_encounter_scan_delta(range_latent, range_infectious,
                                            samples, filename,
                                            @sprintf("%s/disease/delta", root))
    analyse_infectious_encounter_detail(DeltaDiseaseModel(seconds_from_days(2),
                                                          seconds_from_days(3)),
                                                          samples, filename,
                                                          @sprintf("%s/disease/delta_2_3",
                                                                   root))
    analyse_infectious_encounter_detail(DeltaDiseaseModel(seconds_from_days(6),
                                                          seconds_from_days(3)),
                                                          samples, filename,
                                                          @sprintf("%s/disease/delta_6_3",
                                                                   root))

    # randomize, for now we only need this as a reference for the 2d plots
    println("Randomize encounter trains")
    seed = 815
    sur = surrogate_randomize_per_train(samples, seed)
    # myh5write(filename_rand,@sprintf("%s/trains/",root), sur)
    analyse_infectious_encounter_scan_delta(range_latent, range_infectious, sur, filename_rand, @sprintf("%s/disease/delta", root))
end

"""
    sample_surrogate_sample_surrogate_weibull()

generate samples of Weibull renewal processes with same paramters to match the
inter-encounter-interval distributinon of the data.

experiment=Copenhagen(); minimum_duration=15*60; path_dat="./dat"
"""
function sample_surrogate_weibull(
        experiment = Copenhagen(),
        minimum_duration = 15*60,
        path_dat = "./dat",
        path_out = "./out",
        num_samples = 20,
        support_crate = default_support_crate,
        range_latent = default_range_latent,
        range_infectious = default_range_infectious,
        seed=1000,
    )
    filename = @sprintf("%s/surrogate_weibull_%s_filtered_%dmin.h5", path_out, label(experiment), minimum_duration/60)
    _,ets, _ = load_processed_data(experiment, minimum_duration, path_dat);
    filename_rand = @sprintf("%s/surrogate_weibull_randomized_per_train_%s_filtered_%dmin.h5",
        path_out, label(experiment), minimum_duration / 60)
    # specificy features that need to be reproducde
    # number of trains:
    num_sample_trains = length(ets)
    # reference distribution of inter-encounter intervals
    ref_dist = distribution_durations(inter_encounter_intervals(ets), timestep=timestep(ets));
    args_weibull = fit_Weibull(ref_dist)

    time_start = -seconds_from_days(7)
    interval_record = (0,seconds_from_days(28))

    samples = sample_encounter_trains_weibull_renewal(args_weibull,
                                              num_sample_trains,
                                              time_start,
                                              interval_record,
                                              MersenneTwister(seed),
                                              timestep=timestep(ets))

    root="/"
    println("Analyse temporal features")
    analyse_temporal_features_of_encounter_train(samples,
                                                 filename,
                                                 root,
                                                 support_crate=support_crate)

    println("Analyse disease-related features")
    analyse_infectious_encounter_scan_delta(range_latent, range_infectious,
                                            samples, filename,
                                            @sprintf("%s/disease/delta", root))
    analyse_infectious_encounter_detail(DeltaDiseaseModel(seconds_from_days(2),
                                                          seconds_from_days(3)),
                                                          samples, filename,
                                                          @sprintf("%s/disease/delta_2_3",
                                                                   root))
    analyse_infectious_encounter_detail(DeltaDiseaseModel(seconds_from_days(6),
                                                          seconds_from_days(3)),
                                                          samples, filename,
                                                          @sprintf("%s/disease/delta_6_3",
                                                                   root))

    # randomize, for now we only need this as a reference for the 2d plots
    println("Randomize encounter trains")
    seed = 815
    sur = surrogate_randomize_per_train(samples, seed)
    # myh5write(filename_rand,@sprintf("%s/trains/",root), sur)
    analyse_infectious_encounter_scan_delta(range_latent, range_infectious, sur, filename_rand, @sprintf("%s/disease/delta", root))
end

###############################################################################
###############################################################################
###############################################################################
# data-driven branching processes

"""
    sample_braching_process()

data-driven branching proceses that generates offsprings as doubly-stochastic
process by first drawing pot. infectious encounters from empirical distribtion,
which determine binomial distribution that describe independent infection of
each encounter with a given probability

experiment=Copenhagen(); minimum_duration = 15*60; path_dat="./dat"; seed_rand=1000;
T_lat=2;T_ift=3
"""
function sample_branching_process(
        #optional
        experiment=Copenhagen(),
        minimum_duration = 15*60,
        path_dat = "./dat",
        path_out = "./out",
        seed_rand = 1000,
        seed_bp   = 1000,
        samples   = Int(1e5),
        N_0 = 1,
        N_max = Int(1e2),
        T_max = 10,
    )
    filename = @sprintf("%s/branching_process_%s_filtered_%dmin.h5", path_out, label(experiment), minimum_duration/60)

    # load encounter trains (ets)
    _, ets_data, _ = load_processed_data(experiment, minimum_duration, path_dat);
    ets_rand = surrogate_randomize_per_train(ets_data, seed_rand);

    T_ift=3
    T_lat_list=[2,6]

    # get mean infectious contacts for random
    disease_model = DeltaDiseaseModel(seconds_from_days(T_lat_list[1]), seconds_from_days(T_ift))
    dist = distribution_from_samples_infectious_encounter(
                samples_infectious_encounter(disease_model, ets_rand)
           )
    edist = EmpiricalDistribution(dist)
    mean_number_contacts = expectation(edist)
    p_ref = 3.0 / mean_number_contacts

    # in general want representation via R
    Rs = collect(0.6:0.1:6.0)

    #branching process analysis
    for T_lat in T_lat_list
        for (label,ets) in zip(["data","rand"],[ets_data, ets_rand])
            # different data sets can use same seeds without problem but to
            # ensure reproducibility in case I reorder the loops
            rng = MersenneTwister(seed_bp)

            println(T_lat, " ", label)
            # data-driven distributions
            disease_model = DeltaDiseaseModel(seconds_from_days(T_lat), seconds_from_days(T_ift))
            dist = distribution_from_samples_infectious_encounter(
                        samples_infectious_encounter(disease_model, ets)
                   )
            edist = EmpiricalDistribution(dist)

            mean_number_contacts = expectation(edist)
            ps = Rs ./ mean_number_contacts

            # sample survival as a function of infection probability
            p_sur = zeros(length(ps))
            P = Progress(length(ps), 1, "SurvivalProbability: ", offset=0)
            for (j,p) in enumerate(ps)
                pdist = ProbabilisticOffspringDistribution(edist, p)
                step = x->branching_step(rng, x, pdist)
                for i in 1:samples
                    sur, _ = check_survival(step, N_0, N_max)
                    p_sur[j] += sur
                end
                p_sur[j] /= samples
                next!(P)
            end
            Rs = ps * expectation(edist)

            datasetname=@sprintf("/%s/infectious_%.2f_latent_%.2f/survival_probability_p/N0=%d/%d/", label, T_ift, T_lat, N_0, samples)
            myh5write(filename, datasetname, hcat(ps, Rs, p_sur))
            myh5desc(filename, datasetname,
                     "asymptotic survival probability estimates as the fraction of samples that do not fall into the absorbing state (branching process stopped if x>x_max=1000), d1: probability to infect contact, d2: effective R, d3: survival probability")

            # determine survival probability as a function of generation for ensemble
            samples_survived = zeros(T_max)
            N_T = zeros(Int, T_max)
            p = Progress(samples, 1, "BranchingProcess: ", offset=0)
            for i in 1:samples
                N_T .= 0; N_T[1] = N_0
                branching_process!(rng, N_T, ProbabilisticOffspringDistribution(edist, p_ref))
                samples_survived .+= (N_T .> 0)
                next!(p)
            end

            # samples_survived gives survival probability as a function of generation time
            samples_survived ./= samples
            myh5write(filename, @sprintf("/%s/infectious_%.2f_latent_%.2f/survival_probability_generation/p=%f/N0=%d/%d/", label, T_ift, T_lat, p_ref, N_0, samples), samples_survived)

        end
    end
end

"""
    analytic_survivial_probability()

solve numerically the analytic self-consistent equation of the probability
generating function for the data-driven branching process.


experiment=Copenhagen(); minimum_duration = 15*60; path_dat="./dat"; seed_rand=1000;
T_lat=2;T_ift=3
"""
function analytic_survival_probability(
        #optional
        experiment=Copenhagen(),
        minimum_duration = 15*60,
        path_dat = "./dat",
        path_out = "./out",
        seed_rand = 1000,
    )
    filename = @sprintf("%s/analytic_survival_%s_filtered_%dmin.h5", path_out, label(experiment), minimum_duration/60)

    # load encounter trains (ets)
    _, ets_data, _ = load_processed_data(experiment, minimum_duration, path_dat);
    # randomized per train: keeps assumption of distribution of expected
    #                       offspring across individuals but not across time
    ets_rand = surrogate_randomize_per_train(ets_data, seed_rand);
    # NEW: also include assumption of uniform R_0 across individuals
    ets_rand_all = surrogate_randomize_all(ets_data,seed_rand);

    T_ift_list=[1,3]
    T_lat_list = [2,6]

    for T_ift in T_ift_list
        # in general want representation via R
        Rs = collect(0.6:0.1:6.0)

        #branching process analysis
        for T_lat in T_lat_list
            for (label,ets) in zip(["data","rand", "rand_all"],[ets_data, ets_rand, ets_rand_all])
                println(T_ift, " ", T_lat, " ", label)
                # data-driven distributions
                disease_model = DeltaDiseaseModel(seconds_from_days(T_lat), seconds_from_days(T_ift))
                dist = distribution_from_samples_infectious_encounter(
                            samples_infectious_encounter(disease_model, ets)
                       )
                edist = EmpiricalDistribution(dist)

                mean_number_contacts = expectation(edist)
                ps = Rs ./ mean_number_contacts

                # sample survival as a function of infection probability
                p_sur = zeros(length(ps))
                P = Progress(length(ps), 1, "SurvivalProbability: ", offset=0)
                for (j,p) in enumerate(ps)
                    p_sur[j] = solve_survival_probability(edist, p)
                    next!(P)
                end
                Rs = ps * expectation(edist)

                datasetname=@sprintf("/%s/infectious_%.2f_latent_%.2f/survival_probability_p/", label, T_ift, T_lat)
                myh5write(filename, datasetname, hcat(ps, Rs, p_sur))
                myh5desc(filename, datasetname,
                         "semi-analytic solution of asymptotic survival probability, d1: probability to infect contact, d2: effective R, d3: survival probability")
            end
        end
    end

end

"""
    experiment=Copenhagen(); minimum_duration=15*60;path_dat="./dat";I0=100;max_cases=1e6;
    cas, ets, list_contacts = load_processed_data(experiment, minimum_duration, path_dat);
    cond_encounter_rate = conditional_encounter_rate(ets, 0:timestep(ets):seconds_from_days(8+4))

    infectious=3
    latent=6
    seed=1000; rng=MersenneTwister(seed)

    disease_model = DeltaDiseaseModel(seconds_from_days(latent), seconds_from_days(infectious))
    probability_infection = 0.12 # heuristically chosen to match R approx 3.3 for infectious=3 and latent=4 when g=4

"""
function sample_continuous_time_branching(;
        #optional
        experiment=Copenhagen(),
        minimum_duration = 15*60,
        path_dat = "./dat",
        path_out = "./out",
        I0 = 100,
        max_cases=1e6
    )
    mkpath(path_out)

    function do_it(ets, filename, dsetname)
        # encounter rate
        cond_encounter_rate = conditional_encounter_rate(ets, 0:timestep(ets):seconds_from_days(8+4))
        myh5write(filename, @sprintf("/%s/cond_encounter_rate", dsetname),
            hcat(cond_encounter_rate.edges[1][1:end-1], cond_encounter_rate.weights))
        myh5desc(filename, @sprintf("/%s/cond_encounter_rate", dsetname),
            "average encounter rate conditioned on having an encounter at time 0, averaged across all encounter in experiment, d1: times, d2: rate(full)")

        probability_infection = 0.12 # heuristically chosen to match R approx 3.3 (covid) for infectious=3 and latent=4 when g=4
        infectious = 3 # days
        range_latent = 0.0:0.5:8
        #range_latent = 0.0:0.0
        for (l, latent) in enumerate(range_latent)
            # special case of no latent period needs more cases to generate
            # enough days for subsequent analyses
            if latent == 0.0
                max_cases = 5e6
            end
            println(latent)
            disease_model = DeltaDiseaseModel(seconds_from_days(latent), seconds_from_days(infectious))
            #dist_offspring = distribution_from_samples_infectious_encounter(samples_infectious_encounter(disease_model, ets))

            # sample
            range_seeds = 1000:1010
            @showprogress 1 for (s, seed) in enumerate(range_seeds)
                rng = MersenneTwister(seed)
                #println("...", seed)
                times_initial_infections = -1 .* rand(rng, I0) .* seconds_from_days(latent+infectious)
                measurement, sum_avg_Tgen, sum_offsprings, sum_samples  = spread_mean_field(disease_model, cond_encounter_rate, probability_infection, seconds_from_days.(0:1:120.), max_cases=max_cases, seed=seed, initial_infection_times=times_initial_infections)

                myh5write(filename, @sprintf("/%s/cases/latent_%.2f/%d", dsetname, latent, seed), hcat(measurement.edges[1][1:end-1], measurement.weights))
                myh5write(filename, @sprintf("/%s/R0/latent_%.2f/%d", dsetname, latent, seed), [sum_offsprings/sum_samples, sum_samples])
                myh5write(filename, @sprintf("/%s/avg_Tgen/latent_%.2f/%d", dsetname, latent, seed), [sum_avg_Tgen/sum_offsprings, sum_offsprings])
            end
        end
    end

    # use real data
    filename = @sprintf("%s/sample_continuous_branching_%s_filtered_%dmin.h5", path_out, label(experiment), minimum_duration/60)
    cas, ets, list_contacts = load_processed_data(experiment, minimum_duration, path_dat);
    do_it(ets, filename, "measurements")
    analyse_continuous_time_branching(filename, dsetname="measurements")

    # use randomized data
    ets = surrogate_randomize_per_train(ets, 1000)
    do_it(ets, filename, "measurements_randomized_per_train")
    analyse_continuous_time_branching(filename, dsetname="measurements_randomized_per_train")
end

function analyse_continuous_time_branching(filename::String;
        path_out = "./out",
        dsetname::String="measurements"
    )
    infectious = 3
    h5open(filename, "r") do file
        cond_encounter_rate = read(file[dsetname], "cond_encounter_rate")
        cases = file[dsetname]["cases"]
        keys_latent = keys(cases)
        keys_latent = keys_latent[occursin.("latent", keys_latent)]
        list_latent = parse.(Float64, last.(split.(keys_latent, "_")))
        # go through hdf5 file
        #R4_avg = zeros(length(keys_latent))
        #R4_std = zeros(length(keys_latent))
        rate_avg = zeros(length(keys_latent))
        rate_std = zeros(length(keys_latent))
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
            mask_infectious = interval_infectious[1] .< cond_encounter_rate[:,1] .< interval_infectious[2];
            time_generation = sum(cond_encounter_rate[mask_infectious, 1].*cond_encounter_rate[mask_infectious,2])/sum(cond_encounter_rate[mask_infectious, 2])
            time_generation /= seconds_from_days(1)
            keys_seed = keys(cases[key_latent])
            #mean_R4 = zeros(length(keys_seed))
            mean_rate = zeros(length(keys_seed))
            mean_Rg = zeros(length(keys_seed))
            mean_R0 = zeros(Union{Float64, Missing}, length(keys_seed))
            @showprogress 1 for (s, seed) in enumerate(keys_seed)
                println("...", seed)
                measurement = read(file, @sprintf("%s/cases/%s/%s/", dsetname, key_latent, seed))
                time = measurement[:, 1]
                measurement = measurement[:, 2]
                log_meas = log.(measurement)
                # r0 was measured in simulation directly
                r0, stat = read(file, @sprintf("%s/R0/%s/%s/", dsetname, key_latent, seed))
                mean_R0[s] = r0

                # find left border where cases >= some threshold
                large_weights = findall(measurement[:].>1000)
                if length(large_weights) < 1
                    mean_Rg[s] = NaN
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
                #g = 4
                #if length(valid_range) <= g+a
                #    mean_R4[s] = NaN
                #else
                #    R_4 = zeros(length(valid_range)-g-a)
                #    for (i,t) in enumerate(valid_range[1]:valid_range[end]-g-a)
                #        R_4[i] = sum(measurement[t+g : t+g+a]) / sum(measurement[t : t+a])
                #    end
                #    mean_R4[s] = mean(R_4)
                #end

                # spreading rate -> need information about x-axis
                a = floor(Int, 1+list_latent[l])
                if length(valid_range) <= a
                    mean_rate[s] = NaN
                else
                    # calculate R with local (`a`-long) sums for every time point
                    spreading_rate = zeros(length(valid_range)-a)
                    for (i,t) in enumerate(valid_range[1]:valid_range[end]-a)
                        spreading_rate[i] = (log_meas[t+a]-log_meas[t]) / (time[t+a]-time[t])
                    end
                    # and average across all time points
                    mean_rate[s] = mean(spreading_rate)
                end

                println("R0, Rg, rate: ", mean_R0[s], " ", mean_Rg[s], " ", mean_rate[s])
            end
            # l is latent period
            # mean_and_std returns nan if any element is nan.
            #R4_avg[l], R4_std[l] = mean_and_std(mean_R4)
            rate_avg[l], rate_std[l] = mean_and_std(mean_rate)
            Rg_avg[l], Rg_std[l] = mean_and_std(mean_Rg)
            R0_avg[l], R0_std[l] = mean_and_std(skipmissing(mean_R0))
            Tg[l] = time_generation
        end

        # store results somewhere
        open(@sprintf("%s/analysis_continuous_branching_%s.dat", path_out, dsetname), "w") do io
            #write(io, "#latent\t avg(R4)\t std(R4)\t avg(Rg)\t std(Rg)\t avg(R0)\t std(R0)\t g\n")
            #writedlm(io, zip(list_latent, R4_avg, R4_std, Rg_avg, Rg_std, R0_avg, R0_std, Tg))
            write(io, "#latent\t avg(rate)\t std(rate)\t avg(Rg)\t std(Rg)\t avg(R0)\t std(R0)\t g\n")
            writedlm(io, zip(list_latent, rate_avg, rate_std, Rg_avg, Rg_std, R0_avg, R0_std, Tg))
        end
    end
end



###############################################################################
###############################################################################
### quick visualization
"""
#ref
"""
function quick_visualization(ets, test_ets, observable)
    if observable == "iei"
        ref_dist = distribution_durations(
                        inter_encounter_intervals(ets),
                        timestep=timestep(ets)
                    );
        test_dist = distribution_durations(
                        inter_encounter_intervals(test_ets),
                        edges=ref_dist.edges[1]
                    );
        display(plot(ref_dist.edges[1][1:end-1], log.(ref_dist.weights)))
        display(plot!(test_dist.edges[1][1:end-1], log.(test_dist.weights)))
    end
    if observable == "iei_log"
        ref_dist = distribution_durations(
                        inter_encounter_intervals(ets),
                        timestep=timestep(ets)
                    );
        ref_x, ref_P, = logbin(ref_dist)
        test_dist = distribution_durations(
                        inter_encounter_intervals(test_ets),
                        edges=ref_dist.edges[1]
                    );

        test_x, test_P, = logbin(test_dist)
        display(plot(log.(ref_x), log.(ref_P)))
        display(plot!(log.(test_x), log.(test_P)))
    end
    if observable == "rate"
        ref_rate = rate(ets, 0:timestep(ets):seconds_from_days(7))
        test_rate = rate(test_ets, 0:timestep(test_ets):seconds_from_days(7))
        display(plot(ref_rate.edges[1][1:end-1], ref_rate.weights))
        display(plot!(test_rate.edges[1][1:end-1], test_rate.weights))
    end
    if observable == "crate"
        support_crate=0:timestep(ets):seconds_from_days(7*1.5)
        ref_crate = conditional_encounter_rate(ets, support_crate)
        test_crate = conditional_encounter_rate(test_ets, support_crate)
        display(plot(ref_crate.edges[1][1:end-1], ref_crate.weights))
        display(plot!(test_crate.edges[1][1:end-1], test_crate.weights))
    end

    if observable == "encounter_2_3"
        ref_data = samples_infectious_encounter(
                        DeltaDiseaseModel(seconds_from_days(2), seconds_from_days(3)),
                        ets)
        ref_dist_encounter = distribution_from_samples_infectious_encounter(ref_data)
        test_data = samples_infectious_encounter(
                        DeltaDiseaseModel(seconds_from_days(2), seconds_from_days(3)),
                        test_ets)
        test_dist_encounter = distribution_from_samples_infectious_encounter(test_data)
        display(plot(ref_dist_encounter.edges[1][1:end-1], log.(ref_dist_encounter.weights)))
        display(plot!(test_dist_encounter.edges[1][1:end-1], log.(test_dist_encounter.weights)))
    end
end
