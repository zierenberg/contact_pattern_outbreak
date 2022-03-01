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

###############################################################################
###############################################################################
# sample disease spread as branching process that generates offsprings from
# data-driven distribution

"""
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

###############################################################################
###############################################################################
# sample disease spread with generative model that captures cyclic rate
# features of data and analyse effective R from the time course of new cases.

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
function sample_mean_field(;
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

        probability_infection = 0.12 # heuristically chosen to match R approx 3.3 for infectious=3 and latent=4 when g=4
        infectious = 3 # days
        range_latent = 0.0:0.5:8
        for (l, latent) in enumerate(range_latent)
            println(latent)
            disease_model = DeltaDiseaseModel(seconds_from_days(latent), seconds_from_days(infectious))
            #dist_offspring = distribution_from_samples_infectious_encounter(samples_infectious_encounter(disease_model, ets))

            # sample
            range_seeds = 1000:1010
            @showprogress 1 for (s, seed) in enumerate(range_seeds)
                rng = MersenneTwister(seed)
                #println("...", seed)
                times_initial_infections = -1 .* rand(rng, I0) .* seconds_from_days(latent+infectious)
                measurement, sum_offsprings, sum_samples = spread_mean_field(disease_model, cond_encounter_rate, probability_infection, seconds_from_days.(0:1:120.), max_cases=max_cases, seed=seed, initial_infection_times=times_initial_infections)

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

function analyse_mean_field(filename::String;
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
                a = 2
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
        open(@sprintf("%s/analysis_mean-field_%s.dat", path_out, dsetname), "w") do io
            #write(io, "#latent\t avg(R4)\t std(R4)\t avg(Rg)\t std(Rg)\t avg(R0)\t std(R0)\t g\n")
            #writedlm(io, zip(list_latent, R4_avg, R4_std, Rg_avg, Rg_std, R0_avg, R0_std, Tg))
            write(io, "#latent\t avg(rate)\t std(rate)\t avg(Rg)\t std(Rg)\t avg(R0)\t std(R0)\t g\n")
            writedlm(io, zip(list_latent, rate_avg, rate_std, Rg_avg, Rg_std, R0_avg, R0_std, Tg))
        end
    end
end

