using Random
using Distributions
using StatsBase
using LinearAlgebra
using SpecialFunctions

using DelimitedFiles

using Printf
using ProgressMeter

include("./utils.jl")
include("./load_data.jl")


###############################################################################
###############################################################################
### disease model

abstract type AbstractDiseaseModel end

"""
    DiseaseModel

define disease model that characeterises per-agent disease progression starting
from infection at time 0, to being infectious after the latent period, to begin
no longer infectious after the infectious period.
"""
struct UnivariateDiseaseModel{T1,T2} <: AbstractDiseaseModel
    P_latent::UnivariateDistribution{T1}
    P_infectious::UnivariateDistribution{T2}
end

struct DeltaDiseaseModel{T<:Real} <: AbstractDiseaseModel
    P_latent::Normal{T}
    P_infectious::Normal{T}

    function DeltaDiseaseModel{T}(mean_latent, mean_infectious) where T
        new{T}(Normal(mean_latent,0), Normal(mean_infectious, 0))
    end
end
DeltaDiseaseModel(mean_latent, mean_infectious) = DeltaDiseaseModel{Float64}(mean_latent, mean_infectious)

# default example is delta
DiseaseDefault() = DeltaDiseaseModel(seconds_from_days(3.0), seconds_from_days(5.0))

# generically distributed periods (cf. [LLoyed-Smith et al, Nature, 2001] for motivation of Gamma)
GammaDiseaseModel(mean_latent, mean_infectious, k) = UnivariateDiseaseModel(Gamma(k, mean_latent/k), Gamma(k, mean_infectious/k))

# literature based guesses for pandemic diseases
Covid19() =  DeltaDiseaseModel(NaN,NaN)
H1N1()    =  DeltaDiseaseModel(NaN,NaN)

###############################################################################
###############################################################################
### Viral load

"""
    viral_load(model, time_range; samples=Int(1e6), seed=1000)

evaluates for a disease `model` the ensemble probability to be infectious along
a `time_range` (for each step).

# Arguments
    * `model`: DiseaseModel
    * `time_range`: range over which the probability is evaluated.

# Keyword argeuments
    * `samples`: the number of disease progressions realized (ensemble size)
    * `seed`: seed of MersenneTwister random number generator
"""
function viral_load(model::AbstractDiseaseModel, time_range; samples::Int=Int(1e6), seed::Int=1000)
    rng = MersenneTwister(seed)
    load = Histogram(time_range)

    # define local function
    time_to_index(time) = StatsBase.binindex(load,time)

    p = Progress(samples, 1, "Viral Load: ", offset=1)
    for i in 1:samples
        next!(p)
        index_start, index_end = time_to_index.(infectious_interval(model, rng))
        if index_end > length(load.weights)
            load.weights[index_start:index_end-1] .+= 1
        else
            load.weights[index_start:index_end] .+= 1
        end
    end

    load = float(load)
    load.weights ./= float(samples)

    return load
end

function infectious_interval(model::AbstractDiseaseModel, rng::AbstractRNG)
    time_infectious_start = rand(rng, model.P_latent)
    time_infectious_end   = time_infectious_start + rand(rng, model.P_infectious)
    return time_infectious_start, time_infectious_end
end

function infectious_interval(model::DeltaDiseaseModel)
    time_infectious_start = mean(model.P_latent)
    time_infectious_end   = time_infectious_start + mean(model.P_infectious)
    return time_infectious_start, time_infectious_end
end


###############################################################################
###############################################################################
### evaluate/sample infectious encounter


"""
generate samples only with start times that are below time_max.

This is a control.
"""
function samples_infectious_encounter_with_infection_in_early_interval(
        model::AbstractDiseaseModel,
        ets::encounter_trains{I,T},
        time_max::Number;
        #optional
        samples::Int=Int(1e6),
        rng::AbstractRNG=MersenneTwister(1000)
    ) where {I,T}
    # manipulate the cumulative sum of valid encounters to only draw from early
    # interval
    times_onset = similar(trains(ets));
    for (i,train) in enumerate(trains(ets))
        times_onset[i] = trains(ets)[i][trains(ets)[i] .< time_max];
    end
    ets_onset = encounter_trains(times_onset, ids(ets), duration(ets), timestep(ets));

    if sum(length.(ets_onset)) > 0
        data = [Int64[] for i in 1:length(ets)]
        p = Progress(samples, 1, "Sample Infectious Encounter: ", offset=1)
        for i in 1:samples
            next!(p)
            index_train, index_time = random_encoutner(rng, ets_onset)
            times = trains(ets)[index_train]
            time = times[index_time]
            interval_count = time .+ infectious_interval(model, rng)
            encounter = count_times_in_interval(times[index_time+1:end], interval_count)
            push!(data[index_train], encounter)
        end
        return data
    else
        error("The chosen interval leaves no valid disease start points")
    end
end

function samples_infectious_encounter_random_onset(
        model::AbstractDiseaseModel,
        ets::encounter_trains{I,T};
        #optional
        weighted_train::Bool=false,
        weighted_time::Bool=false,
        samples::Int=Int(1e6),
        rng::AbstractRNG=MersenneTwister(1000)
    ) where {I,T}
    support_random_time = 0:timestep(ets):duration(ets)
    if weighted_train
        weight_random_train = ProbabilityWeights(length.(ets))
    end
    if weighted_time
        encounter_rate = rate(ets, 0:timestep(ets):seconds_from_days(7))
        weight_random_time = ProbabilityWeights(vcat(encounter_rate.weights*4))
    end

    data = [Int64[] for i in 1:length(ets)]
    p = Progress(samples, 1, "Sample Infectious Encounter: ", offset=1)
    for i in 1:samples
        next!(p)
        # try several times to sample
        count_bad_tries = 0
        # first draw and fix realization of diases to not introduce a bias when
        # long diseasess are not fitting into data set
        interval_infectious = infectious_interval(model, rng)
        while true
            # draw random starting time unitl disases fits into remaining duration
            if weighted_train
                rand_train = sample(rng, 1:length(ets), weight_random_train)
            else
                rand_train = rand(rng, 1:length(ets))
            end
            if weighted_time
                rand_time = sample(rng, support_random_time, weight_random_time)
            else
                rand_time  = rand(rng, support_random_time)
            end
            train = trains(ets)[rand_train]
            interval_count = rand_time .+ interval_infectious
            if last(interval_count) < duration(ets)
                encounter = count_times_in_interval(train[searchsortedlast(train, rand_time)+1:end], interval_count)
                push!(data[rand_train], encounter)
                break
            end
            count_bad_tries += 1
            if count_bad_tries == 1000
                throw(error("infectious interval does not fit into encounter trains"))
            end
        end
    end
    return data
    return samples_infectious_encounter_random_onset(model, ets, support_random_time = support_random_time, weight_random_time = weight_random_time)
end


"""
    samples_infectious_encounter(model, ets [, samples, rng])

returns a data object (A vector of length trains, each containing a vector with
samples of number of infectious encounter for a particular train). A particular
sample is obtained with `sample_infectious_encounter`.
"""
function samples_infectious_encounter(
        model::AbstractDiseaseModel,
        ets::encounter_trains{I,T};
        #optional
        samples::Int=Int(1e6),
        rng::AbstractRNG=MersenneTwister(1000)
    ) where {I,T}
    data = [Int64[] for i in 1:length(ets)]
    p = Progress(samples, 1, "Sample Infectious Encounter: ", offset=1)
    for i in 1:samples
        next!(p)
        # try several times to sample
        num_encounter, index_train = sample_infectious_encounter(ets, model, rng)
        push!(data[index_train], num_encounter)
    end
    return data
end

"""
   sample_infectious_encounter(source, infectious_interval, rng)
"""
function sample_infectious_encounter(
        ets::encounter_trains{I,T},
        model::AbstractDiseaseModel,
        rng::AbstractRNG
    ) where {I,T}
    count_bad_tries = 0
    # first draw and fix realization of diases to not introduce a bias when
    # long diseasess are not fitting into data set
    interval_infectious = infectious_interval(model, rng)
    while true
        # draw random starting time unitl disases fits into remaining duration
        index_train, index_time = random_encoutner(rng, ets)
        times = trains(ets)[index_train]
        time = times[index_time]
        interval_count = time .+ interval_infectious
        if last(interval_count) < duration(ets)
            encounter = count_times_in_interval(times[index_time+1:end], interval_count)
            return encounter, index_train
        end
        count_bad_tries += 1
        if count_bad_tries == 1000
            throw(error("infectious interval does not fit into contact trains"))
        end
    end
end

#special case for deltaDisease
function samples_infectious_encounter(
        model::DeltaDiseaseModel,
        ets::encounter_trains{I,T};
        #optional
    ) where {I,T}
    interval_infectious  = infectious_interval(model)
    train_duration = duration(ets)
    data = [Int64[] for j in 1:length(ets)]
    for (j,train) in enumerate(trains(ets))
        for (i, time_contact) in enumerate(train)
            interval_infectious_contact = time_contact .+ interval_infectious
            # counter infectious counters if remaining train fits into infectious window
            if last(interval_infectious_contact) < train_duration
                encounter = count_times_in_interval(train[i+1:end], interval_infectious_contact)
                push!(data[j], encounter)
            else # break the for loop because all disease will not fit into remaining duration for later times either
                break
            end
        end
    end
    return data
end

"""
    mean_infectious_encounter(model, contact; samples, rng)

estimatee the mean of infectious encounters for a univariate disease
`model` and a source of `contacts` (can be contact_trains object or something
else)
"""
function mean_infectious_encounter(
        model::AbstractDiseaseModel,
        ets::encounter_trains{I,T};
        #optional
        samples::Int=Int(1e6),
        rng::AbstractRNG=MersenneTwister(1000)
    ) where {I,T}
    data = samples_infectious_encounter(model, ets, samples=samples, rng=rng)
    return mean(vcat(data...))
end
function mean_infectious_encounter(
        model::DeltaDiseaseModel,
        ets::encounter_trains{I,T};
        #optional
    ) where {I,T}
    data = samples_infectious_encounter(model, ets)
    return mean_from_samples_infectious_encounter(data)
end
mean_from_samples_infectious_encounter(data) = mean(vcat(data...))

"""
    distribution_infectious_encounter(model, contact; samples, seed)

sample the distribution of infectious encounters for a univariate disease
`model` and a source of `contacts` (can be contact_trains object or something
else)
"""
function distribution_infectious_encounter(
        model::AbstractDiseaseModel,
        ets::encounter_trains{I,T};
        ### optional
        edges=missing,
        samples::Int=Int(1e6),
        rng::AbstractRNG=MersenneTwister(1000)
    ) where {I,T}
    data = samples_infectious_encounter(model, ets)
    return distribution_from_samples_infectious_encounter(data, edges=edges)
end
function distribution_infectious_encounter(
        model::DeltaDiseaseModel,
        ets::encounter_trains{I,T};
        ### optional
        edges=missing,
    ) where {I,T}
    data = samples_infectious_encounter(model, ets)
    return distribution_from_samples_infectious_encounter(data, edges=edges)
end
function distribution_from_samples_infectious_encounter(
        data;
        ### optional
        edges=missing,
    )
    if ismissing(edges)
        list_encounter = vcat(data...)
        dist = fit(Histogram{Float64}, list_encounter, 0:1:maximum(list_encounter)+2)
    else
        dist = fit(Histogram{Float64}, vcat(data...), edges)
    end
    normalize!(dist)
    return dist
end






function count_times_in_interval(times::Vector{T1}, interval_count::Tuple{T2,T2}) where {T1<:Number,T2<:Number}
    encounter = 0
    for time in times
        encounter += ininterval(time, interval_count)
    end
    return encounter
end
function ininterval(time::T1, interval_count::Tuple{T2,T2}) where {T1<:Number,T2<:Number}
    return (getindex(interval_count,1) <= time < getindex(interval_count,2))
end




