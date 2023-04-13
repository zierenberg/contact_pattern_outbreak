using Statistics
using StatsBase
using SpecialFunctions
using Roots

include("load_data.jl")

###############################################################################
###############################################################################
### functions that work on interaction activity

"""
    coefficient_variation(cas::contact_activities{T}; jackknife_error=false) where T

calculates the coefficient of variation as the ratio of std deviation to the
mean (sigma/mu) accross all individuals from the contact_activities object.
This means that the mean and the variance are really calculates over all
individuals.

With the option `jackknife_error=true` a jackknife error is computed and
returned (which is factor 100+ faster than our cas convenience solution because
we can eaily precompute the partial contributions for the jackknife procedure)
"""
function coefficient_variation(cas::contact_activities{T}; jackknife_error=false) where T
    # data = vector of tuples (mean(a), mean(a^2)) for local activities a per id
    # avg = 1/N sum(mean(a))
    # var = 1/N (sum(mean(a^2)) - avg^2
    function estimate_cv(data)
        N = length(data)
        avg = mean(getindex.(data,1))
        std = sqrt((sum(getindex.(data,2)) - N*avg^2)/(N-1))
        return std/avg
    end
    #print("... coefficient of variation:\n") -> move to analysis all functions
    data = Vector{Tuple{Float64, Float64}}(undef, length(activities(cas)))
    @showprogress 1 for (i,a) in enumerate(activities(cas))
        data[i] = (mean(a), mean(x->x^2, a))
    end
    cv_est = estimate_cv(data)
    if jackknife_error
        return jackknife(estimate_cv, data, naive=cv_est)
    else
        return cv_est
    end
end

"""
    autocorrelation_function(cas::contact_activities, lags;)

calculate the autocorrelation function of the interaction activities accross
participants for the provided lags.
"""
function autocorrelation_function(cas::contact_activities{I,T}, lags::Union{AbstractRange{Int64}, AbstractVector{Int64}}) where {I,T}
    @assert length(lags) > 0
    avg = 0.0
    for a in activities(cas)
        avg += sum(a)/length(a)
    end
    avg /= length(activities(cas))
    #avg = sum(sum.(activities(cas)))/length(activities(cas))/length(times(cas))
    cov = zeros(Float64, length(lags))
    p = Progress(length(cas), 1, "Autocorrelation: ", offset=1)
    for a in activities(cas)
        for (index, k) in enumerate(lags)
            for i in 1:length(a)-k
                cov[index] += a[i]*a[i+k]
            end
        end
        next!(p)
    end
    cov ./= length(activities(cas)).*(length(times(cas)).-lags)

    C  = cov .- avg^2
    C ./= C[1]
    return C
end


"""
    rate(cas::contact_activities, time_range)

calculate the rate of interactions for the times in `time_range`
(start:step:end) averaged over individuals and repetitions of `time_range` in
the data (e.g. if time_range is 0:timestep:1week) and the data has 4
wees, then the average ist also across weeks.
"""
function rate(cas::contact_activities{I,T}, time_range::AbstractRange) where {I,T}
    @assert step(time_range) >= step(times(cas))

    rate = Histogram(time_range)
    norm = Histogram(time_range)
    time_max = last(time_range)
    for a in activities(cas)
        for (i, t) in enumerate(times(cas))
            # modulo for periodic boundaries
            rate[ t%time_max ] += a[i]
            norm[ t%time_max ] += 1
        end
    end

    # rate is avg. number of interactions per timestep
    rate = float(rate)
    rate.weights ./= norm.weights

    return rate
end


###############################################################################
###############################################################################
### functions that operate on list of durations
"""
returns the distribution of contact durations from `list_durations`, which is as usual list of lists of durations

# Example for data preparation:
```
    list_durations = [getindex.(encounter, 4) for encounter in list_encounter];
```

# Error
If interested in the statistics of contacts (most natural) one has to estimate the error of the distribution weights.
Important is that the jackknife routine does not comply with leaving out m=0 elements
```
    full_dist = distribution_contact_durations(list_durations, timestep=timestep(experiment));
    P, Pj, Perr = jackknife_mj(x->distribution_contact_durations(x, edges=full_dist.edges[1]).weights, list_durations[length.(list_durations) .> 0], naive=full_dist.weights);
```

In case one is interested in logartihmic binning, this should be the way to achieve it
```
    xbin, Pbin = logbin(full_dist)
    xP, xP_J, xP_err = jackknife_mj(x->hcat(logbin(distribution_contact_durations(x, edges=full_dist.edges[1]))...)[:,1:2], list_durations[length.(list_durations) .> 0], naive=hcat(xbin,Pbin));
```
which equals
```
    xbin_n, xbin_J, xbin_err = jackknife_mj(x->logbin(distribution_contact_durations(x, edges=full_dist.edges[1]))[1], list_durations[length.(list_durations) .> 0], naive=xbin);
    Pbin_n, Pbin_J, Pbin_err = jackknife_mj(x->logbin(distribution_contact_durations(x, edges=full_dist.edges[1]))[2], list_durations[length.(list_durations) .> 0], naive=Pbin);
```

"""
function distribution_durations(list_x::Vector{T}; edges=missing, timestep=missing) where T
    # flatten the list of lists (not very expansive as it turns out)
    flat_list = vcat(list_x...)
    # construct edges of distribution if not provided
    if ismissing(edges)
        if ismissing(timestep)
            throw(error("either distribution `edges` or `timestep` of experiment have to be specified via keyword arguments"))
        else
            edges = 0:timestep:maximum(flat_list)+timestep
        end
    end
    # sort elements of list into Histogram
    dist = fit(Histogram{Float64}, flat_list, edges)
    normalize!(dist)
    return dist
end



###############################################################################
###############################################################################
### functions that operate on encounter trains (or derivatives such as list_dts)
"""
    rate(ets::encounter_trains, time_range)

calculate the rate of contacts for the times in `time_range`
(start:step:end) averaged over individuals and repetitions of `time_range` in
the data (e.g. if time_range is 0:timestep:1week) and the data has 4
weeks, then the average ist also across weeks.

# Error estimation
should be based on ordinary jackknife method, because the quantity of rate has
same number of entries for each train
 ```
    r,rj,err = jackknife(x->rate(x, 0:300:seconds_from_days(7)), ets)
```

"""
function rate(
        ets::encounter_trains{I,T},
        time_range::AbstractRange;
        # optional arguments
    ) where {I,T}
    @assert step(time_range) >= timestep(ets)
    # for now because I am unsure how do deal with periodic ranges that do not
    # start at 0 -> easier to align data acordingly
    @assert first(time_range) == 0

    num_contacts = Histogram(time_range)
    time_max = last(time_range)
    for train in trains(ets)
        for t in train
            # modulo for periodic boundaries
            num_contacts[ t%time_max ] += 1
        end
    end

    # normalization considers how good time_range fits in duration(ets)
    norm = Histogram(time_range)
    time_max = last(time_range)
    norm.weights .+= floor(Int, duration(ets) / time_max) * length(trains(ets))
    time_left = duration(ets) % time_max
    if time_left > 0
        norm.weights[1:StatsBase.binindex(norm, time_left)] .+= length(trains(ets))
    end

    # rate is avg. number of contacts per second (hence need to devide
    # additionally by time step)
    rate = float(num_contacts)
    rate.weights ./= norm.weights
    rate.weights ./= step(time_range)

    return rate
end

"""


returns the rate of encounters in `time_range` (with according timesteps, unit
of dataset) conditioned on having a contact at time 0.
"""
function conditional_encounter_rate(
        ets::encounter_trains{I,T},
        time_range::AbstractRange;
        # optional arguments
    ) where {I,T}
    num_contacts = Histogram(time_range)
    norm = Histogram(time_range)

    for train in trains(ets)
        #collect statistic over all contacts within the train
        for (i, contact_time) in enumerate(train)
            # add next contacts to histogram
            for j in i+1:length(train)
                time = @inbounds train[j]
                push!(num_contacts, time - contact_time)
                #if time - contact_time > last(num_contacts.edges[1])
                #    break
                #end
            end

            # add a 1 to all those bins that are covered by the dataset
            time_max = duration(ets) - contact_time
            #binindex returns max bin if time_max crosses boundaries
            index_max = StatsBase.binindex(norm, time_max)
            if index_max > length(norm.weights)
                @inbounds norm.weights .+= 1
            else
                for j in 1:index_max
                    @inbounds norm.weights[j] += 1
                end
            end
        end
    end

    rate = float(num_contacts)
    rate.weights ./= step(time_range)
    rate.weights ./= norm.weights

    return rate
end

function integrate(rate::AbstractHistogram, interval::Tuple)
    @assert(first(interval) <= last(interval))
    @assert(first(interval) >= first(rate.edges[1]))
    @assert(first(interval) < last(rate.edges[1]))

    first_idx = StatsBase.binindex(rate, first(interval))
    last_idx = StatsBase.binindex(rate, last(interval))

    integral = sum(rate.weights[first_idx:last_idx])

    return integral*step(rate.edges[1])
end


"""
    distribution_number_encoutner(ets, window, [pattern,])

calculates the distribution of the number of contacts within a certain time
window. With the keyword `pattern` one may specify a pattern of reoccuring
time-windows that should be distinguished. Using pattern one should be aware of
the reference time of the experiment (which can be to some extent be controlled
with the keyword offset). (Copenhagen starts on Sunday assumed at 0:00)

Examples:
calculate distribution of daily number of contacts irrespective of which day
```
    distribution_number_encoutner(ets, seconds_from_days(1))
``

calculate distribution of daily number of contacts for each day of the week separately
```
    distribution_number_encoutner(ets, seconds_from_days(1), pattern=1:7)
``

calculate distribution of daily number of contacts distinguishing weekdays and weekends
```
    distribution_number_encoutner(ets, seconds_from_days(1), pattern=[1,2,2,2,2,2,1])
``


"""
function distribution_number_encounter(
        ets::encounter_trains{I,T},
        window::Real,
        pattern::Union{AbstractRange, AbstractVector};
        ### optional
        offset::Real = 0,
        edges=missing,
    ) where {I,T}
    # find unique elements of pattern
    unique_labels = sort!(unique(pattern))

    # create Histogram to sort count contacts
    num_contacts = Histogram(0:window:duration(ets))

    # create dictionary to sort the number of contacts
    list_num = Dict(unique_labels .=> [Int64[] for i in 1:length(unique_labels)])

    # iterate over contact trains and sort into lists (each part of pattern is
    # `window` long)
    p = Progress(length(ets), 1, "Distribution: ", offset=1)
    for train in trains(ets)
        next!(p)
        num_contacts.weights .= 0
        for time in train
            push!(num_contacts, time-offset)
        end
        # add to list
        for (i, num) in enumerate(num_contacts.weights)
            push!(list_num[pattern[1+(i-1)%length(pattern)]], num)
        end
    end

    if ismissing(edges)
        list_dist = [fit(Histogram{Float64}, list_num[label], 0:1:maximum(list_num[label])+1) for label in unique_labels]
    else
        list_dist = [fit(Histogram{Float64}, list_num[label], edges) for label in unique_labels]
    end
    normalize!.(list_dist)

    return Dict(unique_labels .=> list_dist)
end
distribution_number_encounter(ets::encounter_trains{I,T}, window::Real; offset::Real = 0, edges=missing, return_type=Dict) where {I,T} = distribution_number_encounter(ets, window, [1,], offset=offset, edges=edges)




"""
    inter_encounter_intervals(ets)

calculate the list of inter-encoutner times for each id in `ets` (of type encoutner
trains) and return as vector of vector of inter-encoutner times
(Vector{Vector{T}}).
"""
function inter_encounter_intervals(ets::encounter_trains{I,T})::Vector{Vector{T}} where {I,T}
    list_dts = Vector{T}[]
    for train in trains(ets)
        dts = diff(train)
        push!(list_dts, dts)
    end
    return list_dts
end

function inter_encounter_intervals(
        trains::Vector{Vector{T}}
    )::Vector{Vector{T}} where {T}
    list_dts = Vector{T}[]
    for train in trains
        dts = diff(train)
        push!(list_dts, dts)
    end
    return list_dts
end


"""
    autocorrelation_function(list_dts, lags)

calculate autocorrelation function of inter-contact intervals. Due to the
finite-sample bias, we precalculate the stationary mean interval for each lag
once for the x-data set and once for the y-data set, cf. [Spitzner et al., PLOS
One (2021)].

# Error estimation:
The statistics of interest here is the inter-contact intervals (dts), which
differe from list to list. Hence, jackknife_mj should be used:
```
    c, cj, cerr = jackknife_mj(x->autocorrelation_function(x, lags), list_dts)
```
"""
function autocorrelation_function(list_dts::Vector{Vector{T}}, lags::Union{AbstractRange{Int64}, AbstractVector{Int64}}) where T
    @assert length(lags) > 0

    # calculate global mean assuming stationary inter-contact intervals across
    # individuals
    sum_dts = zero(T)
    num_dts = zero(Int64)
    for dts in list_dts
        sum_dts += sum(dts)
        num_dts += length(dts)
    end
    mean_dt = float(sum_dts)/float(num_dts)
    #mean_dt = sum(sum.(list_dts))/sum(length.(list_dts))

    # calculate actual autocorrelation function E[ (X_1 - E[X_1])(X_2 - E[X_2]) ]
    C  = zeros(Float64, length(lags))
    N  = zeros(Int64, length(lags))
    p = Progress(length(list_dts), 1, "Autocorrelation: ", offset=1)
    for dts in list_dts
        for (index,k) in enumerate(lags)
            for i in 1:length(dts)-k
                term = (dts[i] - mean_dt) * (dts[i+k] - mean_dt)
                C[index] += term
            end
            # N is iteratively calculated because lengths(dts) is not the same
            # across elements
            N[index] += max(length(dts)-k, 0)
        end
        next!(p)
    end

    C  ./= N
    C  ./= C[1]
    return C
end
#wrapper functions
autocorrelation_function(dts::Vector{T}, lags) where T = autocorrelation_function([dts,], lags)
autocorrelation_function(ets::encounter_trains, lags) = autocorrelation_function(inter_contact_times(ets), lags)

function autocorrelation_function_hack(list_dts::Vector{Vector{T}}, lags) where T
    # precalculate mean dt for each lag across trains, distinguishing between
    # [1:end-k] and [1+k:end]
    mean_dt_x = zeros(length(lags))
    mean_dt_y = zeros(length(lags))
    N  = zeros(length(lags))
    for dts in list_dts
        for (j,k) in enumerate(lags)
            mean_dt_x[j] += sum(dts[1:end-k])
            mean_dt_y[j] += sum(dts[1+k:end])
            N[j] += max(length(dts)-k, 0)
        end
    end
    mean_dt_x ./= N
    mean_dt_y ./= N

    # calculate actual autocorrelation function E[ (X_1 - E[X_1])(X_2 - E[X_2]) ]
    C  = zeros(length(lags))
    C2 = zeros(length(lags))
    N  = zeros(length(lags))
    for dts in list_dts
        for (j,k) in enumerate(lags)
            for i in 1:length(dts)-k
                term = (dts[i] - mean_dt_x[j])*(dts[i+k] - mean_dt_y[j])
                C[j]  += term
            end
            N[j] += max(length(dts)-k, 0)
        end
    end
    C  ./= N
    Var = C[1]
    C  ./= Var

    return C
end




###############################################################################
###############################################################################
### cluster

"""
sample encounter cluster from the list of inter-encounter intervals
and a minimum separation that specifies that larger inter-encoutner intervals terminate the cluster

minimum_separation should be on the order of mean(duration)~8*300 for CNS
"""
function sample_cluster_sizes(list_dts::Vector{Vector{T}}, minimum_separation::Number) where T
    samples = [Int[] for i in 1:length(list_dts)]
    for (i,dts) in enumerate(list_dts)
        # start from cluster size 1 because we work on level of dts
        size = 1
        for dt in dts
            if dt < minimum_separation
                size += 1
            else
                push!(samples[i], size)
                size = 1
            end
        end
    end

    return samples
end

"""
evaluate distribution of contact cluster from contact activities as the sum of
successive simultaneious contacts.
"""
function sample_cluster_sizes(cas::contact_activities{I,T}) where {I,T}
    samples = [Int[] for i in 1:length(cas)]
    for (i,activity) in enumerate(activities(cas))
        size = 0
        for a in activity
            if size > 0 && a==0
                push!(samples[i], size)
                size = 0
            end
            size += a
        end
    end

    return samples
end


"""
    This could also be used for distribution_duration
"""
function distribution(samples::Vector{T}; edges=missing, resolution=1) where T
    # flatten the list of lists (not very expansive as it turns out)
    flat_samples = vcat(samples...)
    if ismissing(edges)
        dist = fit(Histogram{Float64}, flat_samples, 0:resolution:maximum(flat_samples)+resolution)
    else
        dist = fit(Histogram{Float64}, flat_samples, edges)
    end
    normalize!(dist)

    return dist
end
