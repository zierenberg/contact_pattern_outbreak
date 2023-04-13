using ProgressMeter
using StatsBase
using LinearAlgebra
using Printf
using DelimitedFiles

include("utils.jl")

seconds_from_days(days) = days*24*60*60

#solitons for different experiments
abstract type ContactData end
abstract type Sociopatterns <: ContactData end
struct InVS15 <: Sociopatterns end
struct InVS13 <: Sociopatterns end
struct LyonSchool <: Sociopatterns end
struct Copenhagen <: ContactData end

label(experiment::InVS15) = "InVS15"
label(experiment::InVS13) = "InVS13"
label(experiment::LyonSchool) = "LyonSchool"
label(experiment::Copenhagen) = "Copenhagen"

timestep(experiment::InVS15) = 20
timestep(experiment::Copenhagen) = 5*60

duration(experiment::InVS15)     = 14*24*60*60
duration(experiment::Copenhagen) = 28*24*60*60

global rssi_threshold = -80


# A sketch
###############################################################################
#
# Interactions
# ID 1 .........|||.....|||..................|||||......
# ID 2 .........|||..........................|||........
# ID 3 .................|||....................|||......
#
# Interaction Activity
# ID 1 .........111.....111..................11211......
# ID 2 .........111..........................112........
# ID 3 .................111....................211......
#
# Contact Train
# ID 1 .........|.......|....................|.|........
# ID 2 .........|............................|..........
# ID 3 .................|......................|........
#
###############################################################################


###############################################################################
###############################################################################
###############################################################################
# functions that load objects from filtered data

"""
    load_processed_data(experiment, minimum_duraton, path [, filter_out_incomplete])

loads contact data for `experiment` (default Copenhagen()) from `path`, filters
for co-location times that persist a `minimum_duration` (in units of
experiment) and are close enough (specified for each experiment), and returns
the objects that carry the contact activity, the encoutners trains and the list
of contacts.

For the Copenhagen Network Study, one can choose to filter out those
individuals for which no signal was recording on neither first nor last day of
experiment, indicating that they did not participate the full duration.
"""
function load_processed_data(
        experiment,                   # soliton which experiment, e.g. `Coppenhagen()`
        minimum_duration,             # absolute time (in units of experiment)
        path::String;                 # filepath to load data from
        filter_out_incomplete=false   # if `true`, trains that do not have any signal in the first and last 24h are discarded
    )

    # ids is a list of ids, colocation_times is a 2d array with columns: timestamp, id, jd
    # @CodeRev: this should be clear now from documentation of the function
    selected_ids, colocation_times = load_data(experiment, path, filter_out_incomplete=filter_out_incomplete);
    @printf("Duration: %.2f days\n", (colocation_times[end, 1] - colocation_times[1, 1])/60/60/24.0)

    # get co-location times sorted per tuple
    # `ids_raw` has same format as `ids`, `pairwise_colocation_times` is a dict with tuple (id, jd) as keys and a list of colocation_times as values.
    # call as e.g. colocation_time_pairwise[(0,15)]
    pairwise_colocation_times = sort_colocation_times_pairwise(colocation_times);

    # get contacts and filter out those that are too short
    list_contacts = get_contacts(pairwise_colocation_times, selected_ids, timestep(experiment))
    filter_out_short_contacts!(list_contacts, minimum_duration)
    @printf("Minimum Duration for Contacts: %.2f minutes\n", minimum_duration / 60)

    # calculate encoutner trains and contact activity from list of contacts
    ets = encounter_trains(list_contacts, selected_ids, duration(experiment), timestep(experiment));
    cas = contact_activities(list_contacts, selected_ids, 0:timestep(experiment):duration(experiment));

    return cas, ets, list_contacts
end
load_processed_data() = load_processed_data(Copenhagen(), 15*60, "./dat")



###############################################################################
###############################################################################
###############################################################################
# load data

"""
    load_data(experiment, path [, filter_out_incomplete])

loads interaction data for selected `experiment` from `path` and returns
participant ids and 2D array of colocation_times with column (time, id A, id B).

In case of Copenhagen(), the raw data is from bluetooth signals such that raw
data is filtered for close proximity (only bluetooth signals with RSSI>=-80, cf
Fig. 2 in [1]) and only indlcudes colocation_times with valid id B >=0 (discarding
signals from empty scans or with devices outside of study).

In the case of Copenhange(), there is the option to filter out incomplete
participant participation by returning only ids that have any bluetooth signal
recording on both first and last day of the experiment.

# References
[1] https://doi.org/10.1371/journal.pone.0100915).
"""
# we know that data is in integer format
function load_data(experiment::Copenhagen, path; filter_out_incomplete=false)
    @printf("Load dataset '%s' \n", label(experiment))
    # timestamp, user_a, user_b, rssi
    data = readdlm(@sprintf("%s/bt_symmetric.csv", path), ',', Int64, skipstart=1)

    # filter out rows where bluetooth signal strength (rssi) is below -80
    # Reason: * rssi value in data is maximum that occured within 5min window
    #         * for the devices used in this study, the authors studied the
    #         relation between rssi and distance and find that signal for 1m is
    #         distributed around -75 and for 2m around -80 (perfect scenario
    #         Fig.2 in [1]
    #         * if rssi < -80 this means that for 5min no single encoutner closer
    #         than 1.5m

    colocation_times = data[data[:,4].>=rssi_threshold,:]

    # make dimension match sociopatterns format
    colocation_times = colocation_times[:,1:3]

    # filter out rows where user_b is not a valid target (>=0)
    colocation_times = colocation_times[colocation_times[:,3].>=0,:]

    # find uniqe ids (from user a and user b)
    unique_a = unique!(colocation_times[:,2])
    unique_b = unique!(colocation_times[:,3])
    ids = sort!(unique!(vcat(unique_a, unique_b)))

    # filter out ids for which signals occur neither on first nor on last day (minority)
    # TODO@CodeRev: should this maybe be in an extra function? If so, how?
    if filter_out_incomplete
        print("... filter out ids of participants that had no signal on both first and last day of study\n")
        time_first_occurence = [data[findfirst(x->x==id, data[:,2]),1] for id in ids]
        time_last_occurence = [data[findlast(x->x==id, data[:,2]),1] for id in ids]
        print("... reduced from ", length(ids))
        ids = ids[(time_first_occurence .< 1*24*60*60) .== (time_last_occurence .> 27*24*60*60)]
        print("... to ", length(ids), " ids\n")
    end

    return ids, colocation_times
end
function load_data(experiment::Sociopatterns, path; filter_out_incomplete=false)
    if filter_out_incomplete
        throw(error("filter_out_incomplete is not implemented for experiments from sociopatterns because they do not rely on bluetooth"))
    end
    @printf("Load dataset '%s' \n", label(experiment))
    colocation_times = readdlm(@sprintf("./dat/tij_pres_%s.dat", label(experiment)), ' ', Int64)
    #meta = readdlm(@sprintf("./dat/metadata_%s.dat", label(experiment)), '\t', Any)

    # get ids as array of Ints
    unique_a = unique!(colocation_times[:,2])
    unique_b = unique!(colocation_times[:,3])
    ids = sort!(unique!(vcat(unique_a, unique_b)))
    #ids = meta[:,1]
    #ids = convert(Array{typeof(ids[1])}, ids)

    return ids, colocation_times
end
#default
load_data(;path="./dat/") = load_data(experiment_Copenhagen(), path)



"""
    sort_colocation_times_pairwise(colocation_times)

sorts 2D array of colocation_times with colums (time, id A, id B) into a dictionary
of pairwise colocation_time times and returns a dictionary of
`pairwise_colocation_times`.

access pairwise_colocation_times[(id A, id B)] yields list of times

Because for all our data sets the ids are integer, this implementation yields a
keys of type Tuple{Int64,Int64}. For other data sets where ids may be
non-integer, this function may have to be implemented differenly (depending on
the format of `colocation_times`, which could also be Vector{Tuple{T, I, I}}).
"""
function sort_colocation_times_pairwise(colocation_times::Array{T,2}) where T
    unique_i = unique!(colocation_times[:,2])
    unique_j = unique!(colocation_times[:,3])
    ids = sort!(unique!(vcat(unique_i, unique_j)))

    # initiate dictionary as empty lists
    pairwise_colocation_times = Dict{Tuple{T,T},Vector{T}}()
    for id in ids
        for jd in ids
            pairwise_colocation_times[(id,jd)] = T[]
        end
    end

    # go through data and append times to lists
    # TODO@CodeRev we go through twice here because id1, id2 could be in prnciple also "A", "B", ...
    for (t, id1, id2) in eachrow(colocation_times)
        push!(pairwise_colocation_times[(id1,id2)], t)
        push!(pairwise_colocation_times[(id2,id1)], t)
    end

    return pairwise_colocation_times
end

"""
    filter_out_short_contacts!(list_contacts, minimum_duration)

removes contacts that are shorter than `minimum_duration` from `list_contacts`.

When used with list_contacts from `get_contacts` then the convention of contact
duration is such that `minimum_duration` sets the duration below which
colocation_times get removed:
* if minimum_duration=300  and  times=[300,]        -> duration 0   -> removed
* if minimum_duration=600  and  times=[300,600,900] -> duration 600 -> kept

Operates either directly on a vector of colocation_time times or on a dictionary of
pairwise colocation_time times.
"""
function filter_out_short_contacts!(list_contacts::Vector{Vector{Tuple{T,T}}}, minimum_duration::T ) where {T}
    for contacts in list_contacts
        filter_out_short_contacts!(contacts, minimum_duration)
    end
end
function filter_out_short_contacts!(contacts::Vector{Tuple{T,T}}, minimum_duration::T) where T
    filter!(x->getindex(x,2)>=minimum_duration, contacts)
end


###############################################################################
###############################################################################
###############################################################################
# Interaction activity

"""
   activity

activity objects stores a timeseries of the number of colocation_times of `id` in a
`time_range` (typically 0:timestep:duration) together with a sum of
colocation_times for some of the sampling algorithms.

# Constructors
1) directly with activity(timeseries)
2) from the contacts of one individual with any other (vector of Tuples (time contact start, contact duration))
"""
struct activity
    timeseries::Vector{Int64}
    sum::Int64

    function activity(timeseries::Vector{Int64})
        new(copy(timeseries), sum(timeseries))
    end
end
function activity(contacts, time_range)
    # store encoutner list
    timeseries = zeros(Int64, length(time_range))

    # iterate over all contacts
    for contact in contacts
        contact_start = contact[1]
        contact_end   = contact[1] + contact[2]
        timeseries[searchsortedlast(time_range, contact_start):searchsortedlast(time_range, contact_end)] .+= 1
    end
    return activity(timeseries)
end
timeseries(a::activity) = a.timeseries
Base.sum(a::activity) = a.sum
Base.length(a::activity) = length(a.timeseries)
Base.getindex(a::activity, idx::Real) = a.timeseries[idx]
Base.getindex(a::activity, idxs::AbstractRange) = a.timeseries[idxs]
Base.lastindex(a::activity) = lastindex(a.timeseries)
Base.iterate(a::activity) = iterate(a.timeseries)
Base.iterate(a::activity, idx::Real) = iterate(a.timeseries, idx)


"""
    contact_activities{I,T}

includes a vector of `ids`, the `time_range` of the experiment, a vector of
`activities`

* `activity' is defined as a timeseries of the number of co-locations (per
timestep defined in time_range) for each id.

# Constructors:
1) Directly from a list of `activities`, a list of `ids`, and a `time_range`
2) From a dictionary of `pairwise_colocation_times`, a lust of `ids` that has to be
a subset of the ids defined in `pairwise_colocation_times`, and a `time_range`,
which additionally calculates the list of activities by counting the number of
colocation_times for each id in the specified time_range (e.g. 0:timestep:duration)

# Access
* activities(cas) returns list of activities
* random_colocation_time(rng, cas) returns (index_id, index_activity) to access activities(cas)[index_train][index_time]
* time_range(cas) returns time_range of the experiemnt
* ids(cas) returns ids of the experiemnt
* cas[1:3] returns a copy of the object with only the first three activitiess and ids
* cas[BitArray] returns a copy of the object with BitArray applied to activities an ids
"""
struct contact_activities{I,T}
    _activities::Vector{activity}     # outer: id, inner: activity (time series)
    _ids::Vector{I}
    _time_range::AbstractRange{T}     # e.g. 1:3:100

    # minimal constructor that processes already calculated activities
    function contact_activities(activities::Vector{activity}, ids::Vector{I}, time_range::AbstractRange{T}) where {I,T}
        @assert length(activities) == length(ids)
        @assert length(activities[1]) == length(time_range)
        @assert length(activities[end]) == length(time_range)
        new{I,T}(activities, ids, time_range)
    end
end
# constructor that creates activites from pairwise colocation times and passes
# those to the inner constructor
function contact_activities(list_contacts::Vector{Vector{Tuple{T,T}}}, ids::Vector{I}, time_range::AbstractRange{T}) where {I,T}
    #this is used because `ids` can be a selection of all potential ids
    activities = activity[]
    @showprogress 1 for contacts in list_contacts
        push!(activities, activity(contacts, time_range))
    end

    contact_activities(activities, ids, time_range)
end


###############################################################################
# access functions
# TODO: bound checks! -> see how this is done in Vectors??
Base.getindex(cas::contact_activities{I,T}, idxs::Union{AbstractRange, BitArray}) where {I,T} = contact_activities(cas._activities[idxs], cas._ids[idxs], cas._time_range)
#
ids(cas::contact_activities{I,T}) where {I,T} = cas._ids
#
activities(cas::contact_activities) = cas._activities
#
times(cas::contact_activities) = cas._time_range
#
Base.length(cas::contact_activities{I,T}) where {I,T} = length(cas._activities)
#
Base.lastindex(cas::contact_activities{I,T}) where {I,T} = lastindex(cas._activities)

# get random contact (e.g. for disease start) -> replaces contact_pointer
random_contact(rng::AbstractRNG, cas::contact_activities{I,T}) where {I,T} = random_contact(rng::AbstractRNG, activities(cas))
function random_contact(rng::AbstractRNG, activities::Vector{activity})
    sum_contacts_target = rand(rng, 1:sum(sum.(activities)))

    # find index_a, index_t such that
    # sum(sum.(activities[1:index_a-1])) + sum(activitites[index_a][1:index_t] >= sum_contacts_target
    sum_contacts = 0
    for (index_a, activity) in enumerate(activities)
        # find index_a such that sum_contacts_target is reached inside
        # @CodeRev: sum(activity) is not calculating the sum each time, it is
        # accessing a value in memory
        if sum_contacts + sum(activity) >= sum_contacts_target
            for (index_t, num_contacts) in enumerate(activity)
                sum_contacts += num_contacts
                # check for >= because num_contacts can be > 1 (thereby
                # adding all degenerate interactiosn can increase
                # sum_contacts above target)
                if  sum_contacts >= sum_contacts_target
                    return (index_a, index_t)
                end
            end
        end
        sum_contacts += sum(activity)
    end

    throw(error("something went wrong when locating a random contact"))
end




###############################################################################
###############################################################################
###############################################################################
# encounter (for encounter duration distribution)

"""
    get_contacts(pairwise_colocation_times, id, separation_threshold)

calculate and returns a list of `contacts` as a vectors of contacts (time
start, duration) for each specific `id` with any other `jd`.  The contacts are
calculated by merging co-location times between id and jd that are separated
apart shorter than separation_threshold into a single encounter.

If instead of a single id a vector of `ids` is provided, this function returns
Vector{Vector{encounter}}

The duration is here defined as the time-difference between the first and the
last time, such that isolated (single timestep) colocation_time times have duration 0.
Examples for Copenhagen() with timestep=300s:
* times=[300,]        -> duration 0
* times=[300,600,900] -> duration 600

"""
function get_contacts(pairwise_colocation_times::Dict{Tuple{I,I},Vector{T}}, ids::Vector{I}, separation_threshold::T) where {I,T}
    # this is needed because `ids` can be a selection of all ids
    all_ids = unique(getindex.(keys(pairwise_colocation_times), 1))
    list_contacts = Vector{Tuple{T, T}}[]
    @showprogress 1 for id in ids
        encounter = get_contacts(pairwise_colocation_times, all_ids, id, separation_threshold)
        push!(list_contacts, encounter)
    end
    return list_contacts
end
function get_contacts(pairwise_colocation_times::Dict{Tuple{I,I},Vector{T}}, ids::Vector{I}, id::I, separation_threshold::T) where {I,T}
    @assert id in ids

    # store encoutner list
    c_list = Tuple{T,T}[]

    # iterate over all possible targets (jd)
    for jd in ids
        # exclude self from targets
        if (jd==id)
            continue
        end

        # get encoutner list for tuple from the co-location times
        colocation_times = pairwise_colocation_times[(id,jd)]
        #TODO@CodeRev: this is called twice! once for id and once for jd. okay or not?
        _contacts_per_tuple!(c_list, colocation_times, separation_threshold)
    end

    #sort and return
    sort!(c_list)
    return c_list
end
#internal use only
#
#fill contact list `c_list` [list of (begin, duration)] from the
#colocation times of pairs (id, jd). A contact last as long as subsequent
#colocation times are seperated at most by `separation_threshold`.
#
#Remark
#The duration for contacts with n time steps is (n-1)*timestep, such that for
#isolated interaction time the duration is 0. This convention considers that an
#interaction in a single timestep has vanishing probability to last exactly for
#1 time step.
function _contacts_per_tuple!(
        c_list::Vector{Tuple{T, T}}, 
        colocation_times::Vector{T}, 
        separation_threshold::T
    ) where {T}
    if (length(colocation_times) == 0)
        return
    end

    # variables in time units of data
    dt = 0

    e_beg = colocation_times[1]

    for i in 1:length(colocation_times)
        c_end = colocation_times[i]
        # end if last colocation time
        if i==length(colocation_times)
            push!(c_list, (e_beg, c_end-e_beg))
        else
            # check for dt between colocation times
            dt = colocation_times[i+1] - colocation_times[i]
            # if two encounter separated long enough then save current encounter
            # to list and reset beginning to the next encounter time
            if dt > separation_threshold
                push!(c_list, (e_beg, c_end-e_beg))
                # reset beginning
                e_beg = colocation_times[i+1]
            end
        end
    end
end



###############################################################################
###############################################################################
###############################################################################
# Contact trains

"""
    encounter_trains{T}

includes a vector of `ids`, a list of encoutner `trains`, the `duration` and
`timestep` of the experiment

* a single `train' for user id is defined as the list of all encoutner times
(start times of encounter, no information about duration) that user id had
during experiment.

# Constructors:
1) Directly from a list of `trains`{list of lists), a list of `ids`(list), a
`duration`(T), and a `timestep` (T)
2) From a list of contacts
2) From a dictionary of `pairwise_colocation_times` (Dict), a list of `ids` that has to be
a subset of the ids defined in `pairwise_colocation_times`, and a `time_range`,
which additionally calculates the list of activities by counting the number of
co-location times for each id in the specified time_range (e.g. 0:timestep:duration)

# Access
* trains(ets) returns list of encoutner trains
* random_encoutner(ets) returns (index_train, index_time) to access trains(ets)[index_train][index_time)
* duration(ets) returns duration of experiment
* timestep(ets) returns timestep of experiment
* ids(ets) returns ids of experiment
* ets[1:3] returns a copy of the object with only the first three contat trains and ids
* ets[BitArray] returns a copy of the object with BitArray applied to trains an ids

"""
struct encounter_trains{I,T}
    _trains::Vector{Vector{T}}  # nested list: outer for participants, inner for the encoutner times of the participant
    _ids::Vector{I}             # 1d list of the ids of participants
    _duration::T                # duration of the experiment
    _timestep::T                # time resolution of the recording
    _cumsum_encounter_for_random_selection::Vector{Int}

    # minimal constructor that processes already calculated trains
    function encounter_trains(trains::Vector{Vector{T}}, ids::Vector{I}, duration, timestep) where {I,T}
        cumsum_encounter = cumsum(length.(trains))
        new{I,T}(trains, ids, duration, timestep, cumsum_encounter)
    end
end
# constructor that extracts trains from list of encounters and passes those to
# the inner constructor
function encounter_trains(list_contacts::Vector{Vector{Tuple{T,T}}}, ids::Vector{I}, duration::T, timestep::T) where {I,T}
    trains = Vector{T}[]
    for contact in list_contacts
        # encounter is a list of tuples (time start, duration) and this
        # extracts only time
        push!(trains, getindex.(contact,1))
    end
    encounter_trains(trains, ids, duration, timestep)
end
# constructor that creates lists of encounters from pairwise co-location times,
# extracts trains, and passes those to the next constructor
function encounter_trains(pairwise_colocation_times::Dict{Tuple{I,I},Vector{T}}, ids::Vector{I}, duration::T, timestep::T) where {I,T}
    @assert length(pairwise_colocation_times) == length(ids)^2

    trains = Vector{T}[]
    @showprogress 1 for id in ids
        train = getindex.(get_contacts(pairwise_colocation_times, ids, id, timestep),1)
        push!(trains, train)
    end

    encounter_trains(trains, ids, duration, timestep)
end

"""
    trains_from_dts(list_dts::Vector{Vector{T}}) where T

convenience function to map a vector (over ids) of vectors of inter-encoutner
times to a vector (over ids) of encoutner times. Because time of first encoutner is
unknown, the most simple convention of encoutner at time zero is assumed.
"""
function trains_from_dts(list_dts::Vector{Vector{T}}) where T
    trains = Vector{T}[]
    for dts in list_dts
        push!(trains, pushfirst!(cumsum(dts),0))
    end
    return trains
end

###############################################################################
# access functions
Base.getindex(ets::encounter_trains{I,T}, idxs::Union{AbstractRange, BitArray}) where {I,T} = encounter_trains(ets._trains[idxs], ets._ids[idxs], ets._duration, ets._timestep)
#
trains(ets::encounter_trains{I,T}) where {I,T} = ets._trains
#
ids(ets::encounter_trains{I,T}) where {I,T} = ets._ids
#
duration(ets::encounter_trains{I,T}) where {I,T} = ets._duration
timestep(ets::encounter_trains{I,T}) where {I,T} = ets._timestep
#
Base.length(ets::encounter_trains{I,T}) where {I,T} = length(trains(ets))
#
Base.lastindex(ets::encounter_trains{I,T}) where {I,T} = lastindex(trains(ets))
# this line makes broadcasting possible by rediricting it to the trains
Broadcast.broadcastable(ets::encounter_trains{I,T}) where {I,T} = trains(ets)

# get random interaction (e.g. for disease start)
#checked to yield identical result as old version
function random_encoutner(rng::AbstractRNG, ets::encounter_trains{I,T}) where {I,T}
    num_encounters_target = rand(rng, 1:ets._cumsum_encounter_for_random_selection[end])

    # find index_train from cumulative number encounters over trains
    index_train = searchsortedfirst(ets._cumsum_encounter_for_random_selection, num_encounters_target)
    if index_train == 1
        num_encounters_past = 0
    else
        num_encounters_past = ets._cumsum_encounter_for_random_selection[index_train-1]
    end

    return (index_train, num_encounters_target - num_encounters_past)
end
#function random_encoutner(rng::AbstractRNG, trains::Vector{Vector{T}}) where T
#    num_encoutners_target = rand(rng, 1:sum(length.(trains)))
#
#    # find index_train, index_time such that
#    num_encoutners = 0
#    for (index_train, train) in enumerate(trains)
#        # find the train in which the num_encoutners_target will be reached (index = [1:length])
#        if num_encoutners + length(train) >= num_encoutners_target
#            return (index_train, num_encoutners_target - num_encoutners)
#        end
#        num_encoutners += length(train)
#    end
#
#    throw(error("something went wrong when locating a random encoutner"))
#end



###############################################################################
###############################################################################
###############################################################################
# tests

function test_encounter_train_vs_contact_activity(;
        experiment=Copenhagen(),
        path="./dat",
    )
    # contact activity
    selected_ids, colocation_times = load_data(experiment, path)
    pairwise_colocation_times = sort_colocation_times_pairwise(colocation_times)
    filter_out_short_contacts!(pairwise_colocation_times, timestep(Copenhagen()), 15*60)
    cas = contact_activities(pairwise_colocation_times, selected_ids, 0:timestep(experiment):duration(experiment))

    # encoutner train
    ets = encounter_trains(pairwise_colocation_times, selected_ids, duration(experiment), timestep(experiment))

    valid = true

    # test if same number of trains are zero
    println("number of zero trains: ", sum(sum.(activities(cas)).== 0) == sum(length.(trains(ets)).== 0))
    valid &= (sum(sum.(activities(cas)).== 0) == sum(length.(trains(ets)).== 0))

    return valid
end

function run_tests()
end

###############################################################################
###############################################################################
###############################################################################
# pauls tinkering

"""
ensure that for every id, in the time_bins where activity is > 0, we also have an entry
in the raw (and filtered) co-location times.

* repeat for activity on the actual raw data
* repeat for encoutner trains
* repeat for encounters ->  is there an easy way to get our encounter representation in code?
    for id in ids:
        for encounter in get_encounters_for_id:
            * check that start time has signal in raw data
            * check that end time has signal in raw data
            * check that _some time between start and end_ has signal in raw data
"""
function test_activity_vs_colocation(;
        experiment=Copenhagen(),
        path="./dat",
    )
    selected_ids, colocation_times = load_data(experiment, path)
    pairwise_colocation_times = sort_colocation_times_pairwise(colocation_times)

    cas = contact_activities(pairwise_colocation_times, selected_ids, 0:timestep(experiment):duration(experiment))

    filter_out_short_contacts!(pairwise_colocation_times, timestep(Copenhagen()), 15*60)
    cas_filtered = contact_activities(pairwise_colocation_times, selected_ids, 0:timestep(experiment):duration(experiment))

    # some trivial sanity checks
    @assert length(ids(cas)) >= length(ids(cas_filtered))
    # @assert that for every id, the sum of activity is <= after filtering
    for (index, id) in enumerate(ids(cas))
        jndex = findall(x->x==id, ids(cas_filtered))[1]
        @assert sum(activities(cas)[index]) >= sum(activities(cas_filtered)[jndex])
    end

    # go over every timestep and check that id is (not) in the raw data
    @showprogress for (tdx, ts) in enumerate(0:timestep(experiment):duration(experiment))
        to_check = colocation_times[colocation_times[:,1].==ts,:]
        # check the unfiltered case
        for (index, id) in enumerate(ids(cas))
            if activities(cas)[index][tdx] == 0
                @assert !(id in to_check[:,2] || id in to_check[:,3])
            else
                @assert (id in to_check[:,2] || id in to_check[:,3])
            end
        end

        # check after filtering, here we cannot check for 0 activity
        for (index, id) in enumerate(ids(cas_filtered))
            if activities(cas_filtered)[index][tdx] > 0
                @assert (id in to_check[:,2] || id in to_check[:,3])
            end
        end
    end
end


"""
simple, artificial cases to test api works as expected
"""
struct Dummy <: ContactData end
label(experiment::Dummy)     = "Dummy"
timestep(experiment::Dummy)  = 500
duration(experiment::Dummy)  = 20*500           # leads to 21 timesteps

function test_dummy_cases()


    experiment = Dummy()
    ts = timestep(experiment)
    passed = true

    selected_ids = [1,2,3]
    colocation_times = [
        1*ts 1 2;
        2*ts 1 2;
        3*ts 1 2;
        5*ts 2 1;   # below separation threshold, but long enough
        6*ts 2 1;
        11*ts 1 2;  # above threshold
        12*ts 1 2;
        12*ts 1 3;  # two encoutners -> activity 2
        12*ts 2 3;
        13*ts 1 2;
        13*ts 1 3;
        13*ts 2 3;
        14*ts 1 2;
        14*ts 1 3;
        14*ts 2 3;
        15*ts 1 2;
        15*ts 1 3;
        15*ts 2 3;
        19*ts 2 3;  # filtered because too short
    ]

    println("\nBefore Filtering:")
    pairwise_colocation_times = sort_colocation_times_pairwise(colocation_times)
    cas = contact_activities(pairwise_colocation_times, selected_ids, 0:timestep(experiment):duration(experiment))

    expected = Dict()
    expected[1] =
    #0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0
    [0,1,1,1,0,1,1,0,0,0,0,1,2,2,2,2,0,0,0,0,0]
    expected[2] =
    #0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0
    [0,1,1,1,0,1,1,0,0,0,0,1,2,2,2,2,0,0,0,1,0]
    expected[3] =
    #0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0
    [0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,0,0,0,1,0]

    for (idx, i) in enumerate(ids(cas))
        println("id: ", i, "\nExpected | Found")
        for t in 1:length(expected[idx])
            println(expected[idx][t], " | ", activities(cas)[idx][t])
        end
        if timeseries(activities(cas)[idx]) == expected[idx]
            println("check")
        else
            println("failed")
            passed = false
        end
    end

    println("\nFiltering Nothing:")
    # dummy filter, filtering for zero should not change data
    # minimum duration of 0 -> no filtering
    pairwise_colocation_times = sort_colocation_times_pairwise(colocation_times)
    filter_out_short_encounter!(pairwise_colocation_times, 1*ts, 0*ts)
    cas = contact_activities(pairwise_colocation_times, selected_ids, 0:timestep(experiment):duration(experiment))

    for (idx, i) in enumerate(ids(cas))
        println("id: ", i, "\nExpected | Found")
        for t in 1:length(expected[idx])
            println(expected[idx][t], " | ", activities(cas)[idx][t])
        end
        if timeseries(activities(cas)[idx]) == expected[idx]
            println("check")
        else
            println("failed")
            passed = false
        end
    end

    println("\nFiltering duration 1step:")
    # now filter > 1 timesteps length -> only interactions longer or equal to 2 timesteps are kept
    pairwise_colocation_times = sort_colocation_times_pairwise(colocation_times)
    filter_out_short_encounter!(pairwise_colocation_times, 1*ts, 1*ts)
    cas = contact_activities(pairwise_colocation_times, selected_ids, 0:timestep(experiment):duration(experiment))

    expected = Dict()
    expected[1] =
    #0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0
    #0,1,1,1,0,1,1,0,0,0,0,1,2,2,2,2,0,0,0,0,0 # unfiltered
    [0,1,1,1,0,1,1,0,0,0,0,1,2,2,2,2,0,0,0,0,0]
    expected[2] =
    #0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0
    #0,1,1,1,0,1,1,0,0,0,0,1,2,2,2,2,0,0,0,1,0] # unfiltered
    [0,1,1,1,0,1,1,0,0,0,0,1,2,2,2,2,0,0,0,0,0] # filter the length-1 event at the end
    expected[3] =
    #0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0
    #0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,0,0,0,1,0] # unfiltered
    [0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,0,0,0,0,0] # filter the length-1 event at the end

    for (idx, i) in enumerate(ids(cas))
        println("id: ", i, "\nExpected | Found")
        for t in 1:length(expected[idx])
            println(expected[idx][t], " | ", activities(cas)[idx][t])
        end
        if timeseries(activities(cas)[idx]) == expected[idx]
            println("check")
        else
            println("failed")
            passed = false
        end
    end

    println("\nFiltering duration 2step:")
    # now filter > 2 timesteps length -> only interactions longer or equal to 3 timesteps are kept
    pairwise_colocation_times = sort_colocation_times_pairwise(colocation_times)
    filter_out_short_encounter!(pairwise_colocation_times, 1*ts, 2*ts)
    cas = contact_activities(pairwise_colocation_times, selected_ids, 0:timestep(experiment):duration(experiment))

    expected = Dict()
    expected[1] =
    #0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0
    #0,1,1,1,0,1,1,0,0,0,0,1,2,2,2,2,0,0,0,0,0 # unfiltered
    [0,1,1,1,0,0,0,0,0,0,0,1,2,2,2,2,0,0,0,0,0]
    expected[2] =
    #0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0
    #0,1,1,1,0,1,1,0,0,0,0,1,2,2,2,2,0,0,0,1,0] # unfiltered
    [0,1,1,1,0,0,0,0,0,0,0,1,2,2,2,2,0,0,0,0,0] # also filter the length-2 event in the beginning
    expected[3] =
    #0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0
    #0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,0,0,0,1,0] # unfiltered
    [0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,0,0,0,0,0] # also filter the length-2 event in the beginning

    for (idx, i) in enumerate(ids(cas))
        println("id: ", i, "\nExpected | Found")
        for t in 1:length(expected[idx])
            println(expected[idx][t], " | ", activities(cas)[idx][t])
        end
        if timeseries(activities(cas)[idx]) == expected[idx]
            println("check")
        else
            println("failed")
            passed = false
        end
    end

    if passed
        println("\nAll checks out")
    else
        println("\nTest failed")
    end

    return passed

end
