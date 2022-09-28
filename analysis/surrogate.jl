using Random

include("load_data.jl")

function surrogate_randomize_per_train(
        ets::encounter_trains{I,T},
        seed::Int
    ) where {I,T}
    rng = MersenneTwister(seed)
    times_surrogate = similar(trains(ets))
    for (i,train) in enumerate(trains(ets))
        times_surrogate[i] = sort(rand(rng, 0:timestep(ets):duration(ets), length(train)))
    end
    return encounter_trains(times_surrogate, ids(ets), duration(ets), timestep(ets))
end

function surrogate_randomize_all(
        ets::encounter_trains{I,T},
        seed::Int
    ) where {I,T}
    rng = MersenneTwister(seed)

    #create empty trains
    times_surrogate = [T[] for i in 1:length(ets)]
    for i in 1:sum(length.(trains(ets)))
        rand_train = rand(rng, 1:length(ets))
        rand_time  = rand(rng, 0:timestep(ets):duration(ets))
        push!(times_surrogate[rand_train], rand_time)
    end
    # sort trains in place
    for i in 1:length(times_surrogate)
        sort!(times_surrogate[i])
    end
    return encounter_trains(times_surrogate, ids(ets), duration(ets), timestep(ets))
end
