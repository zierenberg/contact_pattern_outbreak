using Distributions
using StatsBase

"""
    branching_process(rng, offspring_dist, N_0, N_max, T)

Sample branching process with arbitrary `offspring distribution` for `T`
generations starting from `N_0` units at generation 1.

"""
function branching_process(rng, offspring_dist, N_0::Int, T::Int;
    )
    N_T = zeros(Int, T+1)
    N_T[1] = N_0
    branching_process!(rng, N_T, offspring_dist, p=p)
    return N_T
end
"""
    branching_process(rng, N_T, offspring_dist)

Sample branching process for `length(N_T)` generations starting with initial
number of units from `N_T[1]` and generating for each unit offsprings from
`offspring_dist`.
"""
function branching_process!(rng, N_T::Vector{Int}, offspring_dist
    )
    @assert N_T[1] > 0
    @assert N_T[2:end] .= 0
    for t in 1:length(N_T)-1
        N_T[t+1] = branching_step(rng, N_T[t], offspring_dist)
    end
end


"""
    branching_step(rng, N, offspring_dist)

Generate for each of the `N` units offsprings according to `offspring distribution`.

# Input
* rng = random number generator
* N = integer number
* offspring_dist = abstract distribution for which rand(rng, offspring_dist)
generates integer response
"""
function branching_step(rng, N::Int, offspring_dist)::Int
    N_ = 0
    for i in 1:N
        N_ += rand(rng, offspring_dist)
    end
    return N_
end


"""
    check_survival(rng, branching_step, x0, xmax)

Check if branching process with function `branching_step` starting from
generation 1 with `x0` units survives.
Survival is here defined as suceeding xmax, which works well if not directly at
the critical point.

# Input
* branching_step needs to be a function of type x->function(rng, ..., x, ...)
step needs to be a function of type x->function(x)
* currently x0=N_0 and xmax=N_max is tested, this may generalize.
"""
function check_survival(step::Function, x0, xmax;
        T_max = Int(1e5)
    )
    x = x0
    T = 1
    for T in 1:T_max
        x = step(x)
        if x == 0
            return false, T
        elseif x > xmax
            return true, T
        end
    end
    return true, T
end




##############
struct EmpiricalDistribution{T}
    values::T
    probabilities::ProbabilityWeights{Float64,Float64,Vector{Float64}}

    function EmpiricalDistribution(values::T, probabilities::Vector{F}) where {T,F<:Number}
        new{T}(values, ProbabilityWeights(probabilities))
    end
end
function EmpiricalDistribution(dist::Histogram{Float64})
    @assert dist.isdensity
    dx = step(dist.edges[1])
    values = dist.edges[1][1:end-1]
    EmpiricalDistribution(values, (dist.weights .* dx))
end
EmpiricalDistribution(hist::Histogram{Int}) = EmpiricalDistribution(normalize!(float(hist)))

"""
custom version of random sampling from custom distribution using StatsBase
"""
Base.rand(rng::AbstractRNG, edist::EmpiricalDistribution) = sample(rng, edist.values, edist.probabilities)
Base.rand(edist::EmpiricalDistribution) = rand(Random.GLOBAL_RNG, edist)

"""
estimate of expectation value from empirical distribution
"""
expectation(edist::EmpiricalDistribution) = sum(edist.values .* edist.probabilities)

##############
"""
    ProbabilisticOffspringDistribution

This custom object `pdist` generates offsprings in two steps: First it
generates potentially infectious encounter from pdist.dist, and second it
selects infectious encounters with probability pdist.probability

Use as rand(rng, pdist)

# Remark
pdist.dist can be an `EmpiricalDistribution`.
"""
struct ProbabilisticOffspringDistribution{T}
    dist::T
    probability::Float64
end
function Base.rand(rng::AbstractRNG, pdist::ProbabilisticOffspringDistribution{T}) where T
    X = rand(rng, pdist.dist)
    return rand(rng, Binomial(X,pdist.probability))
end
Base.rand(pdist::ProbabilisticOffspringDistribution) = rand(Random.GLOBAL_RNG, pdist)


