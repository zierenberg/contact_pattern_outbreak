using Distributions
using StatsBase

"""
    branching_process(dist, N_0, N_max, T)
sample branching process with arbitrary offspring distribution for either `T`
generations or until `N` succeeds `N_max`.

"""
function branching_process(rng, offspring_dist, N_0::Int, T::Int;
    )
    N_T = zeros(Int, T+1)
    N_T[1] = N_0
    branching_process!(rng, N_T, offspring_dist, p=p)
    return N_T
end
function branching_process!(rng, N_T::Vector{Int}, offspring_dist
    )
    for t in 1:length(N_T)-1
        N_T[t+1] = branching_step(rng, N_T[t], offspring_dist)
    end
end

function branching_step(rng, N::Int, offspring_dist)::Int
    N_ = 0
    for i in 1:N
        N_ += rand(rng, offspring_dist)
    end
    return N_
end


"""

step needs to be a function of type x->function(rng, ..., x,...)
"""
function check_survival(rng, step::Function, x0, xmax;
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


"""

generate offspings for N
TODO: chekc type stability with @code_warntype; problem is that we only want
distributions that yield integers...
"""
function generate_offspring(rng, dist, p::Number)::Int
    X = rand(rng, dist)
    return rand(rng, Binomial(X,p))
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
struct ProbabilisticOffspringDistribution{T}
    dist::EmpiricalDistribution{T}
    probability::Float64
end
function Base.rand(rng::AbstractRNG, pdist::ProbabilisticOffspringDistribution{T}) where T
    X = rand(rng, pdist.dist)
    return rand(rng, Binomial(X,pdist.probability))
end
Base.rand(pdist::ProbabilisticOffspringDistribution) = rand(Random.GLOBAL_RNG, pdist)


