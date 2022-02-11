using Distributions
using StatsBase

"""
    branching_process(dist, N_0, N_max, T)
sample branching process with arbitrary offspring distribution for either `T`
generations or until `N` succeeds `N_max`.

"""
function branching_process(rng, dist, N_0::Int, T::Int;
        p::Number=1.0
    )
    N_T = zeros(Int, T+1)
    N_T[1] = N_0
    branching_process!(rng, N_T, dist, p=p)
    return N_T
end
function branching_process!(rng, N_T::Vector{Int}, dist;
        p::Number=1.0
    )
    for t in 1:length(N_T)-1
        N = N_T[t]
        N_ = 0
        for i in 1:N
            N_ += generate_offspring(rng, dist, p)
        end
        N_T[t+1] = N_
    end
end


"""

generate offspings for N
TODO: chekc type stability with @code_warntype; problem is that we only want
distributions that yield integers...
"""
function generate_offspring(rng, dist, p::Number)::Int
    if p==1.0
        return rand(rng, dist)
    else
        X = rand(rng, dist)
        return rand(rng, Binomial(X,p))
    end
end



##############
struct EmpiricalDistribution
    values
    probabilities::ProbabilityWeights

    function EmpiricalDistribution(dist::Histogram{Float64})
        @assert dist.isdensity
        dx = step(dist.edges[1])
        new(dist.edges[1][1:end-1], ProbabilityWeights(dist.weights .* dx))
    end

    function EmpiricalDistribution(values, probabilities::Vector{Number})
        new(values, ProbabilityWeights(probabilities))
    end
end
EmpiricalDistribution(hist::Histogram{Int}) = EmpiricalDistribution(normalize!(float(hist)))

"""
custom version of random sampling from custom distribution using StatsBase
"""
Base.rand(rng::AbstractRNG, dist::EmpiricalDistribution) = sample(rng, dist.values, dist.probabilities)
Base.rand(dist::EmpiricalDistribution) = sample(Random.GLOBAL_RNG, dist.values, dist.probabilities)


#rand(rnd::AbstractRNG, dist::AbstractHistogram) = sample(rng, dist.edges[1][1:end-1], dist.weights)
#rand(rnd::AbstractRNG, dist::AbstractHistogram, n::Int) = sample(rng, dist.edges[1][1:end-1], dist.weights, n)

#"""
#    rand(rng, dist, [n])
#
#randomly draw from a histogram/distribution
#
#"""
#rand(rng::AbstractRNG, dist::AbstractHistogram) = sample(rng, dist.edges[1], dist.weights)
#rand(rng::AbstractRNG, dist::AbstractHistogram, n::Int) = sample(rng, dist.edges[1], dist.probabilities, n)
#
#mean(dist::AbstractHistogram) = mean(dist.edges[1], dist.weights)

