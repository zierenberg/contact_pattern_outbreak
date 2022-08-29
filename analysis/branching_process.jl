using Distributions
using StatsBase
using StatsFuns

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
"""
    EmpiricalDistribution{T}

empirical distribution that quantifies the probability for a specific outcome.
Convention is that the probability for values that are not specified is zero.
"""
struct EmpiricalDistribution{T}
    values::T
    probabilities::ProbabilityWeights{Float64,Float64,Vector{Float64}}

    function EmpiricalDistribution(values::T, probabilities::Vector{F}) where {T,F<:Number}
        new{T}(values, ProbabilityWeights(probabilities))
    end
end
function EmpiricalDistribution(hist::Histogram{Float64})
    dist = normalize!(hist, mode=:probability)
    values = dist.edges[1][1:end-1]
    EmpiricalDistribution(values, (dist.weights))
end
EmpiricalDistribution(hist::Histogram{Int}) = EmpiricalDistribution(float(hist))
# access edist[x]
function Base.getindex(edist::EmpiricalDistribution{T}, x::Real) where {T}
    idx = findfirst(val->val==x, edist.values)
    if isnothing(idx)
        # Convention: If the value is not specified, then probability is zero.
        return 0
    else
        return @inbounds edist.probabilities[idx]
    end
end

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








"""
    offspring_distribution(edist, p)

empirical offspring distribution from empirical distribution of pot. inf. encounters.

#example
_, ets_data, _ = load_processed_data();
ets_rand = surrogate_randomize_per_train(ets_data, 1000);
T_lat = 2; T_ift = 3;
disease_model = DeltaDiseaseModel(seconds_from_days(T_lat), seconds_from_days(T_ift))
dist_data = distribution_from_samples_infectious_encounter(
            samples_infectious_encounter(disease_model, ets_data)
       );
dist_rand = distribution_from_samples_infectious_encounter(
            samples_infectious_encounter(disease_model, ets_rand)
       );
edist_data = EmpiricalDistribution(dist_data);
edist_rand = EmpiricalDistribution(dist_rand);
R0 = 3
p_data = R0/expectation(edist_data);
p_rand = R0/expectation(edist_rand);
offspring_dist_data = offspring_distribution(edist_data,p_data);
offspring_dist_rand = offspring_distribution(edist_rand,p_rand);
"""
function offspring_distribution(edist::EmpiricalDistribution{T}, p::Number) where T
    if p > 1
        return NaN
    end
    offspring_values = collect(0:maximum(edist.values))
    offspring_probs = zeros(length(offspring_values))
    for (i,x) in enumerate(offspring_values)
        for n in x:edist.values[end]
            offspring_probs[i] += edist[n] * binompdf(n, p, x)
        end
    end
    offspring_dist = EmpiricalDistribution(offspring_values, offspring_probs)
    return offspring_dist
end

"""
    solve_survival_probability(edist, p)

semi-analytical solution of the survival probability using the empirical
distribution of potentially infectious encounters `edist`=P(n). The method assumes
that infections are independent with probability `p` such that we can
formulate an empirical probability generating function

```math
  \\pi(\\theta) = \\sum_{x=0}^\\infty \\sum_{n=x}^\\infty P(n) \\binom{n}{x} p^x (1-p)^{n-x}\\theta^x
```

The asymptotic survival probability is then obtained by numerically solving

``\\theta=\\pi(\\theta)``

"""
function solve_survival_probability(edist::EmpiricalDistribution{T}, p::Number) where T
    offspring_dist = offspring_distribution(edist, p)

    # extinction probability is given by the smalles root to x-pfg(x)
    p_ext = find_zero(x->x - _pgf(x, offspring_dist), 0.0)

    # by definition one root is at x=1, so that needs to be excluded.
    return 1 - p_ext
end


"""
    fit_negative_binomial(offspring_dist)

_, ets_data, _ = load_processed_data();
ets_rand = surrogate_randomize_per_train(ets_data, 1000);
T_lat = 2; T_ift = 3;
disease_model = DeltaDiseaseModel(seconds_from_days(T_lat), seconds_from_days(T_ift))
dist_data = distribution_from_samples_infectious_encounter(
            samples_infectious_encounter(disease_model, ets_data)
       );
dist_rand = distribution_from_samples_infectious_encounter(
            samples_infectious_encounter(disease_model, ets_rand)
       );
edist_data = EmpiricalDistribution(dist_data);
edist_rand = EmpiricalDistribution(dist_rand);
R0 = 3
p_data = R0/expectation(edist_data);
p_rand = R0/expectation(edist_rand);
offspring_dist_data = offspring_distribution(edist_data,p_data);
offspring_dist_rand = offspring_distribution(edist_rand,p_rand);

rng=MersenneTwister(1000); samples_data = [rand(rng, offspring_dist_data) for i in 1:Int(1e4)];
rng=MersenneTwister(1000); samples_rand = [rand(rng, offspring_dist_rand) for i in 1:Int(1e4)];
using Optim
res_data = optimize(x->-1*sum(log.(pdf.(NegativeBinomial(x[1],x[2]),samples_data))), [0,0.01], [Inf,1], [1,0.1])
res_rand = optimize(x->-1*sum(log.(pdf.(NegativeBinomial(x[1],x[2]),samples_rand))), [0,0.01], [Inf,1], [1,0.1])
NB_data = NegativeBinomial(Optim.minimizer(res_data)...)
NB_rand = NegativeBinomial(Optim.minimizer(res_rand)...)
using Plots
plot(xlabel="offspring",xlims=(0,20),xticks = 0:5:20)
plot!(offspring_dist_data.values, offspring_dist_data.probabilities, label="empirical distribution")
plot!(offspring_dist_data.values, pdf.(NB_data,offspring_dist_data.values), label="Negative Binomial fit")
plot!(offspring_dist_rand.values, offspring_dist_rand.probabilities, label="empirical distribution")
plot!(offspring_dist_rand.values, pdf.(NB_rand,offspring_dist_rand.values), label="Negative Binomial fit")
"""


function _pgf(theta, offspring_dist)
    return sum(theta.^offspring_dist.values .* offspring_dist.probabilities)
end
