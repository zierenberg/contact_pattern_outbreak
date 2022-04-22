using Random
using StatsBase
using Distributions

import StatsBase.mean
import StatsBase.kldivergence


#increment is the factor by which bin increases (casted to integer)
#normalize at end
#interpret P[i] as number of elements in bin i (can be normalized in advance ...)
#this is approximate because int P(x)dx ~ sum P(x) delta x (what we do here)
"""
calculate a log-binned distribution from an evenly space distribution `P(x)`
where bins increase with `inrement_factor`

# Remark
`P` hast to be a probability density
"""
function logbin(x, P; increment_factor=sqrt(2), bin_width_start=1)
    @assert length(x)==length(P)

    dx = x[2]-x[1]

    counter   = 0
    index     = 1
    x_bin = zeros(1)
    P_bin = zeros(1)
    w_bin = ones(1)*floor(Int, bin_width_start)

    bin_width::Float64 = w_bin[end]
    for (i,p) in enumerate(P)
        #print(i, " ", index, " ", x[i], " ", P[i], " ", x_bin[index], " ", P_bin[index], " ",  w_bin[index], "\n")
        # window full; create new window
        if counter == w_bin[end]
            index += 1
            counter = 0
            bin_width *= increment_factor
            push!(x_bin, 0)
            push!(P_bin, 0)
            push!(w_bin, floor(Int, bin_width))
        end

        counter += 1
        # towards expectation value of x in log bin
        x_bin[index] += x[i]*P[i]
        # probabilities of values over bin (sum of equally spaced probabilities)
        P_bin[index] += P[i]
    end
    # normalize x bins (end of expectation value)
    x_bin ./= P_bin

    # transform P into propert probability DENSITY such that <x>=\sum xP(x)dx/sum P(x)dx
    P_bin ./= w_bin

    return x_bin, P_bin, w_bin.*dx
end
logbin(dist; increment_factor=sqrt(2), bin_width_start=1) = logbin(collect(dist.edges[1])[1:end-1], dist.weights, increment_factor=increment_factor, bin_width_start=bin_width_start)

function calculate_cdf(dist)
    x = collect(dist.edges[1])
    dx = x[2]-x[1]
    cdf = similar(x, Float64)
    cdf[1] = 0.0
    for i in 2:length(cdf)
        cdf[i] = cdf[i-1] + dist.weights[i-1]*dx
    end
    return x, cdf
end

###############################################################################
# convenient Histogram access as hist[value] or even hist[value] += number entries
Base.getindex(h::AbstractHistogram{T,1}, x::Real) where {T} = getindex(h, (x,))
function Base.getindex(h::Histogram{T,N}, xs::NTuple{N,Real}) where {T,N}
    idx = StatsBase.binindex(h, xs)
    if checkbounds(Bool, h.weights, idx...)
        return @inbounds h.weights[idx...]
    else
        return missing
    end
end

Base.setindex!(h::AbstractHistogram{T,1}, value::Real, x::Real) where {T} = setindex!(h, value, (x,))
function Base.setindex!(h::Histogram{T,N}, value::Real, xs::NTuple{N,Real}) where {T,N}
    h.isdensity && error("Density histogram must have float-type weights")
    idx = StatsBase.binindex(h, xs)
    if checkbounds(Bool, h.weights, idx...)
        @inbounds h.weights[idx...] = value
    end
end


###############################################################################
"""
    kldivergence(P::AbstractHistogram, Q::AbstractHistogram)

convenience wrapper to estimate kl divergence of empirical distributions.
Ensures consistent domain, ensures normalization, and avoids zero-entry issues.

Kullback-Leibler divergence is not symmetric. Assumes that Q is reference
distribution that we want to compare to.

```
  sum_{i} P_i * log(P_i/Q_i)
```

If Q_i are zero, we omit the entry.
"""
function kldivergence(P::AbstractHistogram, Q::AbstractHistogram)
    @assert P.edges[1] == Q.edges[1]
    @assert abs(sum(P.weights)*step(P.edges[1]) - 1) < 1e-10
    @assert abs(sum(Q.weights)*step(Q.edges[1]) - 1) < 1e-10

    kld = 0
    for i in 1:length(P.weights)
        P_i = P.weights[i]
        if P_i > 0
            Q_i = Q.weights[i]
            if Q_i > 0
                kld += P_i * log(P_i/Q_i)
            end
        end
    end
    return kld
end
