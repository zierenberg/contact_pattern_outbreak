using StatsBase
using ProgressMeter


###############################################################################
###############################################################################
### jackknife error analysis
### based on the paper 'delete-m jackknife for unequal m' by Busing et al.
global skip_jackknife = false


"""
    jackknife(f, data[, naive=missing])

returns the naive estimate f(data) (which is evaluated by the function if not
provided), the jackknife estimate, and the jackknife error.

The jackknife estimate and error are calculated over samples of data that omit
single elements. If blocking is intended, then data should be a vector of
blocks of data. In this case all blocks have to have the same length! (see
jackknife_mj otherwise). This function works for functions `f` with scalar or
vector output.
"""
function jackknife(f, data; naive=missing)
    if ismissing(naive)
        naive = f(data)
    end
    if skip_jackknife
        return naive, fill(NaN, size(naive)), fill(NaN, size(naive))
    end
    g = length(data)
    est = Vector{typeof(naive)}(undef, g)
    p = Progress(g, 2, "Jackknife: ", offset=0)
    for j in 1:g
        est[j] = f(data[1:end .!= j])
        next!(p)
    end
    mean_est = mean(est)
    est_J = g*naive .- (g-1)*mean_est

    argument(est) = (est-mean_est).^2
    var_J = sum(argument, est).*((g-1)/g)
    return naive, est_J, sqrt.(var_J)
end

# delete mj jackknife
"""
    jackknife_mj(f, data[, naive=missing])

returns the naive estimate f(data) (which is evaluated by the function if not
provided), the jackknife estimate, and the jackknife error.

Same as `jackknife` but data can be vector of blocks of different length.
However, none of the blocks may have length zero, so these have to be filtered
out beforehand.
"""
function jackknife_mj(f, data; naive=missing)
    if ismissing(naive)
        naive = f(data)
    end
    if skip_jackknife
        return naive, fill(NaN, size(naive)), fill(NaN, size(naive))
    end
    g = length(data)
    m = length.(data)
    @assert sum(m.==0)==0
    h = sum(m)./m
    est_tilde = Vector{typeof(naive)}(undef, g)
    p = Progress(g, 2, "Jackknife: ", offset=0)
    for j in 1:g
        est_tilde[j] = h[j]*naive .- (h[j]-1)*f(data[1:end .!= j])
        next!(p)
    end
    est_J = sum(est_tilde./h)

    # define summation argument in place to work also for vector quantities
    argument((est_tilde_j, h_j)) = (est_tilde_j .- est_J).^2 / (h_j-1)
    var_J = sum( x->argument(x), zip(est_tilde, h) ) /g

    return naive, est_J, sqrt.(var_J)
end


###############################################################################
# tests

function test_jackknife()
    passed = true
    data=[[1,2,3], [1,1,1], [2,2,2], [1,2,4]]

    # scalar
    result_jackknife = jackknife(x->mean(sum.(x)), data)
    result_jackknife_mj = jackknife_mj(x->mean(sum.(x)), data)
    println(result_jackknife, " ", result_jackknife_mj)
    passed &= sum(result_jackknife .- result_jackknife_mj) < 1e-5
    println(passed)

    # vector
    result_jackknife = jackknife(x->mean(diff.(x)), data)
    result_jackknife_mj = jackknife_mj(x->mean(diff.(x)), data)
    println(result_jackknife, " ", result_jackknife_mj)
    passed &= sum(sum.(result_jackknife .- result_jackknife_mj)) < 1e-5

    # reshaping arrays
    num_blocks = 30
    max_len = 40
    rng = MersenneTwister(1000)
    random_data = [randn(rng,Float64, rand(rng,2:max_len)) for i in 1:num_blocks]
    flat = vcat(random_data...)
    block_len = Int(floor(length(flat)/num_blocks))+1
    equal_blocks = [ flat[i*block_len+1:min((i+1)*block_len,length(flat))] for i in 0:Int(floor(length(flat)/block_len)) ]
    result_jackknife = jackknife(x->sum(sum.(x))/sum(length.(x)),equal_blocks)
    result_jackknife_mj = jackknife_mj(x->sum(sum.(x))/sum(length.(x)),random_data)
    println(result_jackknife, " ", result_jackknife_mj)
    passed &= sum(sum.(result_jackknife .- result_jackknife_mj)) < 1e-3

    return passed
end
