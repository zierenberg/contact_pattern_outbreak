using Random
using StatsBase
using Distributions
using DataStructures

using LsqFit

###############################################################################
###############################################################################
### spread in mean-field population with offsprings generated by conditional encounter rate

"""
initial_infection_times can be specified as an array of times where infections
did occur already or will occur
"""
function spread_mean_field(
        disease_model::DeltaDiseaseModel,
        cond_encounter_rate::AbstractHistogram,
        probability_infection::Float64,
        measurement_times::Vector{T};
        # optional
        max_cases = Inf,
        initial_infection_times::Vector{Float64}=[0.0],
        seed::Int=1000,
        verbose=false
    ) where {T<:Number}
    rng = MersenneTwister(seed)

    # for delta disease the infectious interval is fixed
    interval_infectious  = infectious_interval(disease_model)
    # make sure that the conditional encounter rate is defined over the infectious interval
    @assert cond_encounter_rate.edges[1][1] <= interval_infectious[1]
    @assert cond_encounter_rate.edges[1][end-1] > interval_infectious[2]

    note=
    """
    data = samples_infectious_encounter(disease_model, ets)
    """

    measured_new_infections = Histogram(measurement_times)

    # define a HeapMap that stores events and keeps track of next one to occur
    # (smallest event in lexographical order: first smallest time, then
    # smallest event_type)
    infections = MutableBinaryMinHeap{Float64}()

    # add initial infections
    for time in initial_infection_times
        push!(infections, time)
    end
    # add infinity as final time
    push!(infections, Inf)

    time_last_measurement = measurement_times[end]
    time = -Inf
    sum_offsprings = 0
    sum_avg_Tgen = 0
    sum_samples = 0
    while time < time_last_measurement
        # get next infection time (sorted array)
        time = pop!(infections)
        push!(measured_new_infections, time)

        # check if the number of infections that accumulate at this time exceed limit
        # and if it exceeds, then make all
        index_record = StatsBase.binindex(measured_new_infections, time)
        if index_record > 0
            if measured_new_infections.weights[index_record] > max_cases
                if verbose
                    println("abort because case numbers exceed limit (set all remaining values to zero)")
                end
                measured_new_infections.weights[index_record:end] .= 0
                break
            end
        end

        # generate inhomogeneous poisson process from cond. encounter rate
        # (which is dt from the infection)
        dt_potential_secondary_infections = inhomogeneous_poisson_process(cond_encounter_rate, 0.0, interval_infectious, rng)
        # and add each contact within infectious interval with probability to the
        # list of infections (count num_offsprings for estimate of R0)
        num_offsprings = 0
        avg_Tgen = 0
        for dt in dt_potential_secondary_infections
            if rand(rng) < probability_infection
                push!(infections, time + dt)
                num_offsprings += 1
                avg_Tgen += dt
            end
        end
        sum_avg_Tgen += avg_Tgen
	    sum_offsprings += num_offsprings
	    sum_samples += 1
    end

    return measured_new_infections, sum_avg_Tgen, sum_offsprings, sum_samples
end


###############################################################################
###############################################################################
### encounter trains

"""
sample_encounter_trains_poisson(rate, num_sample_trains, time_start, interval_record, rng, [weights=missing], [timestep=0])

sample `num_sample_trains` encounter trains where encoutner times are generated
with a Poisson process with rate `rate`. If rate is a real number, then this is
a homogeneous poisson process.  If rate is an abstract histogram, then this is
an inhomogeneous Poisson per train with periodic boundary conditions in time
(edges of histogram).

In addition, the rate can be weighted per train with `weights` for which it is
convenient that num_sample_trains is a multiple of weights.

In addition, a timestep can be specified that is typically used as intrinsic
discretization of distributions where the natural timestep of the experiment is
used. Should be tailored to experimenta that one ones to compare with.
"""
function sample_encounter_trains_poisson(
        rate::Real,
        num_sample_trains::Int,
        time_start::Real,
        interval_record::Tuple{Real,Real},
        rng::AbstractRNG;
        weights=missing,
        timestep=0,
    )
    if ismissing(weights)
        encounter_times = [poisson_process(rate, time_start, interval_record, rng) for i in 1:num_sample_trains];
    else
        encounter_times = [poisson_process(rate*weights[mod1(i,end)], time_start, interval_record, rng) for i in 1:num_sample_trains];
    end
    return encounter_trains(encounter_times, collect(1:num_sample_trains), interval_record[2]-interval_record[1], timestep);
end
function sample_encounter_trains_poisson(
        rate::AbstractHistogram,
        num_sample_trains::Int,
        time_start::Real,
        interval_record::Tuple{Real,Real},
        rng::AbstractRNG;
        # optional
        weights=missing,
        timestep=0,
    )
    if ismissing(weights)
        encounter_times = [inhomogeneous_poisson_process(rate, time_start, interval_record, rng) for i in 1:num_sample_trains];
    else
        encounter_times = [inhomogeneous_poisson_process(rate, time_start, interval_record, rng, weight=weights[mod1(i,end)]) for i in 1:num_sample_trains];
    end
    return encounter_trains(encounter_times, collect(1:num_sample_trains), interval_record[2]-interval_record[1], timestep);
end


"""
    sample_encounter_trains_weibull_renewal(args_weibull, num_sample_trains, time_start, interval_record, rng, [weights=missing], [timestep=0]))

sample `num_sample_trains` encounter trains where encoutner times are generated
with a Weibull renewal process with `args_weilbull`.

In addition, the expected rate per train can be weighted with `weights` for
which it is convenient that num_sample_trains is a multiple of weights.

In addition, a timestep can be specified that is typically used as intrinsic
discretization of distributions where the natural timestep of the experiment is
used. Should be tailored to experiment that one aims to compare with.
"""
function sample_encounter_trains_weibull_renewal(
        args_weibull::NamedTuple{(:shape, :scale)},
        num_sample_trains::Int,
        time_start::Real,
        interval_record::Tuple{Real,Real},
        rng::AbstractRNG;
        weights=missing,
        timestep=0,
    )
    shape, scale = args_weibull
    if ismissing(weights)
        encounter_times = [renewal_process(Weibull(shape, scale), time_start, interval_record, rng) for i in 1:num_sample_trains];
    else
        # adjust mean = scale*Gamma(1+1/shape) inversely to weight by preserving shape, i.e. by adjusting scale as scale/weight (increased weight means incrased rate means decreased mean)
        # rate = 1/mean
        # rate*weight = weight/mean = weight /scale *Gamma(1+1/shape) = Gamma(1+1/scale)/scale_new -> scale_new = scale/weight
        encounter_times = [renewal_process(Weibull(shape, scale/weights[mod1(i,end)]), time_start, interval_record, rng) for i in 1:num_sample_trains];
    end
        encounter_times = [renewal_process(Weibull(shape, scale), time_start, interval_record, rng) for i in 1:num_sample_trains];
    return encounter_trains(encounter_times, collect(1:num_sample_trains), interval_record[2]-interval_record[1], timestep);
end


"""
    sample_encounter_trains_tailored(args_weibull, num_sample_trains, time_start, interval_record, rng, [weights=missing], [timestep=0]))

sample encounter trains with ``thinned'' Weibull renewal processes of
heterogeneous rates tailored to reproduce contact statistics from data.
"""
function sample_encounter_trains_tailored(
        rng::AbstractRNG,
        shape_weibull::Number,
        ref_rate::AbstractHistogram,
        train_weights::Vector{T},
        time_start::Real,
        interval_record::Tuple{Real,Real};
        timestep=step(ref_rate.edges[1])
    ) where {T}
    rate_max = maximum(ref_rate.weights)
    # convert to probability to accept encounters
    time_prob = deepcopy(ref_rate)
    time_prob.weights /= rate_max

    ets_sample, scale_weibull = sample_encounter_trains_tailored(rng,
                                                                 shape_weibull,
                                                                 rate_max,
                                                                 time_prob,
                                                                 train_weights,
                                                                 time_start,
                                                                 interval_record,
                                                                 timestep)
    return ets_sample, scale_weibull
end
function sample_encounter_trains_tailored(
        rng::AbstractRNG,
        shape_weibull::Number,
        rate_max::Number,
        time_prob::AbstractHistogram,
        train_weights::Vector{T},
        time_start::Real,
        interval_record::Tuple{Real,Real},
        timestep;
    ) where {T}
    # calculate weibull scale parameter to match max rate
    scale_weibull = 1/(rate_max*gamma(1+1/shape_weibull))

    # generate encounter times with cyclic renewal processes
    encounter_times = [cyclic_renewal_process(time_prob,
                                              Weibull(shape_weibull, scale_weibull/train_weights[i]),
                                              time_start,
                                              interval_record,
                                              rng)
                        for i in 1:length(train_weights)
                       ];
    ets_sample = encounter_trains(encounter_times,
                                  collect(1:length(train_weights)),
                                  interval_record[2]-interval_record[1],
                                  timestep);

    return ets_sample, scale_weibull
end

function _update(rng::AbstractRNG, value, dvalue)
    return value + (rand(rng)*2-1)*dvalue
end

function _update(rng::AbstractRNG, value, dvalue, range::Tuple)
    value_new = value + (rand(rng)*2-1)*dvalue
    if (getindex(range,1) <= value_new < getindex(range,2))
        return value_new
    else
        return value
    end
end



###############################################################################
###############################################################################
### poisson process

"""
    poisson_process(rate, time_start, interval_record)

generates event times according to a homogeneous poisson process with constant
`rate` in the `interval_record` (Tuple of start and end that specify start <=
times < end). `time_start` can be specified in order to relax the process such
that the assumption of an event at time 0 is relaxed.
"""
function poisson_process(rate::Real, time_start::Real, interval_record::Tuple{Real,Real}, rng::AbstractRNG)
    @assert time_start <= first(interval_record)
    time_min = first(interval_record)
    time_max = last(interval_record)

    time = float(time_start)
    times = Float64[]
    while time < time_max
        time += randexp(rng)/rate
        if time_min <= time < time_max
            push!(times, time)
        end
    end

    return times
end

"""
    inhomogeneous_poisson_process(rate, time_start, interval_record [,weight=1])

generates event times according to an inhomogeneous poisson process with
time-dependent `rate` in the `interval_record` (Tuple of start and end that
specify start <= times < end). `time_start` can be specified in order to relax
the process such that the assumption of an event at time 0 is relaxed.

The time-dependent `rate` has to be of type AbstractHistogram, e.g., in the
range 0:timestep:time_week, and will be considered with periodic boundary
conditions, i.e., times that are outside of the times specified in rate are
projected back into the range.

The function can be augmented with a weight that increases or decreases the mean rate.
"""
#TODO: write generalization of this with a function argument, where the
#function itselfs calls the get_rate_pbc for our case. Not sure how to deal
#with rate_max in this case
function inhomogeneous_poisson_process(rate::AbstractHistogram, time_start::Real, interval_record::Tuple{Real,Real}, rng::AbstractRNG; weight=1.0)
    @assert time_start <= first(interval_record)
    time_min = first(interval_record)
    time_max = last(interval_record)
    rate_max = float(maximum(rate.weights))

    time = float(time_start)
    times = Float64[]
    while time < time_max
        # weight is only used in the dt step because then the rate_max and rate have the same scale
        time += randexp(rng)/rate_max/weight
        if time_min <= time < time_max
            if rand(rng) < get_pbc(rate,time)/rate_max
                push!(times, time)
            end
        end
    end

    return times
end

function get_pbc(rate::AbstractHistogram, time::Real)
    time_ref = rate.edges[1][1]
    time_dur = rate.edges[1][end] - time_ref
    # scalar arguments are copied by default and can be operated on
    while time < time_ref
        time += time_dur
    end
    idx = StatsBase.binindex(rate, time_ref + (time-time_ref)%time_dur)
    # per construction this is in bounds (we checkd above to make sure it is
    # larger than time_ref and just now that it is below duration)
    #if checkbounds(Bool, rate.weights, idx...)
    return @inbounds rate.weights[idx...]
end


###############################################################################
###############################################################################
### renewal process
# idea: generalize this with Poisson process where we wuold simple pass the fucntion of randexp(rng)/rate

function renewal_process(Pdt::UnivariateDistribution, time_start::Real, interval_record::Tuple{Real,Real}, rng::AbstractRNG)
    @assert time_start <= first(interval_record)
    time_min = first(interval_record)
    time_max = last(interval_record)

    time = float(time_start)
    times = Float64[]
    while time < time_max
        time += rand(rng, Pdt)
        if time_min <= time < time_max
            push!(times, time)
        end
    end

    return times
end

"""
    cyclic_renewal_process(prob, Pdt, time_start, interval_record)

generates event times according to a renewal process specified by probability
of next inter-encounter-interval `Pdt` but accepts new events with probability
to reproduce time-dependent `rate`.  Times are only written if in the
`interval_record` (Tuple of start and end that specify start <= times < end).
`time_start` can be specified in order to relax the process such that the
assumption of an event at time 0 is relaxed.

The time-dependent `rate` has to be of type AbstractHistogram, e.g., in the
range 0:timestep:time_week, and will be considered with periodic boundary
conditions, i.e., times that are outside of the times specified in rate are
projected back into the range.

In order to incorporate heterogeneous rates, one needs to rescale the
parameters of Pdt that determine the mean
"""
function cyclic_renewal_process(prob::AbstractHistogram, Pdt::UnivariateDistribution, time_start::Real, interval_record::Tuple{Real,Real}, rng::AbstractRNG)
    @assert time_start <= first(interval_record)
    @assert maximum(prob.weights) <= 1
    time_min = first(interval_record)
    time_max = last(interval_record)

    time = float(time_start)
    times = Float64[]
    while time < time_max
        time += rand(rng, Pdt)
        if time_min <= time < time_max
            if rand(rng) < get_pbc(prob,time)
                push!(times, time)
            end
        end
    end

    return times
end


function fit_cyclic_renewal_process()
end

###############################################################################
###############################################################################
### correlated Weibull renewal process

#TODO: Due to long times, start_time should be a random variable itself!
function correlated_weibull_renewal_process(params, args_weibull, time_start::Real, interval_record::Tuple{Real,Real}, rng::AbstractRNG)
    @assert time_start <= first(interval_record)
    time_min = first(interval_record)
    time_max = last(interval_record)

    A0, A1, tau1 = params
    norm = sqrt(A0^2 + A1^2)

    time = float(time_start)
    times = Float64[]
    x_biv = randn(rng)
    x_latent = ( A0 * randn(rng) +  A1 * x_biv ) / norm
    while time < time_max
        time += weibull_from_gauss(x_latent, scale = args_weibull.scale, shape = args_weibull.shape)
        if time_min <= time < time_max
            push!(times, time)
        end
        x_biv = bivariate_gauss(rng, exp(-1/tau1), x_biv)
        x_latent = ( A0 * randn(rng) + A1 * x_biv ) / norm
    end

    return times
end

# TODO: I cannot return Inf for x<0 because this does not broadcast ...
function logpdf_weibull(x, p)
    shape, scale = p
    z = x./ scale
    logpdf = log.(shape./scale) .+ (shape-1).*log.(z) .- z.^shape
    return logpdf
end
function pdf_weibull(x, p)
    shape, scale = p
    z = x ./ scale
    pdf = shape ./ scale .* z .^ (shape-1) .* exp.( - z .^ shape)
    return pdf
end

"""
    fit_Weibull(dist)

fit a Weilbull distribution to an empirical `dist`.

# Remark
This does not seem to be very stable. Initial values, especially of shape parameter, should be reasonable.
"""
function fit_Weibull(dist; p0=[0.1,1000])
    @assert dist.edges[1][1] >= 0
    shift =  dist.edges[1][2] - dist.edges[1][1]
    xbin, Pbin, dx = logbin(dist)
    # exclude data points where xbin is NaN because there were no entries in the original distribution for this range
    mask = isnan.(xbin).==0
    xbin=xbin[mask]
    Pbin=Pbin[mask]
    fit = curve_fit(logpdf_weibull, xbin .+ shift, log.(Pbin), p0, lower=[0.0,0.0], upper=[Inf, Inf])

    return NamedTuple{(:shape, :scale)}(coef(fit))
end

function fit_correlated_Weibull(dist, lags, acf_data, acf_data_err; weibull_correlated! = weibull_correlated_exponential!, p0=[0.5,0.5,100], dp=[0.1, 0.1, 10], seed=1000, num_updates=10000, beta=1) # acf_latent=acf_exponential,
    # some local functions
    uncorrelated_gauss(rng, num_elements) = rand(rng, Normal(0,1), num_elements)
    log_likelihood(acf_data, acf_model) = -sum( (acf_model .- acf_data).^2 )

    rng = MersenneTwister(seed)
    args_weibull = fit_Weibull(dist)
    list_keys = [keys(args_weibull)...]
    list_vals = [values(args_weibull)...]

    # Monte Carlo sampling of autocorrelation parameters to match the empirical autocorrelation function provided
    params = p0
    num_samples = last(lags)*200
    samples = zeros(num_samples)
    weibull_correlated!(samples, params, args_weibull, rng)
    acf_model = autocorrelation_function(samples, lags)
    logL = log_likelihood(acf_data, acf_model)

    best_fit = [params..., logL]
    best_acf = acf_model

    list_logL = Vector{Float64}(undef, num_updates)

    prog = Progress(num_updates, 1, "Fit correlated Weibull samples: ", offset=1)
    for i in 1:num_updates
        backup = parameter_update!(params, dp, rng)

        weibull_correlated!(samples, params, args_weibull, rng)
        acf_model = autocorrelation_function(samples, lags)

        # accept aparameters with probability
        new_logL = log_likelihood(acf_data, acf_model)
        if rand(rng) < exp(beta*(new_logL - logL))
            # accept (keep parameter)
            logL = new_logL
        else
            # reject (undo parameter)
            parameter_update_undo!(params, backup)
        end

        if logL > last(best_fit)
            best_fit = [params..., logL]
            best_acf = acf_model
        end
        list_logL[i] = logL
        next!(prog)
    end

    best_logL = last(best_fit)
    best_params = best_fit[1:end-1]

    #println(best_params)

    return list_logL, args_weibull, best_params, best_acf
end
function parameter_update!(params, dp, rng)
    i = rand(rng, 1:length(params))
    p_i_old = params[i]
    params[i] += dp[i] * (-1 + 2*rand(rng))
    if params[i] < 0
        params[i] = 0
    end
    return i, p_i_old
end
function parameter_update_undo!(params, backup)
    i, p_i_old = backup
    params[i] = p_i_old
end

function weibull_correlated_exponential!(samples::AbstractVector, params, args_weibull, rng)
    A0, A1, tau1 = params

    # generate latent samples in place (bivariat gauss starts with normal distributed random variable)
    # normalize to unit variance for transformation to weilbull. Since delta and bivariate uncorrelated, variances just add
    norm = sqrt(A0^2 + A1^2)
    x_ = randn(rng)
    samples[1] = ( A0 * randn(rng) +  A1 * x_ ) / norm
    for i in 2:length(samples)
        x_ = bivariate_gauss(rng, exp(-1/tau1), x_)
        samples[i] = ( A0 * randn(rng) + A1 * x_ ) / norm
    end

    # transform to weibull samples  in place
    weibull_from_gauss!(samples, scale = args_weibull.scale, shape = args_weibull.shape)

    return
end


"""
    bivariate_gauss(rng, rho, x_)

returns the next expement in a bivariate gauss series with correlation
coefficient `rho`

# Parameters
    * rng : random number generator
    * rho : correlation coefficient, 0 <= rho < 1, and rho = exp(-1/tau_exp)
    * x_ : previous element in series
"""
function bivariate_gauss(rng, rho, x_)
    return rho*x_ + sqrt(1-rho^2)*randn(rng)
end
function bivariate_gauss(rng)
    return randn(rng)
end

"""
    weibull_from_gauss(x; scale, shape)
    weibull_from_gauss!(x; scale, shape)

transform (correlated) Gaussian-distributed numbers into (correlated) Weibull-distributed numbers
"""
function weibull_from_gauss(x; scale, shape)
    return scale*( -log.( (1 .- erf.(x/sqrt(2) ) )/2 ) ) .^ (1/shape)
end
function weibull_from_gauss!(x; scale, shape)
    x .= scale*( -log.( (1 .- erf.(x/sqrt(2) ) )/2 ) ) .^ (1/shape)
    return
end
