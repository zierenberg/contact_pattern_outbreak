using Pkg;
path_project=join(split(@__DIR__, "/")[1:end-1],"/")
println(path_project)
Pkg.activate(path_project)
Pkg.instantiate()


include("data_analysis.jl")
include("data_driven_models.jl")

"""
skip_jackknife=true
default_range_latent     = 0:0.5:8
default_range_infectious = 0.5:0.5:8
"""

function reproduce_paper()
    println("reproduce content of paper; this will take a while")

    # full data analysis
    analyse_all()

    # tailored surrogate data
    sample_surrogate_tailored()

    # specific surrogate data
    sample_surrogate_inhomogeneous_poisson_weighted()
    sample_surrogate_inhomogeneous_poisson()
    sample_surrogate_homogeneous_poisson_weighted()
    sample_surrogate_homogeneous_poisson()
    sample_surrogate_weibull_weighted()
    sample_surrogate_weibull()

    # data-driven branching processes
    analytic_survival_probability()
    sample_continuous_time_branching()

    # controls
    analyse_all(filter_out_incomplete=true, level_of_details=1)
end

function add_on(;
    )
    experiment::ContactData = Copenhagen();
    minimum_duration = 15*60;
    path_dat = "./dat";
    path_out = "./out";
    filter_out_incomplete=false;
    seed=1000;
    _, ets, _ = load_processed_data(experiment, minimum_duration, path_dat, filter_out_incomplete=filter_out_incomplete);    range_R0 = default_range_R0
    mkpath(path_out)
    filename_data = @sprintf("%s/data_%s_filtered_%dmin.h5",
        path_out, label(experiment), minimum_duration/60)
    filename_rand = @sprintf("%s/data_randomized_per_train_%s_filtered_%dmin.h5",
        path_out, label(experiment), minimum_duration/60)
    filename_rand_all = @sprintf("%s/data_randomized_all_%s_filtered_%dmin.h5",
        path_out, label(experiment), minimum_duration/60)

    range_latent = 0:0.5:8
    range_infectious = 0.5:0.5:8
    range_R0 = 1:1.0:5

    # data
    analyse_dispersion_scan_delta(range_latent,
                                  range_infectious,
                                  range_R0,
                                  ets,
                                  filename_data,
                                  "/disease/delta/"
                                 )
    # randomized
    analyse_dispersion_scan_delta(range_latent,
                                  range_infectious,
                                  range_R0,
                                  surrogate_randomize_per_train(ets, seed),
                                  filename_rand,
                                  "/disease/delta/"
                                 )

    # poisson limit
    analyse_dispersion_scan_delta(range_latent,
                                  range_infectious,
                                  range_R0,
                                  surrogate_randomize_all(ets, seed),
                                  filename_rand_all,
                                  "/disease/delta/"
                                 )
end

function add_on_surival_probability(;
)
    experiment::ContactData = Copenhagen()
    minimum_duration = 15 * 60
    path_dat = "./dat"
    path_out = "./out"
    filter_out_incomplete = false
    seed = 1000
    _, ets, _ = load_processed_data(experiment, minimum_duration, path_dat, filter_out_incomplete=filter_out_incomplete)
    range_R0 = default_range_R0
    mkpath(path_out)
    filename_data = @sprintf("%s/data_%s_filtered_%dmin.h5",
        path_out, label(experiment), minimum_duration / 60)
    filename_rand = @sprintf("%s/data_randomized_per_train_%s_filtered_%dmin.h5",
        path_out, label(experiment), minimum_duration / 60)
    filename_rand_all = @sprintf("%s/data_randomized_all_%s_filtered_%dmin.h5",
        path_out, label(experiment), minimum_duration / 60)

    range_latent = 0:0.5:8
    range_infectious = 0.5:0.5:8
    range_R0 = 1:1.0:5

    # data
    analyse_survival_scan_delta(range_latent,
        range_infectious,
        range_R0,
        ets,
        filename_data,
        "/disease/delta/"
    )
    # randomized
    analyse_survival_scan_delta(range_latent,
        range_infectious,
        range_R0,
        surrogate_randomize_per_train(ets, seed),
        filename_rand,
        "/disease/delta/"
    )

    # poisson limit
    analyse_survival_scan_delta(range_latent,
        range_infectious,
        range_R0,
        surrogate_randomize_all(ets, seed),
        filename_rand_all,
        "/disease/delta/"
    )
end
