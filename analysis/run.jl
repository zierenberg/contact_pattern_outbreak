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
end
