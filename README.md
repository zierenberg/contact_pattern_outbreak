# resonance_contact_disease
Code acompanying our paper "Resonance between contact patterns and disease progression shapes epidemic spread"

# Data source:
Copenhagen:
https://figshare.com/articles/dataset/The_Copenhagen_Networks_Study_interaction_data/7267433/1?file=14000795

Sociopatterns:
http://www.sociopatterns.org/datasets/co-location-data-for-several-sociopatterns-data-sets/

# Installation
```
import Pkg;
Pkg.add.([
    "DataStructures",
    "DelimitedFiles",
    "Distributions",
    "HDF5",
    "IterTools",
    "LinearAlgebra",
    "LsqFit",
    "Printf",
    "ProgressMeter",
    "Random",
    "Roots",
    "SpecialFunctions",
    "Statistics",
    "StatsBase",
]);
```

# Prepare
```
cd cloned_directory
mkdir ./out/
mkdir ./out_mf/
~/bin/julia-1.6.2/bin/julia
```

# Running the analysis
```
include("analysis/data_analysis.jl")

# set this to `true` to skip error estimates, as they take most of the time.
skip_jackknife = false

# main analysis
# reduce level of details to be faster but skip some analysis
analyse_all(Copenhagen(), path_out = "./out/", level_of_details=3)

#  filter out participants that had no rssi signal on both first and last day of study
analyse_all(Copenhagen(), path_out = "./out/", level_of_details=3,
    filter_out_incomplete=false)

# for InVS15
analyse_all(InVS15(), path_out = "./out/", level_of_details=3)
```


# Epidemic spread in mean-field model
```
include("analysis/data_analysis.jl")

# create the data from mean field model
sample_mean_field_for_effective_R(path_out = "./out_mf")

# analyse from measurements
analyse_effective_R(
    "./out_mf/mean_field_samples_Copenhagen_filtered_15min.h5",
    path_out = "./out_mf",
    dsetname = "measurements")

# analyse from surrogates
analyse_effective_R(
    "./out_mf/mean_field_samples_Copenhagen_filtered_15min.h5",
    path_out = "./out_mf",
    dsetname = "measurements_randomized_per_train")

```


# Plotting

```
h5.recursive_load("/Users/paul/mpi/simulation/disease_spread_contact_structure/_latest/out_lvl3/results_Copenhagen_filtered_15min.h5", dtype=ph.bdict, keepdim=True)
```
