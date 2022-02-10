# resonance_contact_disease
Code acompanying our paper "Resonance between contact patterns and disease progression shapes epidemic spread"

# Data source:
Copenhagen:
https://figshare.com/articles/dataset/The_Copenhagen_Networks_Study_interaction_data/7267433/1?file=14000795

Sociopatterns:
http://www.sociopatterns.org/datasets/co-location-data-for-several-sociopatterns-data-sets/

# Installation

Install required packages in a julia shell

```julia
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
Go to your directory

```bash
cd cloned_directory
```

Download the physical proximity data from the Copenhagen Networks Study

```bash
mkdir ./dat/
wget https://figshare.com/ndownloader/files/14000795 -O ./dat/bt_symmetric.csv
```

Create folders for output

```bash
mkdir ./out/
mkdir ./out_mf/
```

Start julia
```bash
~/bin/julia-1.6.2/bin/julia
```

# Running the analysis
```julia
include("analysis/data_analysis.jl")

# set this to `true` to skip error estimates, as they take most of the time.
skip_jackknife = false

# main analysis
# takes around ~6h
# reduce level of details to be faster but skip some analysis
analyse_all(Copenhagen(), path_out = "./out/", level_of_details=3)

#  filter out participants that had no rssi signal on both first and last day of study
analyse_all(Copenhagen(), path_out = "./out/", level_of_details=3,
    filter_out_incomplete=false)

# for InVS15
analyse_all(InVS15(), path_out = "./out/", level_of_details=3)
```


# Epidemic spread in mean-field model
```julia
include("analysis/data_analysis.jl")

# create the data from mean field model
# takes around ~18h
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

Plotting is implemented in python.
Install required packages, new conda enviornment recommended. Some smaller packages are only available via pip

```bash
conda install numpy scipy matplotlib seaborn h5py tqdm
pip install python-benedict addict palettable
```

Start an interactive python shell with our `plot_helper`

```bash
  cd resonance_contact_disease
  python -i ./plotting/plot_helper.py
  # or if you prefer ipython
  ipython -i ./plotting/plot_helper.py
```

We have some global settings that affect all panels. Beware, when setting `use_compact_size = True` this may clip axis labels and ticks. They are still part of the pdf, just not in the viewer.

```python
show_title = True
show_xlabel = True
show_ylabel = True
show_legend = True
show_legend_in_extra_panel = False
use_compact_size = False  # this recreates the small panel size of the manuscript
```

Load the main hdf5 file from the analysis. Then, figures can be created as shown below. They should open automatically, else try `plt.show()` to show them manually or `plt.ion()` to set matplotlib to interactive mode. We provide a helper to save all currently open figures into a folder.

```python
h5f = h5.recursive_load("./out/results_Copenhagen_filtered_15min.h5", dtype=bdict, keepdim=True)

figure_1(h5f)
figure_2(h5f)
figure_3(h5f)
figure_4(h5f)
figure_5(h5f)  # this guy needs the extra files in './out_mf/'

save_all_figures("./figs/", fmt="pdf", dpi=300)
```

Supplementary Figures may need different files to be loaded. Load the `h5f` as needed.

```python
figure_sm_external(h5f)
figure_sm_overview(h5f)
figure_sm_dispersion(h5f)
```

When done, it is recommended to close the hdf5 files to release the file lock (by default we do not load the full content into ram, but keep files open).

```python
h5.close_hot()
```

