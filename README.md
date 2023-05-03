# contact_pattern_outbreak
Code acompanying our paper "How contact patterns destabilize and modulate epidemic outbreaks"

# Data source:
Copenhagen:
https://figshare.com/articles/dataset/The_Copenhagen_Networks_Study_interaction_data/7267433/1?file=14000795

Sociopatterns:
http://www.sociopatterns.org/datasets/co-location-data-for-several-sociopatterns-data-sets/

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
```





# Running the analysis
Make sure you installed [julia](https://julialang.org/downloads/).

Start a julia REPL in the project folder

```bash
cd /path_to_cloned_directory
```

```julia
# Including run.jl will install required packages and provides easy acesse to functions to reproduce content of paper.
include("analysis/run.jl")

# To reproduce complete content for main paper run
reproduce_paper()

# For a more specific reproduction of individual content, follow the steps in reproduce_paper().
# We here provide an example for the data analysis (takes around ~6h)

# Set this to `true` to skip error estimates, as they take most of the time:
skip_jackknife = false

# You can reduce level of details to be faster but skip some analysis.
analyse_all(Copenhagen(), path_out = "./out/", level_of_details=3)

# You can filter out participants that had no rssi signal on both first and last day of study
analyse_all(Copenhagen(), path_out = "./out/", level_of_details=3,
    filter_out_incomplete=false)

# Repeat analysis for another dataset (e.g. InVS15)
analyse_all(InVS15(), path_out = "./out/", level_of_details=3)
```

# Plotting

Plotting is implemented in python.
It assumes that analysed files are placed in './out/'
Install required packages, new conda enviornment recommended. Some smaller packages are only available via pip

```bash
conda install numpy scipy matplotlib seaborn h5py tqdm
pip install python-benedict addict palettable
pip install git+https://github.com/pSpitzner/bitsandbobs
```

Start an interactive python shell with our `plot_helper`

```bash
  cd resonance_contact_disease
  python -i ./plotting/plot_helper.py
  # or if you prefer ipython
  ipython -i ./plotting/plot_helper.py
```

We have some global settings that affect all panels.

```python
# select things to draw for every panel
show_title = False
show_xlabel = False
show_ylabel = False
show_legend = False
show_legend_in_extra_panel = False
use_compact_size = True  # this recreates the small panel size of the manuscript
figures_only_to_disk = True
debug = False  # set to True to stop when a plot fails
figure_path = "./figs/njp"
data_input_path = "./out"
```

Note that, since we arranged panels in postprocessing we did not generate the labels and titles in matplotlib. Thus, there may be clipping of the automatically genereated axis labels and titles. They are still part of the pdf, just not in the viewer.

Load the main hdf5 file from the analysis. Then, figures can be created as shown below. They should open automatically, else try `plt.show()` to show them manually or `plt.ion()` to set matplotlib to interactive mode.


```python
main_manuscript()
# or
figure_1()
figure_2()
figure_3()
figure_4()
```

To recreate SM figures see the `figure_sm_` functions in `/plotting/plot_helper.py`.

