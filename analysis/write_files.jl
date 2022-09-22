using DelimitedFiles
using HDF5
using Printf
using StatsBase

include("load_data.jl")


"""
set (overwrite) a description as an attribute for a hdf5 object
"""
function myh5desc(filename, name, description::String)
    name = replace(name, r"\/+" => "/")
    filename = replace(filename, r"\/+" => "/")
    fid = h5open(filename, "cw", swmr=true)
    obj = fid[name]
    if haskey(attributes(obj), "description")
        delete_attribute(obj, "description")
    end
    attributes(obj)["description"] = description
    close(obj)
    close(fid)
end

"""
wrapper for h5write to replace dataset if existing (replace currently does not work)
"""
function myh5write(filename, datasetname, data::AbstractArray)
    datasetname = replace(datasetname, r"\/+" => "/")
    filename = replace(filename, r"\/+" => "/")

    h5open(filename, "cw", swmr=true) do fid
        if haskey(fid, datasetname)
            delete_object(fid, datasetname)
        end
        fid[datasetname, compress=4] = data
    end
end
myh5write(filename, datasetname, data::Number) = myh5write(filename, datasetname, [data,])

"""
wrapper for h5write to directly deal with a StatsBase distribution
"""
function myh5write(filename, datasetname, dist::AbstractHistogram)
    myh5write(filename, datasetname, hcat(collect(dist.edges[1][1:end-1]), dist.weights))
end

"""
wrapper for h5write to directly deal with contact train object
"""
function myh5write(filename, groupname::AbstractString, ets::encounter_trains{T}) where T
    myh5write(filename, @sprintf("%s/ids", groupname), ids(ets))
    myh5desc(filename, groupname,
        "Encoutner trains, where `ids` are labels for agents. Agents are numbered, and each agent has a train: `train_agentnumber` with the times of contacts in seconds."
    )

    for (i, train) in enumerate(trains(ets))
        myh5write(filename, @sprintf("%s/train_%d", groupname, i), train)
    end
end
