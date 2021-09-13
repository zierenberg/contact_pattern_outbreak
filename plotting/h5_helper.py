# ------------------------------------------------------------------------------ #
# @Author:        F. Paul Spitzner
# @Email:         paul.spitzner@ds.mpg.de
# @Created:       2020-07-21 11:11:40
# @Last Modified: 2021-09-13 22:13:41
# ------------------------------------------------------------------------------ #
# Helper functions to work conveniently with hdf5 files
#
# Example
# ```
# import h5_helper as h5
# fpath = '~/demo/file.hdf5'
# h5.ls(fpath)
# h5.recursive_ls(fpath)
# h5.load(fpath, '/some/dataset/')
# h5.recursive_load(fpath, hot=False)
# ```
# ------------------------------------------------------------------------------ #

import os
import sys
import glob
import numbers
import h5py
import numpy as np

import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s [%(name)s] %(message)s")
log = logging.getLogger(__name__)

_benedict_is_installed = False
_addict_is_installed = False
try:
    from benedict import benedict
    _benedict_is_installed = True
except ImportError:
    log.debug("benedict is not installed")

try:
    from addict import Dict
    _addict_is_installed = True
except ImportError:
    log.debug("addict is not installed")


def recursive_load(filename, dsetname="/", skip=None, hot=False, keepdim=False, dtype = None):
    """
        Load a hdf5 file as a nested dict.
        enhenced dicts via benedict or addict are supported via `type` argument

        # Paramters:
        filename : string
            path to hdf5 file
        dsetname : string
            not implemented yet. eventually should allow to only load parts of files.
        skip : list
            names of dsets to exclude
        hot : bool
            if True, does not load dsets to ram, but only links to the hdf5 file. this keeps the file open, call `close_hot()` when done!
            Use this if a dataset in your file is ... big
        keepdim : set to true to preserve original data-set shape for 1d arrays
            instead of casting to numbers
        dtype : str, dtype or None
            which dictionary class to use. Default, None uses a normal dict,
            "benedict" or "addict" use those types,
            if a type is passed, it is assumed to be a subclass of a normal dict and will be called as the constructor
    """

    if dtype is None:
        dtype = dict
    elif isinstance(dtype, str):
        if dtype.lower() == "benedict":
            assert _benedict_is_installed, "try `pip install python-benedict`"
            dtype = benedict
        elif dtype.lower() == "addict":
            assert _addict_is_installed, "try `pip install addict`"
            dtype = Dict
        else:
            raise ValueError("unsupported value passed for `dtype`")
    else:
        # assert isinstance(dtype, type), "unsupported value passed for `dtype`"
        assert isinstance(dtype(), dict), "unsupported value passed for `dtype`"

    if skip is not None:
        assert isinstance(skip, list)
    else:
        skip = []

    if dsetname != "/":
        raise NotImplementedError

    assert isinstance(hot, bool)
    assert isinstance(keepdim, bool)
    assert isinstance(dsetname, str)
    assert isinstance(filename, str)

    candidates = recursive_ls(filename, dsetname)

    cd_len = []
    # res = BetterDict()
    # res._set_h5_filename(filename)
    res = dtype()
    res["h5"] = dtype()
    res["h5"]["filename"] = filename
    if hot:
        f = h5py.File(filename, "r")
        # res._set_h5_file(f)
        res["h5"]["file"] = f
        remember_file_is_hot(f)

    maxdepth = 0
    for cd in candidates:
        l = len(cd.split("/"))
        cd_len.append(l)
        if l > maxdepth:
            maxdepth = l

    # iterate by depth, creating hierarchy
    for ddx in range(1, maxdepth + 1):
        for ldx, l in enumerate(cd_len):
            if l == ddx:
                cd = candidates[ldx]
                components = cd.split("/")
                if len([x for x in skip if x in components]) > 0:
                    continue
                temp = res
                if ddx > 1:
                    for cp in components[0:-1]:
                        temp = temp[cp]
                cp = components[-1]
                if len(ls(filename, cd)) > 0:
                    # temp[cp] = BetterDict()
                    temp[cp] = dtype()
                else:
                    if hot:
                        temp[cp] = load_hot(filename, cd, keepdim)
                    else:
                        temp[cp] = load(filename, cd, keepdim)

    return res


def _recursive_tree(d, t=None, depth=0):
    # helper to call recursively and build the tree structure of the content
    if t is None:
        t = {key: [] for key in ["pcs", "scs", "varname", "varval", "vartype"]}
        t["prev_pc"] = ""

    for vdx, var in enumerate(d.keys()):
        if vdx < len(d.keys()) - 1:
            sc = "├── "
            pc = "│   "
        else:
            sc = "└── "
            pc = "    "

        t["pcs"].append(t["prev_pc"])
        t["scs"].append(sc)
        t["varname"].append(var)

        if isinstance(d[var], dict):
            # get all content of the dict, while incrementing the padding `pc`
            t["vartype"].append("")
            t["varval"].append("")
            t["prev_pc"] += pc
            _recursive_tree(d[var], t, depth + 1)
            t["prev_pc"] = t["prev_pc"][0:-4]
        else:
            # extract type and certain variables
            t["vartype"].append(f"{d[var].__class__.__name__}")
            # number
            if isinstance(d[var], numbers.Number):
                t["varval"].append(str(d[var]))
            # numpy byte strings
            elif isinstance(d[var], np.bytes_):
                string = d[var].decode("UTF-8").replace("\n", " ")
                if len(string) > 14:
                    string = f"{string:.11s}..."
                t["varval"].append(string)
            # base strings
            elif isinstance(d[var], str):
                string = d[var].replace("\n", " ")
                if len(string) > 14:
                    string = f"{string:.11s}..."
                t["varval"].append(string)
            # numpy arrays, print shape
            elif isinstance(d[var], np.ndarray):
                t["varval"].append(f"{d[var].shape}")
            # list, print length
            elif isinstance(d[var], list):
                t["varval"].append(f"({len(d[var])})")
            # hdf5 datset
            elif isinstance(d[var], h5py.Dataset):
                try:
                    # this will throw an exception if file is closed
                    d[var].file
                    t["varval"].append(f"{d[var].shape}")
                except ValueError:
                    t["varval"].append(f"(file closed)")
            # unknown
            else:
                t["varval"].append("")

    return t

def view(d, width=82, fill="."):
    """
    ```
    └── profile
          ├── firstname ................. str  Fabio
          └── lastname .................. str  Caccamo
    ```
    """
    res = ""
    p = _recursive_tree(d)
    for l in range(0, len(p["varname"])):
        left = f"{p['pcs'][l]}{p['scs'][l]}{p['varname'][l]}"
        right = f"{p['vartype'][l]}"
        ws = " " if p["vartype"][l] == "" else fill
        res += f"{left} {ws*(width-len(left)-len(right))} {right}"
        res += f"  {p['varval'][l]}\n" if len(p["varval"][l]) > 0 else "\n"

    print(res)


def load(filenames, dsetname, keepdim=False, raise_ex=False, silent=False):
    """
        load a h5 dset into an array. opens the h5 file and closes it
        after reading.

        # Parameters
        filenames: str path to h5file(s).
                   if wildcard given, result from globed files is returned
        dsetname:  str, which dset to read
        keepdim:   bool, per default arrays with 1 element will be mapped to scalars
                   set this to `True` to keep them as arrays
        raise_ex: whether to raise exceptions. default false,
                  in this case, np.nan is returned if loading fails
        silent:   if set to true, exceptions will not be reported

        # Returns
        res: ndarray or scalar, depending on loaded datatype
    """

    def local_load(filename):
        try:
            file = h5py.File(filename, "r")
            res = file[dsetname]
            # map 1 element arrays to scalars
            if res.shape == (1,) and not keepdim:
                res = res[0]
            elif res.shape == ():
                res = res[()]
            else:
                res = res[:]
            file.close()
            return res
        except Exception as e:
            if not silent:
                log.error(f"failed to load {dsetname} from {filename}: {e}")
            if raise_ex:
                raise e
            else:
                return np.nan

    files = glob.glob(os.path.expanduser(filenames))
    res = []
    for f in files:
        res.append(local_load(f))

    if len(files) == 1:
        return res[0]
    else:
        return res


def ls(filename, dsetname="/"):
    """
        list the keys in a dsetname

        Parameters
        ----------
        filename: path to h5file
        dsetname: which dset to list

        Returns
        -------
        list: containing the contained keys as strings
    """
    try:
        file = h5py.File(os.path.expanduser(filename), "r")
        try:
            res = list(file[dsetname].keys())
        except Exception as e:
            res = []
        file.close()
    except Exception as e:
        res = []

    return res


_h5_files_currently_open = dict(files=[], filenames=[])


def load_hot(filename, dsetname, keepdim=False):
    """
        sometimes we do not want to hold the whole dataset in RAM, because it is too
        large. Remember to close the file after processing!

        hmmm, two lists where indices have to match seem a bit fragile
    """
    global _h5_files_currently_open
    filename = os.path.expanduser(filename)
    if filename not in _h5_files_currently_open["filenames"]:
        file = h5py.File(filename, "r")
        _h5_files_currently_open["files"].append(file)
        _h5_files_currently_open["filenames"].append(filename)
    else:
        idx = _h5_files_currently_open["filenames"].index(filename)
        file = _h5_files_currently_open["files"][idx]

    try:
        # if its a xsingle value, load it even though this is 'hot'
        if file[dsetname].shape == (1,) and not keepdim:
            return file[dsetname][0]
        elif file[dsetname].shape == ():
            return file[dsetname][()]
        else:
            return file[dsetname]
    except Exception as e:
        log.error(f"Failed to load hot {filename} {dsetname}")
        raise e

def close_hot(which="all"):
    """
        hot files require a bit of care:
        * If a dict is opened from a hot hdf5 file, and `all` hot files are closed, the datasets are no longer accessible.
        * from the outside, currently it is hard to check whether an element of a dict is loaded
    """
    global _h5_files_currently_open
    # everything we opened
    if which == "all":
        for file in _h5_files_currently_open["files"]:
            try:
                file.close()
            except:
                log.debug("File already closed")
        _h5_files_currently_open["files"] = []
        _h5_files_currently_open["filenames"] = []
    # by index
    elif isinstance(which, int):
        del _h5_files_currently_open["files"][which]
        del _h5_files_currently_open["filenames"][which]
        try:
            _h5_files_currently_open["files"][which].close()
        except:
            log.debug("File already closed")
    # by passed hdf5 file handle
    elif isinstance(which, h5py.File):
        _h5_files_currently_open["files"].remove(which)
        _h5_files_currently_open["filenames"].remove(which.filename)
        try:
            which.close()
        except:
            log.debug("File already closed")

def remember_file_is_hot(file):
    # helper to keep a collection of open files
    global _h5_files_currently_open
    _h5_files_currently_open["files"].append(file)
    _h5_files_currently_open["filenames"].append(file.filename)


def recursive_ls(filename, dsetname=""):
    if dsetname == "":
        dsetname = "/"

    candidates = ls(filename, dsetname)
    res = candidates.copy()
    for c in candidates:
        temp = recursive_ls(filename, dsetname + f"{c}/")
        if len(temp) > 0:
            temp = [f"{c}/{el}" for el in temp]
            res += temp
    return res


