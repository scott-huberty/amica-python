"""Tests for Fortran AMICA I/O utilities."""
from pathlib import Path

import mne
import numpy as np
from numpy.testing import assert_allclose

from amica.datasets import data_path
from amica.utils.fortran import (
    load_data,
    write_data,
    write_param_file,
)


def test_write_param_file(tmp_path):
    """Test writing a paraam file for use with the Fortran AMICA Program."""
    fpath = data_path() / "eeglab_sample_data" / "eeglab_data.set"
    raw = mne.io.read_raw_eeglab(fpath, preload=True)
    data = raw.get_data().T

    param_fpath, _ = write_param_file(
        tmp_path / "foo.param",
        files="./tests/eeglab_sample_data/eeglab_data.fdt",
        outdir="./tests/eeglab_sample_data/amicaout_test/",
        data=data,
        **{
            "writestep": 20,
        },
    )
    content = param_fpath.read_text()
    want = (Path(__file__).parent / "assets" / "amicadefs_test.param").read_text()
    assert content == want


def test_write_param_file_formats_small_floats_for_fortran_fixed_reads(tmp_path):
    """Small floats must include a decimal mantissa for Fortran fixed reads."""
    data = np.array([[0.0, 1.0], [2.0, 3.0]])

    param_fpath, _ = write_param_file(
        tmp_path / "small_float.param",
        files="data.fdt",
        outdir="out/",
        data=data,
        min_dll=1e-7,
        min_grad_norm=1e-7,
        minlrate=1e-8,
        mineig=1e-15,
        invsigmin=1e-8,
    )

    params = dict(
        line.split(maxsplit=1)
        for line in param_fpath.read_text().splitlines()
    )

    for key in ("min_dll", "min_grad_norm", "minlrate", "mineig", "invsigmin"):
        assert "." in params[key]
        assert "e" in params[key].lower()
        assert float(params[key]) > 0.0


def test_io(tmp_path):
    """Test reading and writing binary data files for Fortran AMICA."""
    fpath = data_path() / "eeglab_sample_data" / "eeglab_data.set"
    raw = mne.io.read_raw_eeglab(fpath, preload=True)
    data = raw.get_data().T
    fpath = tmp_path / "data.bin"
    write_data(data, fpath)
    data_in = load_data(fpath, shape=data.T.shape).T
    assert_allclose(data.astype("<f4"), data_in)
