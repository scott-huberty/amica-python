"""Tests for Fortran AMICA I/O utilities."""
from pathlib import Path

import mne
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
            "block_size": 512,
            "blk_min": 256,
            "blk_step": 256,
            "blk_max": 1024,
            "writestep": 20,
        },
    )
    content = param_fpath.read_text()
    want = (Path(__file__).parent / "assets" / "amicadefs_test.param").read_text()
    assert content == want


def test_io(tmp_path):
    """Test reading and writing binary data files for Fortran AMICA."""
    fpath = data_path() / "eeglab_sample_data" / "eeglab_data.set"
    raw = mne.io.read_raw_eeglab(fpath, preload=True)
    data = raw.get_data().T
    fpath = tmp_path / "data.bin"
    write_data(data, fpath)
    data_in = load_data(fpath, shape=data.T.shape).T
    assert_allclose(data.astype("<f4"), data_in)
