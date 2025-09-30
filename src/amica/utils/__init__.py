from .fetch import fetch_datasets, fetch_test_data, fetch_fortran_outputs
from .fortran import (
    load_data,
    load_initial_weights,  # rename to load_weights
    load_fortran_results,
    write_data,
    write_param_file,
)
from .simulation import generate_toy_data
