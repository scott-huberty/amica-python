import mne
from numpy.testing import assert_allclose

from amica import AMICA


def test_to_mne(sklearn_example_data):
    """Test migrating AMICA outputs to an MNE-Python ICA instance."""
    X = sklearn_example_data
    transformer = AMICA(random_state=0)
    transformer.fit(X)
    info = mne.create_info(
        ch_names=["EEG_1", "EEG_2", "EEG_3"], sfreq=500, ch_types="eeg"
        )
    ica = transformer.to_mne(info=info)
    raw = mne.io.RawArray(X.T, info=info)

    # both should produce the same IC activations.
    X_sources = transformer.transform(X)
    mne_sources = ica.get_sources(raw).get_data().T
    assert_allclose(mne_sources, X_sources)

    # ica.apply zero's out excluded components and does an inverse_transform.
    # so without any comps to exclude, it should simply be the inverse_transform
    raw_reconstructed = ica.apply(raw.copy())
    X_reconstructed = transformer.inverse_transform(X_sources)
    assert_allclose(raw_reconstructed.get_data().T, X_reconstructed)
