

def test_fetch_photos():
    from amica.utils.fetch import fetch_photos

    photo_fpath = fetch_photos()
    fpaths = list(photo_fpath.glob("*"))
    assert len(fpaths) == 5