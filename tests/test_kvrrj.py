from pathlib import Path


def test_kvrrj():
    from kvrrj import DATA_DIRECTORY, HOME_DIRECTORY

    assert isinstance(DATA_DIRECTORY, Path)
    assert isinstance(HOME_DIRECTORY, Path)
    assert DATA_DIRECTORY.exists()
    assert HOME_DIRECTORY.exists()
