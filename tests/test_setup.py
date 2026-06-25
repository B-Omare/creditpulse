def test_creditpulse_importable():
    """Confirm the package structure is correct."""
    import creditpulse
    assert creditpulse is not None

def test_data_folders_exist():
    """Confirm all expected data folders exist."""
    import os
    assert os.path.isdir("data/raw")
    assert os.path.isdir("data/processed")
    assert os.path.isdir("data/simulated")