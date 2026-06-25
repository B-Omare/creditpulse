def test_creditpulse_importable():
    """Confirm the package structure is correct."""
    import creditpulse
    assert creditpulse is not None

def test_data_folders_exist():
    """Confirm all expected data folders exist, creating them if needed."""
    import os
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/simulated", exist_ok=True)
    assert os.path.isdir("data/raw")
    assert os.path.isdir("data/processed")
    assert os.path.isdir("data/simulated")