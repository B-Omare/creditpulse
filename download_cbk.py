import urllib.request
import os

os.makedirs("data/cbk_reports", exist_ok=True)

reports = [
    (
        "https://www.centralbank.go.ke/uploads/financial_sector_stability/1857203631_Financial%20Stability%20Report%202023.pdf",
        "data/cbk_reports/fsr_2023.pdf"
    ),
    (
        "https://www.centralbank.go.ke/uploads/bank_supervision_reports/1506502325_Annual%20Report%202022.pdf",
        "data/cbk_reports/cbk_annual_2022.pdf"
    ),
]

for url, path in reports:
    print(f"Downloading {path}...")
    try:
        urllib.request.urlretrieve(url, path)
        size = os.path.getsize(path) / 1024 / 1024
        print(f"  Done — {size:.1f} MB")
    except Exception as e:
        print(f"  Failed: {e}")