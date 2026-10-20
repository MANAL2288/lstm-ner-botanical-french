"""Download FloraNER (Zenodo record 12765869, CC-BY-4.0) and build the BIO splits.

Usage: python src/download_data.py
"""
import urllib.request
import zipfile

from data import RAW_DIR, TASKS, build_processed

RECORD = "https://zenodo.org/api/records/12765869/files/{}/content"
FILES = ["Train.zip", "Test.zip"]


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = RAW_DIR / name
        if not target.exists():
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(RECORD.format(name), target)
        with zipfile.ZipFile(target) as z:
            z.extractall(RAW_DIR)
    for task in TASKS:
        report = build_processed(task)
        print(task, {split: (r["segments"], r["entities"]) for split, r in report.items()})


if __name__ == "__main__":
    main()
