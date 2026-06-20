from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests


def filename_from_url(url: str) -> str:
    path = urlparse(url).path
    name = Path(path).name
    return name or "downloaded_file.pdf"


def download_pdf(url: str, output_dir: Path, timeout: int = 30) -> Path:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename_from_url(url)
    output_path.write_bytes(response.content)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True, help="CSV containing a url column")
    parser.add_argument("--url-column", default="url")
    parser.add_argument("--output-dir", default="data/raw_pdfs")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    output_dir = Path(args.output_dir)

    for url in df[args.url_column].dropna():
        path = download_pdf(url, output_dir)
        print(f"downloaded: {path}")


if __name__ == "__main__":
    main()

