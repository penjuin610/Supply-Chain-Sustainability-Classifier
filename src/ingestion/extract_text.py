from __future__ import annotations

import argparse
import json
from pathlib import Path

from pypdf import PdfReader

from src.cleaning.basic_clean import basic_clean


def extract_pdf_text(pdf_path: Path) -> list[dict]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for index, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        pages.append(
            {
                "page_number": index,
                "raw_text": raw_text,
                "clean_text": basic_clean(raw_text),
                "extraction_method": "direct_text",
            }
        )
    return pages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-path", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    pages = extract_pdf_text(Path(args.pdf_path))
    Path(args.output_json).write_text(json.dumps(pages, indent=2), encoding="utf-8")
    print(f"saved extraction to {args.output_json}")


if __name__ == "__main__":
    main()

