"""TDMS -> CSV 변환 스크립트.

사용법:
    python tdms_to_csv.py <file.tdms>                # 같은 폴더에 <file>.csv 생성
    python tdms_to_csv.py <file.tdms> -o out_dir     # out_dir 안에 저장
    python tdms_to_csv.py <file.tdms> --per-group    # 그룹마다 별도 CSV 생성
    python tdms_to_csv.py <file.tdms> --inspect      # 구조만 출력하고 종료

요구사항: pip install npTDMS pandas
"""

import argparse
import sys
from pathlib import Path

from nptdms import TdmsFile


def inspect(tdms: TdmsFile) -> None:
    for group in tdms.groups():
        print(f"[Group] {group.name}")
        for ch in group.channels():
            try:
                length = len(ch)
            except Exception:
                length = "?"
            props = ", ".join(f"{k}={v}" for k, v in ch.properties.items())
            print(f"  - {ch.name} (len={length}) {{{props}}}")


def save_single_csv(tdms: TdmsFile, out_path: Path) -> Path:
    df = tdms.as_dataframe()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def save_per_group_csv(tdms: TdmsFile, out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for group in tdms.groups():
        df = group.as_dataframe()
        if df.empty:
            continue
        safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in group.name)
        out_path = out_dir / f"{stem}__{safe_name}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        written.append(out_path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a TDMS file to CSV")
    parser.add_argument("tdms_path", type=Path, help="Path to .tdms file")
    parser.add_argument("-o", "--out-dir", type=Path, default=None,
                        help="Output directory (default: same as input)")
    parser.add_argument("--per-group", action="store_true",
                        help="Write one CSV per TDMS group instead of one big CSV")
    parser.add_argument("--inspect", action="store_true",
                        help="Only print TDMS structure, do not write CSV")
    args = parser.parse_args()

    tdms_path: Path = args.tdms_path
    if not tdms_path.exists():
        print(f"File not found: {tdms_path}", file=sys.stderr)
        return 1

    print(f"Reading {tdms_path} ...")
    tdms = TdmsFile.read(str(tdms_path))

    if args.inspect:
        inspect(tdms)
        return 0

    out_dir = args.out_dir or tdms_path.parent
    stem = tdms_path.stem

    if args.per_group:
        written = save_per_group_csv(tdms, out_dir, stem)
        if not written:
            print("No data written (no non-empty groups).")
            return 1
        for p in written:
            print(f"Wrote {p}")
    else:
        out_path = out_dir / f"{stem}.csv"
        save_single_csv(tdms, out_path)
        print(f"Wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
