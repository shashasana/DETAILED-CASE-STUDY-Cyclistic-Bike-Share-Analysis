# Cyclistic Bike Analysis — Repository Scaffold

This document contains a ready-to-copy repository layout and the core code files to run the Cyclistic analysis described in your project brief. Drop these files into a GitHub repo and run the instructions in **README.md**.

---

## Repo structure

```
cyclistic-bike-analysis/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                  # put raw monthly CSVs here (not committed)
├── notebooks/             # optional: exploratory notebooks
│   └── 01-exploration.ipynb
├── src/
│   ├── __init__.py
│   ├── ingest.py         # combine CSVs, detect file types
│   ├── clean.py          # cleaning functions
│   ├── analyze.py        # aggregations & metrics
│   ├── visualize.py      # plotting helpers + PDF report export
│   └── utils.py          # shared helpers
├── outputs/
│   ├── cleaned.csv
│   ├── analysis_summary.csv
│   └── figures/
└── .github/workflows/
    └── python-app.yml
```

---

## README.md

````markdown
# Cyclistic Bike Analysis

Repository to reproduce the Cyclistic marketing analysis (members vs casual riders).

## Quick start

1. Create a Python venv and install dependencies:

```bash
python -m venv venv
source venv/bin/activate   # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
````

2. Place the 12 monthly CSV files into `data/`.

3. Run the pipeline:

```bash
python -m src.ingest --data-dir data --out outputs/combined_raw.csv
python -m src.clean --in outputs/combined_raw.csv --out outputs/cleaned.csv
python -m src.analyze --in outputs/cleaned.csv --out outputs/analysis_summary.csv
python -m src.visualize --in outputs/analysis_summary.csv --out outputs/figures --pdf outputs/report.pdf
```

4. Check `outputs/` for cleaned data, figures, and the PDF report.

## Files

* `src/ingest.py` — read and concatenate CSVs robustly
* `src/clean.py` — cleaning rules described in the project brief
* `src/analyze.py` — produce key metrics and aggregations
* `src/visualize.py` — create charts listed in the brief and export a PDF report

## Notes

* Do not commit raw data to GitHub. Add `data/` to `.gitignore`.
* Pipeline scripts are standalone and designed to be simple to run locally.

```
```

---

## requirements.txt

```text
pandas>=1.3
numpy
matplotlib
seaborn
pyyaml
python-dateutil
pdfkit
reportlab
```

(You can remove `seaborn` if you prefer pure matplotlib; it's optional.)

---

## .gitignore

```text
__pycache__/
*.pyc
venv/
.env
data/
outputs/cleaned.csv
outputs/combined_raw.csv
outputs/report.pdf
.idea/
.vscode/
```

---

## src/ingest.py

```python
"""Ingest CSVs from a folder and combine into a single CSV."""
import argparse
from pathlib import Path
import pandas as pd


def read_csv_smart(path: Path):
    # pandas can usually read; catch encoding errors
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, encoding='latin-1', errors='replace')


def concat_csvs(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob('*.csv'))
    if not files:
        raise RuntimeError('No CSV files found in ' + str(data_dir))
    dfs = []
    for f in files:
        print('Reading', f)
        df = read_csv_smart(f)
        df['_source_file'] = f.name
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True, sort=False)
    return combined


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True, help='Folder with monthly CSVs')
    p.add_argument('--out', required=True, help='Output combined CSV path')
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out = Path(args.out)
    df = concat_csvs(data_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print('Wrote', out)
```

---

## src/clean.py

```python
"""Cleaning pipeline: normalize columns, compute ride_length, filter bad rows."""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Lowercase and replace spaces
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    return df


def parse_datetimes(df: pd.DataFrame):
    # common names: started_at, ended_at
    for col in ['started_at', 'start_time', 'startdate', 'start']:
        if col in df.columns:
            df['started_at'] = pd.to_datetime(df[col], errors='coerce')
            break
    for col in ['ended_at', 'end_time', 'enddate', 'end']:
        if col in df.columns:
            df['ended_at'] = pd.to_datetime(df[col], errors='coerce')
            break
    return df


def add_derived_cols(df: pd.DataFrame) -> pd.DataFrame:
    df['ride_length_min'] = (df['ended_at'] - df['started_at']).dt.total_seconds() / 60.0
    df['ride_length_min'] = df['ride_length_min'].astype(float)
    df['day_of_week'] = df['started_at'].dt.day_name()
    df['month'] = df['started_at'].dt.to_period('M')
    df['hour'] = df['started_at'].dt.hour
    return df


def clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)
    df = parse_datetimes(df)
    df = add_derived_cols(df)

    # Standardize rider type
    for col in ['member_casual', 'usertype', 'usertype', 'member_type']:
        if col in df.columns:
            df['rider_type'] = df[col].str.lower().str.strip()
            break
    if 'rider_type' not in df.columns:
        df['rider_type'] = np.nan

    # Filter invalid durations
    df = df[~df['ride_length_min'].isna()]
    df = df[df['ride_length_min'] >= 0]
    df = df[df['ride_length_min'] <= 24 * 60]

    # Drop duplicated ride ids if available
    for idcol in ['ride_id', 'rideid', 'id']:
        if idcol in df.columns:
            df = df.drop_duplicates(subset=idcol)
            break

    return df


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--in', dest='infile', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    df = pd.read_csv(args.infile)
    df_clean = clean_pipeline(df)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(args.out, index=False)
    print('Wrote cleaned data to', args.out)
```

---

## src/analyze.py

```python
"""Compute core metrics and export summary CSV."""
import argparse
from pathlib import Path
import pandas as pd


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    # Basic aggregates
    def agg_by(column):
        return df.groupby(column).agg(
            rides=('ride_length_min', 'count'),
            avg_duration_min=('ride_length_min', 'mean'),
            median_duration_min=('ride_length_min', 'median')
        ).reset_index()

    by_rider = agg_by('rider_type')
    by_day_user = df.groupby(['day_of_week', 'rider_type']).agg(rides=('ride_length_min','count')).reset_index()
    by_month_user = df.groupby(['month', 'rider_type']).agg(rides=('ride_length_min','count')).reset_index()
    by_hour_user = df.groupby(['hour', 'rider_type']).agg(rides=('ride_length_min','count')).reset_index()
    by_bike = agg_by('rideable_type') if 'rideable_type' in df.columns else pd.DataFrame()

    # Combine into a dict of dataframes saved to CSVs or a single summary
    summary = {'by_rider': by_rider, 'by_day_user': by_day_user, 'by_month_user': by_month_user, 'by_hour_user': by_hour_user, 'by_bike': by_bike}
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--in', required=True)
    p.add_argument('--out', required=True, help='folder to write summary CSVs')
    args = p.parse_args()

    df = pd.read_csv(args.in)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    summary = compute_metrics(df)
    for name, sdf in summary.items():
        if sdf is None or sdf.empty:
            continue
        sdf.to_csv(Path(args.out) / f'{name}.csv', index=False)
        print('Wrote', name)
```

---

## src/visualize.py

```python
"""Create recommended plots and export a PDF report using matplotlib."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def plot_avg_duration(by_rider_df, outdir: Path):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(by_rider_df['rider_type'], by_rider_df['avg_duration_min'])
    ax.set_ylabel('Avg Duration (min)')
    ax.set_title('Average Ride Duration: Member vs Casual')
    out = outdir / 'avg_duration_by_rider.png'
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def build_pdf_report(summary_dir: Path, out_pdf: Path):
    by_rider = pd.read_csv(summary_dir / 'by_rider.csv')
    by_day = pd.read_csv(summary_dir / 'by_day_user.csv')
    by_month = pd.read_csv(summary_dir / 'by_month_user.csv')

    with PdfPages(out_pdf) as pdf:
        # Page 1: avg duration
        fig, ax = plt.subplots(figsize=(8,6))
        ax.bar(by_rider['rider_type'], by_rider['avg_duration_min'])
        ax.set_title('Avg ride duration (min)')
        ax.set_ylabel('Minutes')
        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: heatmap (day vs rider)
        pivot = by_day.pivot(index='day_of_week', columns='rider_type', values='rides').reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
        fig, ax = plt.subplots(figsize=(8,6))
        im = ax.imshow(pivot.fillna(0).values, aspect='auto')
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title('Rides by day of week and rider type')
        pdf.savefig(fig)
        plt.close(fig)

        # Page 3: monthly trend
        mdf = by_month.copy()
        mdf['month'] = pd.PeriodIndex(mdf['month']).to_timestamp()
        fig, ax = plt.subplots(figsize=(10,5))
        for rt in mdf['rider_type'].unique():
            sub = mdf[mdf['rider_type']==rt]
            ax.plot(sub['month'], sub['rides'], label=rt)
        ax.legend()
        ax.set_title('Monthly trend by rider type')
        pdf.savefig(fig)
        plt.close(fig)

    print('Wrote PDF report to', out_pdf)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--in', required=True, help='folder with summary CSVs')
    p.add_argument('--pdf', required=True, help='output pdf path')
    args = p.parse_args()
    build_pdf_report(Path(args.in), Path(args.pdf))
```

---

## .github/workflows/python-app.yml

```yaml
name: Python package

on:
  push:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
      - name: Lint (optional)
        run: |
          python -m pip install flake8
          flake8 src || true
```

---

## Usage notes & next steps

* Replace quick heuristics with domain-specific logic as needed.
* Add unit tests for `clean.py` functions.
* Add an exploratory notebook in `notebooks/` to visualize intermediate steps.
