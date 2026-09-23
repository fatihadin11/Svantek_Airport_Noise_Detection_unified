import csv
from collections import defaultdict
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from database import get_connection
from models import CsvGraphOut, CsvSeries
from services import encryption_service

router = APIRouter(prefix="/api/recordings", tags=["csv-views"])
MAX_POINTS = 2000


class CsvViewItem(BaseModel):
    id: str
    label: str
    view: str
    metric: Optional[str] = None
    rows: int


class CsvViewsOut(BaseModel):
    views: list[CsvViewItem]


def _csv_paths_for(recording_id: int) -> tuple[Optional[Path], Optional[Path]]:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT csv_path, csv_encrypted_path
            FROM recordings
            WHERE id = ?
            """,
            (recording_id,),
        ).fetchone()

        if row is None:
            raise HTTPException(status_code=404, detail="Kayıt bulunamadı")

        plain_path = Path(row["csv_path"]) if row["csv_path"] else None
        encrypted_path = Path(row["csv_encrypted_path"]) if row["csv_encrypted_path"] else None

        if plain_path is None and encrypted_path is None:
            raise HTTPException(status_code=404, detail="Bu kayıt için CSV dosyası yok")

        return plain_path, encrypted_path


def _read_rows(path: Path) -> list[dict]:
    last_error = None

    for encoding in ("utf-8-sig", "utf-8", "cp1254", "latin-1"):
        try:
            with path.open(newline="", encoding=encoding) as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError as exc:
            last_error = exc

    raise HTTPException(status_code=400, detail=f"CSV encoding okunamadı: {last_error}")


def _read_rows_for_recording(recording_id: int) -> list[dict]:
    plain_path, encrypted_path = _csv_paths_for(recording_id)

    if plain_path and plain_path.exists():
        return _read_rows(plain_path)

    if encrypted_path and encrypted_path.exists():
        try:
            with encryption_service.decrypt_file_to_temp(encrypted_path, suffix=".csv") as temp_csv:
                return _read_rows(temp_csv)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Şifreli CSV okunamadı: {exc}")

    raise HTTPException(status_code=404, detail="CSV dosyası bulunamadı")


def _to_float(value) -> Optional[float]:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except ValueError:
        return None


def _downsample(x_values: list[float], series: list[dict], max_points: int = MAX_POINTS):
    if len(x_values) <= max_points:
        return x_values, series

    step = len(x_values) / max_points
    indices = [min(len(x_values) - 1, int(i * step)) for i in range(max_points)]

    sampled_x = [x_values[i] for i in indices]
    sampled_series = [
        {"name": item["name"], "values": [item["values"][i] for i in indices]}
        for item in series
    ]

    return sampled_x, sampled_series


@router.get("/{recording_id}/csv-views", response_model=CsvViewsOut)
def list_csv_views(recording_id: int):
    rows = _read_rows_for_recording(recording_id)

    if not rows or "view" not in rows[0]:
        return CsvViewsOut(
            views=[
                CsvViewItem(
                    id="default",
                    label="CSV grafiği",
                    view="default",
                    rows=len(rows),
                )
            ]
        )

    counts = defaultdict(int)

    for row in rows:
        view = row.get("view", "")
        metric = row.get("metric", "") if view == "logger_octave" else ""
        counts[(view, metric)] += 1

    items: list[CsvViewItem] = []

    if counts.get(("logger_octave", "Leq")):
        items.append(
            CsvViewItem(
                id="logger_octave:Leq",
                label="Logger 1/1 Octave · Leq",
                view="logger_octave",
                metric="Leq",
                rows=counts[("logger_octave", "Leq")],
            )
        )

    if counts.get(("logger_octave", "Peak")):
        items.append(
            CsvViewItem(
                id="logger_octave:Peak",
                label="Logger 1/1 Octave · Peak",
                view="logger_octave",
                metric="Peak",
                rows=counts[("logger_octave", "Peak")],
            )
        )

    if counts.get(("wave", "")):
        items.append(
            CsvViewItem(
                id="wave:amplitude",
                label="Wave Results",
                view="wave",
                metric="amplitude",
                rows=counts[("wave", "")],
            )
        )

    if counts.get(("metadata", "")):
        items.append(
            CsvViewItem(
                id="metadata",
                label="Metadata",
                view="metadata",
                rows=counts[("metadata", "")],
            )
        )

    if counts.get(("summary_raw", "")):
        items.append(
            CsvViewItem(
                id="summary_raw",
                label="Summary Raw",
                view="summary_raw",
                rows=counts[("summary_raw", "")],
            )
        )

    return CsvViewsOut(views=items)


@router.get("/{recording_id}/csv-view-graph", response_model=CsvGraphOut)
def get_csv_view_graph(
    recording_id: int,
    view: str = Query("logger_octave"),
    metric: Optional[str] = Query("Leq"),
):
    rows = _read_rows_for_recording(recording_id)

    if not rows:
        return CsvGraphOut(x_values=[], series=[], rows=0)

    if "view" not in rows[0]:
        return CsvGraphOut(x_values=[], series=[], rows=len(rows))

    if view == "wave":
        selected = [
            r for r in rows
            if r.get("view") == "wave" and r.get("metric") == "amplitude"
        ]

        x_values, values = [], []

        for row in selected:
            x = _to_float(row.get("time_sec"))
            y = _to_float(row.get("value"))

            if x is not None and y is not None:
                x_values.append(x)
                values.append(y)

        series = [{"name": "Amplitude", "values": values}]
        x_values, series = _downsample(x_values, series)

        return CsvGraphOut(
            x_values=x_values,
            x_label="Saniye",
            series=[CsvSeries(**s) for s in series],
            rows=len(selected),
        )

    if view == "logger_octave":
        selected = [
            r for r in rows
            if r.get("view") == "logger_octave"
            and (not metric or r.get("metric") == metric)
        ]

        times = sorted({
            float(r["time_sec"])
            for r in selected
            if _to_float(r.get("time_sec")) is not None
        })

        band_labels = []

        for row in selected:
            label = row.get("band_label") or row.get("band_hz") or "value"

            if label and label not in band_labels:
                band_labels.append(label)

        by_key = {}

        for row in selected:
            t = _to_float(row.get("time_sec"))
            y = _to_float(row.get("value"))
            label = row.get("band_label") or row.get("band_hz") or "value"

            if t is not None and y is not None:
                by_key[(t, label)] = y

        series = []

        for label in band_labels:
            vals = [by_key.get((t, label)) for t in times]

            if all(v is not None for v in vals):
                series.append({"name": label, "values": vals})

        x_values, series = _downsample(times, series)

        return CsvGraphOut(
            x_values=x_values,
            x_label="Saniye",
            series=[CsvSeries(**s) for s in series],
            rows=len(selected),
        )

    return CsvGraphOut(x_values=[], series=[], rows=len(rows))
