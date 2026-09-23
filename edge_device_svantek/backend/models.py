from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict

class FolderOut(BaseModel):
    name: str

class FolderCreate(BaseModel):
    name: str

class RecordingOut(BaseModel):
    id: int
    title: str
    file_format: str
    duration_sec: Optional[float] = None
    file_size_bytes: Optional[int] = None
    source: Literal["microphone", "upload", "svantek"]
    created_at: str
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    folder: Optional[str] = "Genel"
    tag: Optional[str] = None
    color: Optional[str] = "#f2a65a"
    has_csv: bool = False
    encryption_status: Optional[str] = "plain"
    encryption_algorithm: Optional[str] = None
    encrypted_at: Optional[str] = None
    plain_deleted: bool = False
    has_encrypted_audio: bool = False
    has_encrypted_csv: bool = False
    has_encrypted_raw: bool = False

class RecordingUpdate(BaseModel):
    title: Optional[str] = None
    folder: Optional[str] = None
    tag: Optional[str] = None
    color: Optional[str] = None

class RecordingStartResponse(BaseModel):
    status: str
    message: str

class RecordingStopResponse(BaseModel):
    status: str
    recording: RecordingOut

class WaveformOut(BaseModel):
    points: list[float]
    duration_sec: float

class CsvSeries(BaseModel):
    name: str
    values: list[float]

class CsvGraphOut(BaseModel):
    x_values: list[float]
    x_label: str = "Örnek"
    series: list[CsvSeries]
    rows: int


class NoiseEventOut(BaseModel):
    id: int | None = None
    start_sec: float
    end_sec: float
    label: str
    confidence: float


class RecordingAnalysisOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: str
    model_name: str | None = None
    error_message: str | None = None
    events: list[NoiseEventOut] = []
