import argparse
import csv
import struct
from pathlib import Path


CSV_COLUMNS = [
    "record_id",
    "source_file",
    "view",
    "time_sec",
    "metric",
    "band_hz",
    "band_label",
    "value",
    "unit",
    "sample_index",
    "raw_value",
    "offset_hex",
    "note",
]


STANDARD_OCTAVE_BANDS = [
    1, 2, 4, 8, 16,
    31.5, 63, 125, 250, 500,
    1000, 2000, 4000, 8000, 16000,
    31500,
]


def u16(data, offset):
    if offset + 2 > len(data):
        return None
    return struct.unpack_from("<H", data, offset)[0]


def u32_words(data, offset):
    lo = u16(data, offset)
    hi = u16(data, offset + 2)
    if lo is None or hi is None:
        return None
    return lo | (hi << 16)


def decode_db100(raw):
    if raw is None:
        return ""

    # Common undefined/sentinel values
    if raw in (0xFFFF, 0x7FFF, 0x8000):
        return ""

    # Usually positive dB*100. Keep signed fallback for safety.
    if raw >= 32768:
        raw = raw - 65536

    return raw / 100.0


def decode_s24le(payload, i):
    b0 = payload[i]
    b1 = payload[i + 1]
    b2 = payload[i + 2]

    v = b0 | (b1 << 8) | (b2 << 16)

    if v & 0x800000:
        v -= 1 << 24

    return v


def fmt_num(value):
    if value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def band_label(value):
    if value == "":
        return ""

    if abs(value - int(value)) < 1e-9:
        n = int(value)
        if n >= 1000:
            if n % 1000 == 0:
                return f"{n // 1000}k"
            return f"{n / 1000:g}k"
        return str(n)

    return f"{value:g}"


def make_octave_labels(lowest_freq_x100, noct, total_count):
    lowest_hz = lowest_freq_x100 / 100.0

    # Pick from standard list when possible, so 31.5, 63, 125, ...
    start_index = 0
    best_error = float("inf")

    for i, f in enumerate(STANDARD_OCTAVE_BANDS):
        err = abs(f - lowest_hz)
        if err < best_error:
            best_error = err
            start_index = i

    bands = []

    for i in range(noct):
        idx = start_index + i
        if idx < len(STANDARD_OCTAVE_BANDS):
            f = STANDARD_OCTAVE_BANDS[idx]
        else:
            f = lowest_hz * (2 ** i)

        bands.append((f, band_label(f)))

    for i in range(total_count):
        bands.append(("", f"total_{i + 1}"))

    return bands


def find_logger_header(data):
    candidates = []

    for offset in range(0, len(data) - 28, 2):
        word = u16(data, offset)
        if word is None:
            continue

        block_id = word & 0x00FF
        length_words = (word >> 8) & 0x00FF

        if block_id == 0x0F and 14 <= length_words <= 80:
            candidates.append(offset)

    if not candidates:
        raise RuntimeError("Logger header bulunamadı.")

    # Usually the first valid candidate is the real logger header.
    offset = candidates[0]

    buff_t_sec = u16(data, offset + 2) or 0
    buff_t_ms = u16(data, offset + 4) or 0
    lowest_freq = u16(data, offset + 6) or 0
    noct = u16(data, offset + 8) or 0
    noct_total = u16(data, offset + 10) or 0
    buff_length = u32_words(data, offset + 12) or 0
    records_in_buff = u32_words(data, offset + 16) or 0
    records_in_observ = u32_words(data, offset + 20) or 0
    audio_records = u32_words(data, offset + 24) or 0

    return {
        "offset": offset,
        "step_sec": buff_t_sec + buff_t_ms / 1000.0,
        "lowest_freq_x100": lowest_freq,
        "noct": noct,
        "noct_total": noct_total,
        "buff_length": buff_length,
        "records_in_buff": records_in_buff,
        "records_in_observ": records_in_observ,
        "audio_records": audio_records,
    }


def find_audio_frames(data):
    frames = []

    for offset in range(0, len(data) - 8, 2):
        hs = u16(data, offset)

        if hs is None:
            continue

        # b15..b12 = 9 means audio frame marker.
        if (hs & 0xF000) != 0x9000:
            continue

        # b11 = 0 means starting header.
        if hs & 0x0800:
            continue

        length_words = u16(data, offset + 2)

        if length_words is None or length_words < 4 or length_words > 20000:
            continue

        total_bytes = length_words * 2
        end_offset = offset + total_bytes

        if end_offset > len(data):
            continue

        tail_len = u16(data, end_offset - 4)
        he = u16(data, end_offset - 2)

        if he is None:
            continue

        valid_tail = tail_len == length_words
        valid_he = (he & 0xF000) == 0x9000 and (he & 0x0800)

        if not valid_tail or not valid_he:
            continue

        payload = data[offset + 4:end_offset - 4]

        # SVAN 971: 3 bytes per sample.
        if len(payload) % 3 != 0:
            continue

        frames.append(
            {
                "offset": offset,
                "end": end_offset,
                "hs": hs,
                "he": he,
                "length_words": length_words,
                "payload": payload,
                "first": bool(hs & 0x0400),
                "last": bool(hs & 0x0200),
                "error": bool(hs & 0x0080),
            }
        )

    return frames


def find_gaps_between_frames(data, frames):
    gaps = []

    for i in range(1, len(frames)):
        prev_end = frames[i - 1]["end"]
        curr_start = frames[i]["offset"]

        if curr_start != prev_end:
            gaps.append(
                {
                    "before_frame": i + 1,
                    "offset": prev_end,
                    "end": curr_start,
                    "bytes": data[prev_end:curr_start],
                }
            )

    return gaps


def find_summary_frames(data):
    frames = []

    for offset in range(0, len(data) - 4, 2):
        hs = u16(data, offset)

        if hs is None:
            continue

        high = (hs >> 8) & 0xFF
        low = hs & 0xFF

        # Summary frame HS high byte is C3. Ending HE high byte is CB.
        if high != 0xC3:
            continue

        if low == 0:
            length_words = u16(data, offset + 2)
            payload_start = offset + 4
            has_explicit_length = True
        else:
            length_words = low
            payload_start = offset + 2
            has_explicit_length = False

        if length_words is None or length_words < 2 or length_words > 10000:
            continue

        end_offset = offset + length_words * 2

        if end_offset > len(data):
            continue

        he = u16(data, end_offset - 2)

        if he is None:
            continue

        if ((he >> 8) & 0xFF) != 0xCB:
            continue

        if has_explicit_length:
            # HS + L + D + L + HE
            payload_end = end_offset - 4
        else:
            # HS + D + HE
            payload_end = end_offset - 2

        payload = data[payload_start:payload_end]

        frames.append(
            {
                "offset": offset,
                "end": end_offset,
                "hs": hs,
                "he": he,
                "length_words": length_words,
                "payload": payload,
            }
        )

    return frames


def write_row(writer, **kwargs):
    row = {key: "" for key in CSV_COLUMNS}

    for key, value in kwargs.items():
        if key in row:
            row[key] = fmt_num(value)

    writer.writerow(row)


def export_metadata(writer, record_id, source_file, header, audio_frame_count, summary_count):
    metadata = {
        "logger_step_sec": header["step_sec"],
        "lowest_freq_hz": header["lowest_freq_x100"] / 100.0,
        "octave_count": header["noct"],
        "total_octave_values": header["noct_total"],
        "logger_length_bytes": header["buff_length"],
        "records_in_buffer": header["records_in_buff"],
        "records_observed": header["records_in_observ"],
        "audio_records": header["audio_records"],
        "audio_frame_count": audio_frame_count,
        "summary_frame_count": summary_count,
    }

    for metric, value in metadata.items():
        write_row(
            writer,
            record_id=record_id,
            source_file=source_file,
            view="metadata",
            metric=metric,
            value=value,
            offset_hex=f"0x{header['offset']:08X}" if metric.startswith("logger") else "",
        )


def export_wave(writer, record_id, source_file, frames, sample_rate, wave_step):
    sample_index = 0
    written = 0

    for frame in frames:
        payload = frame["payload"]

        for i in range(0, len(payload), 3):
            raw = decode_s24le(payload, i)
            amplitude = raw / 8388608.0
            time_sec = sample_index / sample_rate

            if sample_index % wave_step == 0:
                write_row(
                    writer,
                    record_id=record_id,
                    source_file=source_file,
                    view="wave",
                    time_sec=time_sec,
                    metric="amplitude",
                    value=amplitude,
                    unit="normalized",
                    sample_index=sample_index,
                    raw_value=raw,
                    offset_hex=f"0x{frame['offset']:08X}",
                    note="24-bit little-endian sample from SVL audio frame",
                )
                written += 1

            sample_index += 1

    return written


def export_logger_octave(writer, record_id, source_file, gaps, header):
    n_bands = header["noct"] + header["noct_total"]
    expected_gap_len = n_bands * 2 * 2
    labels = make_octave_labels(
        header["lowest_freq_x100"],
        header["noct"],
        header["noct_total"],
    )

    rows = 0
    gap_index = 0

    for gap in gaps:
        raw = gap["bytes"]

        if len(raw) != expected_gap_len:
            # Keep unexpected gaps as raw, so no data is lost.
            write_row(
                writer,
                record_id=record_id,
                source_file=source_file,
                view="gap_raw",
                time_sec="",
                metric="raw_gap_bytes",
                value=len(raw),
                unit="bytes",
                raw_value=raw.hex(),
                offset_hex=f"0x{gap['offset']:08X}",
                note=f"Unexpected gap length before audio frame {gap['before_frame']}",
            )
            rows += 1
            continue

        words = [u16(raw, i) for i in range(0, len(raw), 2)]
        peak_words = words[:n_bands]
        leq_words = words[n_bands:]

        # Record is written between audio frame groups; assign it to the end of each logger step.
        time_sec = (gap_index + 1) * header["step_sec"]

        for metric_name, metric_words in [("Peak", peak_words), ("Leq", leq_words)]:
            for idx, word in enumerate(metric_words):
                band_hz, label = labels[idx]
                value = decode_db100(word)

                write_row(
                    writer,
                    record_id=record_id,
                    source_file=source_file,
                    view="logger_octave",
                    time_sec=time_sec,
                    metric=metric_name,
                    band_hz=band_hz,
                    band_label=label,
                    value=value,
                    unit="dB",
                    raw_value=word,
                    offset_hex=f"0x{gap['offset']:08X}",
                    note="Parsed from 52-byte logger gap as octave Peak/Leq dB*100",
                )
                rows += 1

        gap_index += 1

    return rows


def export_summary_raw(writer, record_id, source_file, summary_frames):
    rows = 0

    for frame_index, frame in enumerate(summary_frames, start=1):
        payload = frame["payload"]
        words = [u16(payload, i) for i in range(0, len(payload), 2)]

        for word_index, word in enumerate(words):
            write_row(
                writer,
                record_id=record_id,
                source_file=source_file,
                view="summary_raw",
                metric=f"summary_frame_{frame_index}_word_{word_index}",
                value=word,
                unit="raw_word",
                raw_value=word,
                offset_hex=f"0x{frame['offset']:08X}",
                note=(
                    f"HS=0x{frame['hs']:04X}; HE=0x{frame['he']:04X}; "
                    "raw summary/main/total data, naming map will be added later"
                ),
            )
            rows += 1

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input SVL file, e.g. downloads/88.SVL")
    parser.add_argument("--output", required=True, help="Output master CSV, e.g. downloads/88_all.csv")
    parser.add_argument("--record-id", default=None)
    parser.add_argument("--rate", type=int, default=8000)
    parser.add_argument(
        "--wave-step",
        type=int,
        default=1,
        help="Write every Nth waveform sample. Use 1 for full waveform.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    record_id = args.record_id or input_path.stem

    data = input_path.read_bytes()

    header = find_logger_header(data)
    audio_frames = find_audio_frames(data)
    gaps = find_gaps_between_frames(data, audio_frames)
    summary_frames = find_summary_frames(data)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        export_metadata(
            writer,
            record_id=record_id,
            source_file=input_path.name,
            header=header,
            audio_frame_count=len(audio_frames),
            summary_count=len(summary_frames),
        )

        wave_rows = export_wave(
            writer,
            record_id=record_id,
            source_file=input_path.name,
            frames=audio_frames,
            sample_rate=args.rate,
            wave_step=max(args.wave_step, 1),
        )

        octave_rows = export_logger_octave(
            writer,
            record_id=record_id,
            source_file=input_path.name,
            gaps=gaps,
            header=header,
        )

        summary_rows = export_summary_raw(
            writer,
            record_id=record_id,
            source_file=input_path.name,
            summary_frames=summary_frames,
        )

    print(f"Written: {output_path}")
    print(f"metadata rows      : 10")
    print(f"wave rows          : {wave_rows}")
    print(f"logger_octave rows : {octave_rows}")
    print(f"summary_raw rows   : {summary_rows}")
    print(f"audio frames       : {len(audio_frames)}")
    print(f"gaps               : {len(gaps)}")
    print(f"summary frames     : {len(summary_frames)}")
    print(f"logger step        : {header['step_sec']} s")
    print(f"octave bands       : {header['noct']} + total {header['noct_total']}")


if __name__ == "__main__":
    main()
