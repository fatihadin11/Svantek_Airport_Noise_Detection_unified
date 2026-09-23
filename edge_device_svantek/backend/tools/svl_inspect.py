import argparse
import struct
from pathlib import Path


TOP_LEVEL_BLOCKS = {
    0x01: "File header",
    0x02: "Unit and software specification",
    0x03: "Calibration settings",
    0x04: "User text / measurement header",
    0x05: "Global settings",
    0x06: "Measurement trigger parameters",
    0x0B: "Logger trigger / time-domain related block",
    0x0F: "Logger file header",
    0x59: "Summary Results Record header",
    0x07: "Main results in SLM mode",
}


def u16(data, offset):
    if offset + 2 > len(data):
        return None
    return struct.unpack_from("<H", data, offset)[0]


def u32_words(data, offset):
    """
    Read a 32-bit value stored as two little-endian words.
    Word at offset is low word, next word is high word.
    """
    lo = u16(data, offset)
    hi = u16(data, offset + 2)
    if lo is None or hi is None:
        return None
    return lo | (hi << 16)


def is_printable_ascii(raw):
    return all((32 <= b <= 126) or b == 0 for b in raw)


def parse_svanpc_header(data):
    print("=== SVL INSPECT ===")
    print(f"File size: {len(data)} bytes")

    magic = data[:6]
    print(f"Magic: {magic!r}")

    if magic.startswith(b"SvanPC"):
        print("SvanPC header: OK")
    else:
        print("SvanPC header: NOT FOUND")


def parse_top_level_blocks(data, max_blocks=80):
    print("\n=== TOP-LEVEL BLOCK SCAN ===")

    # SvanPC header is 16 words = 32 bytes.
    offset = 32
    blocks = []

    for _ in range(max_blocks):
        if offset + 2 > len(data):
            break

        word = u16(data, offset)

        if word is None:
            break

        block_id = word & 0x00FF
        block_len_words = (word >> 8) & 0x00FF

        # Basic sanity check.
        if block_len_words == 0:
            print(f"Stopped at 0x{offset:08X}: zero block length")
            break

        block_len_bytes = block_len_words * 2

        if offset + block_len_bytes > len(data):
            print(
                f"Stopped at 0x{offset:08X}: block too long "
                f"id=0x{block_id:02X}, len_words={block_len_words}"
            )
            break

        name = TOP_LEVEL_BLOCKS.get(block_id, "Unknown / not mapped")

        print(
            f"0x{offset:08X}  "
            f"id=0x{block_id:02X}  "
            f"len_words={block_len_words:<4}  "
            f"len_bytes={block_len_bytes:<5}  "
            f"{name}"
        )

        blocks.append((offset, block_id, block_len_words, block_len_bytes, name))

        offset += block_len_bytes

        # If the next area is logger contents, the general top-level block chain may end.
        # We do not force stop here, because we want to see where it naturally breaks.

    return blocks


def inspect_file_header(data, blocks):
    print("\n=== FILE HEADER DETAILS ===")

    file_blocks = [b for b in blocks if b[1] == 0x01]

    if not file_blocks:
        print("File header block not found.")
        return

    offset, block_id, length_words, length_bytes, _ = file_blocks[0]

    name_raw = data[offset + 2 : offset + 10]
    name = name_raw.decode("ascii", errors="replace").rstrip("\x00 ").strip()

    print(f"File header offset: 0x{offset:08X}")
    print(f"File name raw: {name_raw!r}")
    print(f"File name: {name}")


def inspect_logger_header(data, blocks):
    print("\n=== LOGGER HEADER DETAILS ===")

    candidates = []

    # First, use parsed top-level blocks.
    for b in blocks:
        offset, block_id, length_words, length_bytes, name = b
        if block_id == 0x0F:
            candidates.append(offset)

    # Also scan the whole file for 0xnn0F-like words.
    for offset in range(0, len(data) - 28, 2):
        word = u16(data, offset)
        if word is None:
            continue

        block_id = word & 0x00FF
        block_len_words = (word >> 8) & 0x00FF

        if block_id == 0x0F and 14 <= block_len_words <= 80:
            if offset not in candidates:
                candidates.append(offset)

    if not candidates:
        print("Logger header candidate not found.")
        return []

    logger_infos = []

    for offset in candidates:
        word = u16(data, offset)
        length_words = word >> 8

        buff_t_sec = u16(data, offset + 2)
        buff_t_ms = u16(data, offset + 4)
        lowest_freq = u16(data, offset + 6)
        noct_ter = u16(data, offset + 8)
        noct_ter_tot = u16(data, offset + 10)
        buff_length = u32_words(data, offset + 12)
        recs_in_buff = u32_words(data, offset + 16)
        recs_in_observ = u32_words(data, offset + 20)
        audio_records = u32_words(data, offset + 24)

        print(f"\nLogger header candidate at 0x{offset:08X}")
        print(f"  header word      : 0x{word:04X}")
        print(f"  length words     : {length_words}")
        print(f"  logger step      : {buff_t_sec}.{buff_t_ms:03d} s")
        print(f"  lowest freq      : {lowest_freq}")
        print(f"  octave count     : {noct_ter}")
        print(f"  total octave vals: {noct_ter_tot}")
        print(f"  logger length    : {buff_length} bytes")
        print(f"  records in buff  : {recs_in_buff}")
        print(f"  records observed : {recs_in_observ}")
        print(f"  audio records    : {audio_records}")

        logger_infos.append(
            {
                "offset": offset,
                "length_words": length_words,
                "buff_length": buff_length,
                "records": recs_in_buff,
                "audio_records": audio_records,
            }
        )

    return logger_infos


def scan_audio_frames(data, max_print=50):
    print("\n=== AUDIO FRAME SCAN ===")

    frames = []

    for offset in range(0, len(data) - 8, 2):
        hs = u16(data, offset)

        if hs is None:
            continue

        # Audio frame marker: b15..b12 = 9.
        if (hs & 0xF000) != 0x9000:
            continue

        # Starting header has b11 = 0.
        if hs & 0x0800:
            continue

        length_words = u16(data, offset + 2)

        if length_words is None:
            continue

        # Audio frame length must be plausible.
        if length_words < 4 or length_words > 20000:
            continue

        total_bytes = length_words * 2
        end_offset = offset + total_bytes

        if end_offset > len(data):
            continue

        tail_len = u16(data, end_offset - 4)
        he = u16(data, end_offset - 2)

        valid_tail_length = tail_len == length_words

        # Ending header: same audio marker, b11 = 1.
        valid_he = (
            he is not None
            and (he & 0xF000) == 0x9000
            and (he & 0x0800) != 0
        )

        sample_bytes = total_bytes - 8
        sample_count = sample_bytes // 3 if sample_bytes > 0 else 0

        first_frame = bool(hs & 0x0400)
        last_frame = bool(hs & 0x0200)
        error_flag = bool(hs & 0x0080)

        # Keep strong candidates.
        if valid_tail_length and valid_he and sample_count > 0:
            frames.append(
                {
                    "offset": offset,
                    "hs": hs,
                    "length_words": length_words,
                    "total_bytes": total_bytes,
                    "sample_count": sample_count,
                    "first": first_frame,
                    "last": last_frame,
                    "error": error_flag,
                    "end_offset": end_offset,
                    "he": he,
                }
            )

    if not frames:
        print("No valid audio frames found.")
        print("Bu, bu SVL içinde event audio yok demek olabilir veya frame yapısını biraz farklı okumamız gerekebilir.")
        return frames

    print(f"Valid audio frames found: {len(frames)}")

    for i, frame in enumerate(frames[:max_print], start=1):
        print(
            f"{i:03d}  "
            f"offset=0x{frame['offset']:08X}  "
            f"HS=0x{frame['hs']:04X}  "
            f"len_words={frame['length_words']:<6}  "
            f"samples={frame['sample_count']:<8}  "
            f"first={frame['first']}  "
            f"last={frame['last']}  "
            f"error={frame['error']}  "
            f"end=0x{frame['end_offset']:08X}  "
            f"HE=0x{frame['he']:04X}"
        )

    if len(frames) > max_print:
        print(f"... {len(frames) - max_print} more frames not printed")

    total_samples = sum(f["sample_count"] for f in frames)
    print(f"Total candidate audio samples: {total_samples}")

    return frames


def scan_summary_frames(data, max_print=30):
    print("\n=== SUMMARY / VIEW FRAME SCAN ===")

    frames = []

    for offset in range(0, len(data) - 4, 2):
        hs = u16(data, offset)

        if hs is None:
            continue

        high = (hs >> 8) & 0xFF
        low = hs & 0xFF

        # Summary frame HS high byte is C3. HE high byte is CB.
        if high != 0xC3:
            continue

        if low == 0:
            length_words = u16(data, offset + 2)
            data_start = offset + 4
        else:
            length_words = low
            data_start = offset + 2

        if length_words is None or length_words < 2 or length_words > 10000:
            continue

        total_bytes = length_words * 2
        end_offset = offset + total_bytes

        if end_offset > len(data):
            continue

        he = u16(data, end_offset - 2)

        if he is None:
            continue

        he_high = (he >> 8) & 0xFF

        if he_high == 0xCB:
            frames.append(
                {
                    "offset": offset,
                    "hs": hs,
                    "length_words": length_words,
                    "end_offset": end_offset,
                    "he": he,
                }
            )

    if not frames:
        print("No summary/view frames found.")
        return frames

    print(f"Summary/view frames found: {len(frames)}")

    for i, frame in enumerate(frames[:max_print], start=1):
        print(
            f"{i:03d}  "
            f"offset=0x{frame['offset']:08X}  "
            f"HS=0x{frame['hs']:04X}  "
            f"len_words={frame['length_words']:<6}  "
            f"end=0x{frame['end_offset']:08X}  "
            f"HE=0x{frame['he']:04X}"
        )

    if len(frames) > max_print:
        print(f"... {len(frames) - max_print} more frames not printed")

    return frames


def print_nearby_strings(data, min_len=4, max_print=80):
    print("\n=== PRINTABLE STRINGS ===")

    found = []
    current = bytearray()
    start = None

    for i, b in enumerate(data):
        if 32 <= b <= 126:
            if start is None:
                start = i
            current.append(b)
        else:
            if len(current) >= min_len:
                found.append((start, bytes(current).decode("ascii", errors="replace")))
            current = bytearray()
            start = None

    if len(current) >= min_len:
        found.append((start, bytes(current).decode("ascii", errors="replace")))

    for offset, text in found[:max_print]:
        print(f"0x{offset:08X}: {text}")

    if len(found) > max_print:
        print(f"... {len(found) - max_print} more strings not printed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="SVL file path, for example downloads/88.SVL")
    args = parser.parse_args()

    path = Path(args.file)

    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    data = path.read_bytes()

    parse_svanpc_header(data)
    blocks = parse_top_level_blocks(data)
    inspect_file_header(data, blocks)
    inspect_logger_header(data, blocks)
    scan_summary_frames(data)
    scan_audio_frames(data)
    print_nearby_strings(data)


if __name__ == "__main__":
    main()
