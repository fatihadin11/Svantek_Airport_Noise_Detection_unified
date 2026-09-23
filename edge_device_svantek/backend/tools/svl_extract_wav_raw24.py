import argparse
import os
import struct
import wave
from pathlib import Path


def u16(data, offset):
    if offset + 2 > len(data):
        return None
    return struct.unpack_from("<H", data, offset)[0]


def find_audio_frames(data):
    frames = []

    for offset in range(0, len(data) - 8, 2):
        hs = u16(data, offset)

        if hs is None:
            continue

        # b15..b12 = 1001 -> 0x9xxx audio frame marker
        if (hs & 0xF000) != 0x9000:
            continue

        # b11 = 0 means HS / starting header
        if hs & 0x0800:
            continue

        length_words = u16(data, offset + 2)

        if length_words is None:
            continue

        if length_words < 4 or length_words > 20000:
            continue

        total_bytes = length_words * 2
        end_offset = offset + total_bytes

        if end_offset > len(data):
            continue

        tail_len = u16(data, end_offset - 4)
        he = u16(data, end_offset - 2)

        if he is None:
            continue

        # Tail L must equal starting L
        if tail_len != length_words:
            continue

        # HE: b15..b12 = 1001 and b11 = 1
        if (he & 0xF000) != 0x9000 or not (he & 0x0800):
            continue

        payload = data[offset + 4:end_offset - 4]

        # SVAN 971: 3 bytes per sample
        if len(payload) % 3 != 0:
            continue

        frames.append({
            "offset": offset,
            "end": end_offset,
            "hs": hs,
            "he": he,
            "length_words": length_words,
            "payload": payload,
            "first": bool(hs & 0x0400),
            "last": bool(hs & 0x0200),
            "error": bool(hs & 0x0080),
        })

    return frames


def pcm24_bytes_to_ints(pcm24_bytes):
    samples = []
    for i in range(0, len(pcm24_bytes), 3):
        value = pcm24_bytes[i] | (pcm24_bytes[i + 1] << 8) | (pcm24_bytes[i + 2] << 16)
        if value & 0x800000:
            value -= 0x1000000
        samples.append(value)
    return samples


def ints_to_pcm16_bytes(samples):
    if not samples:
        return b"", {"dc_offset": 0.0, "peak_before": 0.0, "scale": 1.0}

    dc_offset = sum(samples) / len(samples)
    centered = [sample - dc_offset for sample in samples]
    peak = max(abs(sample) for sample in centered)

    if peak <= 0:
        pcm16 = [0] * len(samples)
        scale = 1.0
    else:
        # %98 headroom: browser/player tarafında clipping riskini azaltır.
        scale = (32767 * 0.98) / peak
        pcm16 = []
        for sample in centered:
            value = int(round(sample * scale))
            value = max(-32768, min(32767, value))
            pcm16.append(value)

    payload = struct.pack("<" + "h" * len(pcm16), *pcm16)
    return payload, {
        "dc_offset": dc_offset,
        "peak_before": peak,
        "scale": scale,
    }


def write_wav(output_path, payload, sample_rate, sample_width):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--rate", type=int, default=8000)
    parser.add_argument(
        "--mode",
        choices=["web16", "raw24"],
        default=os.getenv("SVAN_WAV_OUTPUT_MODE", "web16"),
        help="web16: DC offset temizlenmiş/normalize 16-bit WAV. raw24: eski ham 24-bit çıktı.",
    )
    args = parser.parse_args()

    data = Path(args.input).read_bytes()
    frames = find_audio_frames(data)

    if not frames:
        raise SystemExit("Audio frame bulunamadı.")

    print("frames:", len(frames))

    total_payload = b"".join(frame["payload"] for frame in frames)
    total_samples = len(total_payload) // 3

    print("payload bytes:", len(total_payload))
    print("samples:", total_samples)
    print("duration:", total_samples / args.rate, "sec")

    print("first frame HS:", hex(frames[0]["hs"]), "first:", frames[0]["first"])
    print("last frame HS :", hex(frames[-1]["hs"]), "last:", frames[-1]["last"])
    print("error frames  :", sum(1 for f in frames if f["error"]))
    print("mode:", args.mode)

    output_path = Path(args.output)

    if args.mode == "raw24":
        write_wav(output_path, total_payload, args.rate, sample_width=3)
        print("written raw24:", args.output)
        return

    samples = pcm24_bytes_to_ints(total_payload)
    pcm16_payload, stats = ints_to_pcm16_bytes(samples)
    write_wav(output_path, pcm16_payload, args.rate, sample_width=2)

    print("dc_offset:", stats["dc_offset"])
    print("peak_before:", stats["peak_before"])
    print("normalize_scale:", stats["scale"])
    print("written web16:", args.output)


if __name__ == "__main__":
    main()
