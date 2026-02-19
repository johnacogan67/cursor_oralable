"""
Oralable raw log parser: extract PPG, accelerometer, thermometer, and battery from BLE logs.

The same Bluetooth log file contains multiple characteristics; each parser filters by UUID:
- 3A0FF001: PPG (Red, IR, Green) at 50 Hz.
- 3A0FF002: Accelerometer (actigraphy, sync taps per .cursorrules) at ~100 Hz.
- 3A0FF003: Thermometer (temp_raw, temp_c).
- 3A0FF007 / 3A0FF008: Battery (if present in log).

Use parse_oralable_log() for PPG only, or parse_all() to get every stream in one pass.

Follows .cursorrules: PPG at 50 Hz (20 ms); accelerometer sync with 50 Hz PPG.
"""

from pathlib import Path
import re
from datetime import datetime

import pandas as pd


# BLE characteristic UUIDs in Oralable logs (match oralable_nrf tgm_service.h)
PPG_CHAR_UUID = "3A0FF001-98C4-46B2-94AF-1AEE0FD4C48E"
ACCEL_CHAR_UUID = "3A0FF002-98C4-46B2-94AF-1AEE0FD4C48E"
THERMOMETER_CHAR_UUID = "3A0FF003-98C4-46B2-94AF-1AEE0FD4C48E"
# Firmware uses 3A0FF004 for battery notify; 3A0FF007/008 are PPG register read/write
BATTERY_CHAR_UUIDS = (
    "3A0FF004-98C4-46B2-94AF-1AEE0FD4C48E",
    "3A0FF007-98C4-46B2-94AF-1AEE0FD4C48E",
    "3A0FF008-98C4-46B2-94AF-1AEE0FD4C48E",
)
# Samples per packet at 50Hz (firmware CONFIG_PPG_SAMPLES_PER_FRAME can be 20 or 50)
SAMPLES_PER_PACKET = 50
# Full PPG payload per oralable_nrf: 1 × 32-bit frame_counter + 50 × (Red, IR, Green) = 302 words
PPG_WORDS_PER_FULL_PACKET = 1 * 2 + SAMPLES_PER_PACKET * 3 * 2  # 302
# 20-sample config: 2 + 20×6 = 122 words (valid firmware setting, not truncation)
PPG_WORDS_20_SAMPLES = 2 + 20 * 6  # 122
# 50 Hz → 20 ms between samples
SAMPLE_INTERVAL_S = 1.0 / 50.0
# Accelerometer: .cursorrules say up to 100Hz, sync with 50Hz; packet often has 50 samples
ACCEL_SAMPLES_PER_PACKET = 50
ACCEL_SAMPLE_INTERVAL_S = 1.0 / 100.0  # 100 Hz typical

# PPG channel order per firmware tgm_service.h and oralable_swift OralableDevice.swift:
# Each sample = 12 bytes (3 × uint32 LE): [Red, IR, Green] at offsets 0, 4, 8
PPG_CHANNEL_ORDER = "R_IR_G"  # Red, IR, Green per sample (matches iOS app)

# -----------------------------------------------------------------------------
# Firmware byte layout (oralable_nrf / github.com/johnacogan67/oralable_nrf)
# -----------------------------------------------------------------------------
# PPG (3A0FF001): tgm_service_ppg_data_t = frame_counter + ppg_data[CONFIG_PPG_SAMPLES_PER_FRAME]
#   - Bytes 0-3:   frame_counter (uint32_t LE)
#   - Bytes 4-15:  sample 0: red (4), ir (4), green (4)  [struct ppg_sample]
#   - Bytes 16-27: sample 1: red, ir, green
#   - ... (20 samples → 244 bytes total when CONFIG_PPG_SAMPLES_PER_FRAME=20)
# ACC (3A0FF002): tgm_service_acc_data_t = frame_counter + acc_data[CONFIG_ACC_SAMPLES_PER_FRAME]
#   - Bytes 0-3:   frame_counter
#   - Bytes 4-9:   sample 0: x, y, z (int16_t LE each)
#   - ... (25 samples → 154 bytes when CONFIG_ACC_SAMPLES_PER_FRAME=25)
# tdm_interleave: Default False. Firmware sends per-sample (R,IR,G). Use True if BLE log
#   shows cyclic zeros (one channel per sample zero) from alternate FIFO/TDM capture.
# -----------------------------------------------------------------------------


def _angle_bracket_hex_to_word_hex(hex_str: str) -> str:
    """Convert 'notified: <ad0c0000 00000000 ...>' space-separated 8-char (32-bit) hex to space-separated 16-bit words (same format as CSV)."""
    tokens = hex_str.strip().split()
    words: list[str] = []
    for t in tokens:
        if len(t) != 8 or not all(c in "0123456789AaBbCcDdEeFf" for c in t):
            continue
        b0, b1, b2, b3 = (int(t[i : i + 2], 16) for i in (0, 2, 4, 6))
        w0 = b0 + (b1 << 8)
        w1 = b2 + (b3 << 8)
        words.append(f"{w0:04X}")
        words.append(f"{w1:04X}")
    return " ".join(words)


def _nrf_hyphen_hex_to_word_hex(hyphen_hex: str) -> str:
    """Convert nRF Connect 'value: (0x) A7-1F-00-00-...' hyphenated bytes to space-separated 16-bit words (same format as CSV ' to F600 0000 ...')."""
    parts = [p.strip() for p in hyphen_hex.replace(" ", "").split("-") if len(p.strip()) == 2 and all(c in "0123456789AaBbCcDdEeFf" for c in p.strip())]
    if not parts:
        return ""
    words: list[str] = []
    for i in range(0, len(parts), 4):
        if i + 3 >= len(parts):
            break
        b0, b1, b2, b3 = (int(parts[i + k], 16) for k in range(4))
        words.append(f"{b0:02X}{b1:02X}")
        words.append(f"{b2:02X}{b3:02X}")
    return " ".join(words)


def _parse_hex_payload(hex_str: str) -> list[int]:
    """Parse space-separated HEX into 32-bit little-endian values (ARM/nRF52).
    Handles both 8-char (32-bit) tokens and 4-char (16-bit) word pairs."""
    words = hex_str.strip().split()
    if not words:
        return []
    values = []
    # Detect format: 8-char = 32-bit LE per token; 4-char = 16-bit pairs
    first_len = len(words[0])
    if first_len == 8:
        for t in words:
            if len(t) != 8:
                continue
            b0, b1, b2, b3 = (int(t[i : i + 2], 16) for i in (0, 2, 4, 6))
            values.append(b0 + (b1 << 8) + (b2 << 16) + (b3 << 24))
    else:
        for i in range(0, len(words) - 1, 2):
            low = int(words[i], 16)
            high = int(words[i + 1], 16)
            values.append(low + (high << 16))
    return values


def _parse_hex_16bit_words(hex_str: str, signed: bool = True) -> list[int]:
    """Parse space-separated HEX 16-bit words. If signed, interpret as signed int16."""
    words = hex_str.strip().split()
    out = []
    for w in words:
        v = int(w, 16)
        if signed and v >= 0x8000:
            v -= 0x10000
        out.append(v)
    return out


def _hex_string_to_bytes(hex_str: str) -> list[int]:
    """Convert HEX string to list of bytes (little-endian where applicable).
    Handles hyphenated (A7-1F-00), space-separated 4-char (7619 0000), or 8-char (ad0c0000)."""
    s = hex_str.strip()
    if "-" in s:
        parts = [p.strip() for p in s.split("-") if len(p.strip()) == 2]
        return [int(p, 16) for p in parts if all(c in "0123456789AaBbCcDdEeFf" for c in p)]
    tokens = s.split()
    out: list[int] = []
    for t in tokens:
        t = t.strip()
        if len(t) == 4:
            # 16-bit LE word -> low byte, high byte
            out.append(int(t[2:4], 16))
            out.append(int(t[0:2], 16))
        elif len(t) == 8:
            # 32-bit LE -> 4 bytes
            out.append(int(t[6:8], 16))
            out.append(int(t[4:6], 16))
            out.append(int(t[2:4], 16))
            out.append(int(t[0:2], 16))
        else:
            for i in range(0, len(t), 2):
                if i + 1 < len(t):
                    out.append(int(t[i : i + 2], 16))
    return out


def _parse_12byte_chunk_to_row(bytes_chunk: list[int], ts_s: float) -> tuple[float, int, int, int, int, int, int]:
    """Convert 12 bytes to (timestamp_s, green, red, ir, accel_x, accel_y, accel_z). Accel is signed int16 LE."""
    if len(bytes_chunk) < 12:
        raise ValueError("Need at least 12 bytes")
    green = bytes_chunk[0] + (bytes_chunk[1] << 8)
    red = bytes_chunk[2] + (bytes_chunk[3] << 8)
    ir = bytes_chunk[4] + (bytes_chunk[5] << 8)

    def signed16(lo: int, hi: int) -> int:
        v = lo + (hi << 8)
        return v - 0x10000 if v >= 0x8000 else v

    accel_x = signed16(bytes_chunk[6], bytes_chunk[7])
    accel_y = signed16(bytes_chunk[8], bytes_chunk[9])
    accel_z = signed16(bytes_chunk[10], bytes_chunk[11])
    return (ts_s, green, red, ir, accel_x, accel_y, accel_z)


def _extract_raw_hex_from_line(line: str) -> str | None:
    """Extract raw HEX string from a log line (for 12-byte parsing). Handles nRF 'value: (0x)', iOS 'notified: <...>', CSV ' to '."""
    if "value: (0x)" in line:
        start = line.find("value: (0x)") + len("value: (0x) ")
        return line[start:].strip()
    if "notified:" in line and "<" in line and ">" in line:
        start = line.find("<") + 1
        end = line.find(">", start)
        if start > 0 and end > start:
            return line[start:end].strip()
    if " to " in line and ("3A0FF001" in line or "3A0FF002" in line or "3a0ff001" in line or "3a0ff002" in line):
        start = line.find(" to ") + 4
        end = line.rfind(".")
        if end == -1:
            end = len(line)
        return line[start:end].strip()
    return None


def parse_nrf_log_to_raw_csv(log_path: str | Path, out_path: str | Path | None = None) -> pd.DataFrame:
    """
    Use regex to find timestamps and HEX strings in an nRF Connect (or compatible) log.
    Convert 12-byte HEX blocks into 6 columns: green, red, ir, accel_x, accel_y, accel_z.
    Accelerometer values are signed 16-bit. Write a raw CSV to data/processed/ (default: raw.csv).
    """
    log_path = Path(log_path)
    df_log = _read_log_rows(log_path)
    rows: list[tuple[float, int, int, int, int, int, int]] = []
    for _, row in df_log.iterrows():
        line = str(row["Line"])
        raw_hex = _extract_raw_hex_from_line(line)
        if not raw_hex:
            continue
        try:
            ts_s = _parse_log_timestamp(str(row["Timestamp"]))
        except Exception:
            continue
        bytes_list = _hex_string_to_bytes(raw_hex)
        for i in range(0, len(bytes_list), 12):
            if i + 12 > len(bytes_list):
                break
            chunk = bytes_list[i : i + 12]
            rows.append(_parse_12byte_chunk_to_row(chunk, ts_s))
    df = pd.DataFrame(
        rows,
        columns=["timestamp_s", "green", "red", "ir", "accel_x", "accel_y", "accel_z"],
    )
    if out_path is None:
        processed_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        out_path = processed_dir / "raw.csv"
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def _extract_hex_from_line(line: str, uuid: str | None = None) -> str | None:
    """Extract HEX payload from a log line (after ' to ', from quoted value, or nRF 'value: (0x) XX-YY-...'). If uuid is set, only match that characteristic."""
    uuid = uuid or PPG_CHAR_UUID
    # "Updated Value of Characteristic ... to F600 0000 ..."
    if " to " in line and uuid in line and "Updated Value of Characteristic" in line:
        start = line.find(" to ") + 4
        end = line.rfind(".")
        if end == -1:
            end = len(line)
        return line[start:end].strip()
    # nRF Connect Android: "Notification received from 3a0ff001-..., value: (0x) A7-1F-00-00-..."
    if "value: (0x) " in line and uuid in line.lower():
        start = line.find("value: (0x) ") + len("value: (0x) ")
        rest = line[start:].strip()
        return _nrf_hyphen_hex_to_word_hex(rest)
    # iOS / dash format: "Characteristic (3A0FF001-...) notified: <ad0c0000 00000000 ...>"
    if "notified:" in line and "Characteristic" in line and "<" in line and ">" in line and uuid.lower() in line.lower():
        start = line.find("<") + 1
        end = line.find(">", start)
        if start > 0 and end > start:
            return _angle_bracket_hex_to_word_hex(line[start:end])
    # ""F600 0000 ..." value received."
    if uuid not in line:
        return None
    match = re.search(r'"([0-9A-Fa-f][0-9A-Fa-f](?:\s+[0-9A-Fa-f]{4})+)\s*"\s*value received', line)
    if match:
        return match.group(1).strip()
    return None


def _parse_log_timestamp(ts_str: str) -> float:
    """Parse log timestamp HH:MM:SS.ffff to seconds since midnight."""
    # Handle optional date; log format is typically HH:MM:SS.ffff
    ts_str = ts_str.strip()
    try:
        dt = datetime.strptime(ts_str, "%H:%M:%S.%f")
    except ValueError:
        dt = datetime.strptime(ts_str, "%H:%M:%S")
    return dt.hour * 3600.0 + dt.minute * 60.0 + dt.second + dt.microsecond / 1e6


def check_ppg_truncation(csv_path: str | Path) -> dict:
    """
    Check whether PPG lines in the log appear truncated (e.g. app/export line length limit).

    Full PPG packet per oralable_nrf: 1 frame_counter + 50×(R,IR,G) = 302 hex words. If lines
    have fewer words, the log source may be truncating (or firmware uses CONFIG_PPG_SAMPLES_PER_FRAME=20
    → 122 words). Returns e.g. {"truncated": True, "sample_word_count": 122, "expected_words": 302, ...}.

    Typical cause: the app that records BLE notifications (e.g. nRF Connect) or the CSV export
    limits log line length (~682 chars). See data/raw/README.md for details.
    """
    csv_path = Path(csv_path)
    df_log = _read_log_rows(csv_path)
    rows = _iter_characteristic_payloads(df_log, PPG_CHAR_UUID)
    if not rows:
        return {"truncated": False, "sample_word_count": 0, "expected_words": PPG_WORDS_PER_FULL_PACKET, "samples_per_packet": 0}
    word_counts = [len(hex_str.strip().split()) for _, hex_str in rows]
    min_w = min(word_counts)
    max_w = max(word_counts)
    expected = PPG_WORDS_PER_FULL_PACKET
    # 122 words = 20 samples (firmware CONFIG_PPG_SAMPLES_PER_FRAME=20), not truncation
    firmware_20_samples = min_w == PPG_WORDS_20_SAMPLES and max_w == PPG_WORDS_20_SAMPLES
    truncated = max_w < expected and not firmware_20_samples
    # 1 frame_counter (2 words) + n*3 channel values*2 words = 2 + 6n words for n samples
    n_values = min_w // 2
    samples_per_packet = (n_values - 1) // 3 if n_values >= 1 + 3 else 0
    return {
        "truncated": truncated,
        "firmware_20_samples": firmware_20_samples,
        "sample_word_count": min_w,
        "max_word_count": max_w,
        "expected_words": expected,
        "samples_per_packet": samples_per_packet,
    }


def _parse_ppg_sample(
    payload: list[int], j: int, bits: int = 32, channel_order: str = "G_R_IR"
) -> tuple[int, int, int] | None:
    """Extract one PPG sample (green, red, ir) from payload at offset j.
    bits: 32 (default) or 24 (mask 0xFFFFFF for packed 24-bit).
    channel_order: firmware triple order, e.g. 'G_R_IR' or 'R_IR_G'. ARM/nRF52 LE."""
    if j + 2 >= len(payload):
        return None
    mask = 0xFFFFFF if bits == 24 else 0xFFFFFFFF
    v0 = payload[j] & mask
    v1 = payload[j + 1] & mask
    v2 = payload[j + 2] & mask
    if v0 == 0 and v1 == 0 and v2 == 0:
        return None  # ghost sample
    order = channel_order.upper().split("_")
    out = {}
    for ch, val in zip(order, (v0, v1, v2)):
        out[ch] = val
    return (out["G"], out["R"], out["IR"])


def parse_oralable_log(
    csv_path: str | Path,
    channel_order: str | None = None,
    bits: int = 32,
    discard_ghost: bool = True,
    tdm_interleave: bool = False,
) -> pd.DataFrame:
    """
    Parse an Oralable CSV log and return PPG samples at 50Hz.

    Byte layout per oralable_nrf tgm_service.h (see module-level firmware layout comment):
    - Bytes 0-3: Frame counter (uint32_t LE)
    - Bytes 4+: 20 samples × 12 bytes (3 × uint32_t: Red, IR, Green per sample)

    tdm_interleave: If True, treat as TDM (every 3 raw samples = one triplet). Default False
    matches firmware: each sample has (R, IR, G). Use True if BLE log shows cyclic zeros
    (one channel per sample zero) from alternate FIFO or capture path.
    """
    csv_path = Path(csv_path)
    df_log = _read_log_rows(csv_path)
    order = channel_order or PPG_CHANNEL_ORDER

    rows = []
    for _, row in df_log.iterrows():
        line = str(row["Line"])
        if PPG_CHAR_UUID.lower() not in line.lower():
            continue
        is_updated = "Updated Value of Characteristic" in line and " to " in line
        is_nrf_notif = "Notification received" in line and "value: (0x)" in line
        is_angle_notif = "Characteristic" in line and "notified:" in line and "<" in line
        if not (is_updated or is_nrf_notif or is_angle_notif):
            continue
        hex_str = _extract_hex_from_line(line)
        if not hex_str:
            continue
        if not re.match(r"^[0-9A-Fa-f]", hex_str):
            continue
        ts_s = _parse_log_timestamp(str(row["Timestamp"]))
        rows.append((ts_s, hex_str))

    if not rows:
        return pd.DataFrame(columns=["timestamp_s", "green", "red", "ir"])

    all_ts: list[float] = []
    all_green: list[int] = []
    all_red: list[int] = []
    all_ir: list[int] = []

    for ts_packet, hex_str in rows:
        values = _parse_hex_payload(hex_str)
        if len(values) < 1 + 3:
            continue
        payload = values[1:]  # reset offset: skip frame_counter at start of each packet
        n_raw = min(SAMPLES_PER_PACKET, len(payload) // 3)

        if tdm_interleave:
            # TDM: BLE log shows cyclic zeros (one channel missing per slot).
            # Slot 3k has (IR, G), slot 3k+1 has (R, G), slot 3k+2 has (R, IR).
            # Combine: R from slot 3k+1, IR from slot 3k, G from slot 3k.
            n_triplets = n_raw // 3
            for k in range(n_triplets):
                j0, j1, j2 = (3 * k) * 3, (3 * k + 1) * 3, (3 * k + 2) * 3
                s0 = _parse_ppg_sample(payload, j0, bits=bits, channel_order=order)
                s1 = _parse_ppg_sample(payload, j1, bits=bits, channel_order=order)
                s2 = _parse_ppg_sample(payload, j2, bits=bits, channel_order=order)
                if s0 is None:
                    s0 = (0, 0, 0)
                if s1 is None:
                    s1 = (0, 0, 0)
                if s2 is None:
                    s2 = (0, 0, 0)
                # _parse_ppg_sample returns (G, R, IR); s[0]=G, s[1]=R, s[2]=IR
                g = s0[0]   # G from slot 3k
                r = s1[1]   # R from slot 3k+1
                ir_val = s0[2]  # IR from slot 3k
                if discard_ghost and g == 0 and r == 0 and ir_val == 0:
                    continue
                t = ts_packet + (3 * k + 1) * SAMPLE_INTERVAL_S  # center of triplet
                all_ts.append(t)
                all_green.append(g)
                all_red.append(r)
                all_ir.append(ir_val)
        else:
            for i in range(n_raw):
                j = i * 3
                sample = _parse_ppg_sample(payload, j, bits=bits, channel_order=order)
                if sample is None:
                    if discard_ghost:
                        continue
                    sample = (0, 0, 0)
                g, r, ir_val = sample
                t = ts_packet + i * SAMPLE_INTERVAL_S
                all_ts.append(t)
                all_green.append(g)
                all_red.append(r)
                all_ir.append(ir_val)

    return pd.DataFrame({
        "timestamp_s": all_ts,
        "green": all_green,
        "red": all_red,
        "ir": all_ir,
    })


def parse_oralable_log_to_elapsed(csv_path: str | Path) -> pd.DataFrame:
    """
    Same as parse_oralable_log but timestamps are elapsed seconds from first sample (for 50Hz pipelines).
    """
    df = parse_oralable_log(csv_path)
    if df.empty:
        return df
    t0 = df["timestamp_s"].iloc[0]
    df = df.copy()
    df["elapsed_s"] = df["timestamp_s"] - t0
    return df


def _read_log_rows(path: Path) -> pd.DataFrame:
    """Load log from CSV or TXT. Returns DataFrame with columns Timestamp, Line."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")

    if path.suffix.lower() == ".txt":
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        # nRF Connect Android: Level\tHH:MM:SS.mmm\tMessage (first line often "nRF Connect, 2026-02-07")
        nrf_pattern = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d+$")
        def _is_nrf_format() -> bool:
            if not lines:
                return False
            if "nRF Connect" in lines[0]:
                return True
            for line in lines[:50]:
                if "\t" not in line:
                    continue
                parts = line.split("\t", 2)
                if len(parts) >= 3 and nrf_pattern.match(parts[1].strip()):
                    return True
            return False
        is_nrf = _is_nrf_format()
        rows_ts: list[str] = []
        rows_line: list[str] = []
        for line in lines:
            if is_nrf:
                parts = line.split("\t", 2)
                if len(parts) >= 3 and nrf_pattern.match(parts[1].strip()):
                    rows_ts.append(parts[1].strip())
                    rows_line.append(parts[2].strip())
            else:
                m = re.match(r"\[(\d{2}:\d{2}:\d{2}\.\d+)\]\s*\S+:\s*(.*)", line)
                if m:
                    rows_ts.append(m.group(1))
                    rows_line.append(m.group(2).strip())
                else:
                    # iOS / dash format: "HH:MM:SS.mmm - Characteristic (3A0FF001-...) notified: <...>"
                    m2 = re.match(r"^(\d{2}:\d{2}:\d{2}\.\d+)\s+-\s+(.*)", line)
                    if m2:
                        rows_ts.append(m2.group(1))
                        rows_line.append(m2.group(2).strip())
        return pd.DataFrame({"Timestamp": rows_ts, "Line": rows_line})

    df = pd.read_csv(path)
    if "Timestamp" not in df.columns or "Line" not in df.columns:
        raise ValueError("CSV must have columns: Timestamp, Line")
    return df


def _iter_characteristic_payloads(
    df_log: pd.DataFrame, uuid: str
) -> list[tuple[float, str]]:
    """Yield (timestamp_s, hex_str) for lines: 'Updated Value... to <HEX>', nRF 'Notification... value: (0x) XX-YY-...', or iOS 'Characteristic (uuid) notified: <hex>'."""
    uuid_lower = uuid.lower()
    rows = []
    for _, row in df_log.iterrows():
        line = str(row["Line"])
        if uuid_lower not in line.lower():
            continue
        is_updated = "Updated Value of Characteristic" in line and " to " in line
        is_nrf_notif = "Notification received" in line and "value: (0x)" in line
        is_angle_notif = "Characteristic" in line and "notified:" in line and "<" in line
        if not (is_updated or is_nrf_notif or is_angle_notif):
            continue
        hex_str = _extract_hex_from_line(line, uuid)
        if not hex_str or not re.match(r"^[0-9A-Fa-f]{4}\s+[0-9A-Fa-f]{4}", hex_str):
            continue
        ts_s = _parse_log_timestamp(str(row["Timestamp"]))
        rows.append((ts_s, hex_str))
    return rows


def parse_accelerometer_log(csv_path: str | Path) -> pd.DataFrame:
    """
    Parse accelerometer (3A0FF002) from Oralable log. Used for actigraphy and sync taps (.cursorrules).

    Payload per oralable_nrf (tgm_service_acc_data_t): 4-byte frame_counter, then N × (x, y, z) signed 16-bit.
    Returns DataFrame with timestamp_s, accel_x, accel_y, accel_z (sample spacing ~10 ms at 100 Hz).
    """
    csv_path = Path(csv_path)
    df_log = _read_log_rows(csv_path)
    rows = _iter_characteristic_payloads(df_log, ACCEL_CHAR_UUID)
    if not rows:
        return pd.DataFrame(columns=["timestamp_s", "accel_x", "accel_y", "accel_z"])

    all_ts: list[float] = []
    all_x: list[int] = []
    all_y: list[int] = []
    all_z: list[int] = []

    for ts_packet, hex_str in rows:
        words = _parse_hex_16bit_words(hex_str, signed=True)
        if len(words) < 2 + 3:
            continue
        payload = words[2:]  # skip 2-word header (e.g. packet id)
        n_samples = min(ACCEL_SAMPLES_PER_PACKET, len(payload) // 3)
        for i in range(n_samples):
            j = i * 3
            if j + 2 >= len(payload):
                break
            t = ts_packet + i * ACCEL_SAMPLE_INTERVAL_S
            all_ts.append(t)
            all_x.append(payload[j])
            all_y.append(payload[j + 1])
            all_z.append(payload[j + 2])

    return pd.DataFrame({
        "timestamp_s": all_ts,
        "accel_x": all_x,
        "accel_y": all_y,
        "accel_z": all_z,
    })


def parse_thermometer_log(csv_path: str | Path) -> pd.DataFrame:
    """
    Parse thermometer (3A0FF003) from Oralable log.

    Payload per oralable_nrf (tgm_service_temp_data_t): 4-byte frame_counter, then int16 centidegree C
    (bytes 4-5, signed). temp_c = temp_raw / 100.
    Returns DataFrame with timestamp_s, sequence, temp_raw, temp_c.
    """
    csv_path = Path(csv_path)
    df_log = _read_log_rows(csv_path)
    rows = _iter_characteristic_payloads(df_log, THERMOMETER_CHAR_UUID)
    if not rows:
        return pd.DataFrame(columns=["timestamp_s", "sequence", "temp_raw", "temp_c"])

    all_ts: list[float] = []
    all_seq: list[int] = []
    all_temp_raw: list[int] = []

    for ts_s, hex_str in rows:
        values = _parse_hex_payload(hex_str)
        if len(values) < 2:
            continue
        all_ts.append(ts_s)
        all_seq.append(values[0])
        raw = values[1] & 0xFFFF
        if raw >= 0x8000:
            raw -= 0x10000
        all_temp_raw.append(raw)

    df = pd.DataFrame({
        "timestamp_s": all_ts,
        "sequence": all_seq,
        "temp_raw": all_temp_raw,
    })
    df["temp_c"] = df["temp_raw"] / 100.0
    return df


def parse_battery_log(csv_path: str | Path) -> pd.DataFrame:
    """
    Parse battery characteristic(s) (3A0FF004 battery notify; 3A0FF007/008 if present) from Oralable log.

    Returns DataFrame with timestamp_s, characteristic (UUID suffix), and payload columns.
    If no battery updates are in the log, returns empty DataFrame.
    """
    csv_path = Path(csv_path)
    df_log = _read_log_rows(csv_path)
    rows: list[tuple[float, str, str]] = []

    for _, row in df_log.iterrows():
        line = str(row["Line"])
        for uuid in BATTERY_CHAR_UUIDS:
            if uuid.lower() not in line.lower():
                continue
            is_updated = "Updated Value of Characteristic" in line and " to " in line
            is_angle = "notified:" in line and "<" in line
            if not (is_updated or is_angle):
                continue
            hex_str = _extract_hex_from_line(line, uuid)
            if not hex_str:
                continue
            ts_s = _parse_log_timestamp(str(row["Timestamp"]))
            rows.append((ts_s, uuid.split("-")[0][-4:], hex_str))
            break  # one line matches one uuid

    if not rows:
        return pd.DataFrame(columns=["timestamp_s", "characteristic", "hex_payload"])

    rows.sort(key=lambda x: x[0])
    return pd.DataFrame(rows, columns=["timestamp_s", "characteristic", "hex_payload"])


def parse_all(csv_path: str | Path, tdm_interleave: bool = True) -> dict[str, pd.DataFrame]:
    """
    Parse PPG, accelerometer, thermometer, and battery from one Oralable log.

    Returns dict with keys: "ppg", "accelerometer", "thermometer", "battery".
    Missing or empty streams are still present as keys with empty DataFrames (with correct columns).

    tdm_interleave: Default True. BLE logs show cyclic zeros (one channel per sample);
    use True to combine every 3 TDM slots into (R, IR, G) triplets.
    """
    csv_path = Path(csv_path)
    return {
        "ppg": parse_oralable_log(csv_path, tdm_interleave=tdm_interleave),
        "accelerometer": parse_accelerometer_log(csv_path),
        "thermometer": parse_thermometer_log(csv_path),
        "battery": parse_battery_log(csv_path),
    }


if __name__ == "__main__":
    import sys

    raw_dir = Path(__file__).resolve().parents[2] / "data" / "raw"
    processed_dir = raw_dir.parent / "processed"
    default_path = raw_dir / "Oralable.csv"
    if not default_path.exists() and raw_dir.exists():
        # Fallback: most recent Oralable_*.txt (from ble_logger)
        candidates = sorted(raw_dir.glob("Oralable_*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            default_path = candidates[0]

    args = [a for a in sys.argv[1:] if a != "--raw"]
    use_raw_12byte = "--raw" in sys.argv[1:]
    path = Path(args[0]) if args else default_path

    if use_raw_12byte:
        df = parse_nrf_log_to_raw_csv(path)
        out = processed_dir / "raw.csv"
        print(f"Wrote {len(df)} rows to {out}")
        print(df.head(10))
        sys.exit(0)

    # Report PPG packet size (20 or 50 samples; 20 = firmware config, not truncation)
    ppg_check = check_ppg_truncation(path)
    if ppg_check.get("firmware_20_samples"):
        print(f"PPG: 20 samples/packet (firmware CONFIG_PPG_SAMPLES_PER_FRAME=20).")
    elif ppg_check.get("truncated"):
        print(
            f"PPG truncation detected: {ppg_check['sample_word_count']} words per line "
            f"(expected {ppg_check['expected_words']}) → ~{ppg_check['samples_per_packet']} samples/packet. "
            "Increase log/export line length or use a non-truncating export."
        )
    elif ppg_check.get("sample_word_count"):
        print(f"PPG: full packets ({ppg_check['sample_word_count']} words/line, 50 samples/packet).")

    all_streams = parse_all(path)

    processed_dir.mkdir(parents=True, exist_ok=True)
    for name, df in all_streams.items():
        n = len(df)
        print(f"{name}: {n} rows")
        if not df.empty:
            print(df.head(5))
            df.to_csv(processed_dir / f"{name}_50hz.csv" if name == "ppg" else processed_dir / f"{name}.csv", index=False)
    print(f"Wrote CSVs to {processed_dir}")
