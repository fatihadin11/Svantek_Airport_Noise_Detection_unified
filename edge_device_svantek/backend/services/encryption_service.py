import argparse
import base64
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAGIC = b"PISESENC1\n"
NONCE_SIZE = 12
KEY_SIZE = 32
KEY_ENV_NAME = "AES_KEY_B64"
ALGORITHM_NAME = "AES-256-GCM"


class EncryptionConfigError(RuntimeError):
    pass


class EncryptionFormatError(ValueError):
    pass


def generate_key_b64() -> str:
    """AES-256 için 32 byte rastgele key üretip base64 döndürür."""
    return base64.b64encode(AESGCM.generate_key(bit_length=256)).decode("ascii")


def get_key_from_env(env_name: str = KEY_ENV_NAME) -> bytes:
    value = os.getenv(env_name, "").strip()
    if not value:
        raise EncryptionConfigError(
            f"{env_name} ortam değişkeni tanımlı değil. "
            "Önce AES key üretip backend ve worker tarafında aynı değeri kullanın."
        )

    try:
        key = base64.b64decode(value, validate=True)
    except Exception as exc:
        raise EncryptionConfigError(f"{env_name} geçerli base64 değil") from exc

    if len(key) != KEY_SIZE:
        raise EncryptionConfigError(f"{env_name} 32 byte olmalı; şu an {len(key)} byte")

    return key


def encrypt_bytes(plain_data: bytes, key: bytes | None = None) -> bytes:
    key = key or get_key_from_env()
    nonce = os.urandom(NONCE_SIZE)
    encrypted_payload = AESGCM(key).encrypt(nonce, plain_data, None)
    return MAGIC + nonce + encrypted_payload


def decrypt_bytes(encrypted_data: bytes, key: bytes | None = None) -> bytes:
    if not encrypted_data.startswith(MAGIC):
        raise EncryptionFormatError("Dosya bu proje için beklenen şifreli formatta değil")

    payload = encrypted_data[len(MAGIC):]
    if len(payload) <= NONCE_SIZE:
        raise EncryptionFormatError("Şifreli dosya eksik veya bozuk")

    key = key or get_key_from_env()
    nonce = payload[:NONCE_SIZE]
    ciphertext = payload[NONCE_SIZE:]
    return AESGCM(key).decrypt(nonce, ciphertext, None)


def is_encrypted_blob(data: bytes) -> bool:
    return data.startswith(MAGIC)


def encrypted_path_for(path: Path) -> Path:
    return path.with_name(path.name + ".enc")


def encrypt_file(source_path: Path, destination_path: Path | None = None, key: bytes | None = None) -> Path:
    source_path = Path(source_path)
    destination_path = Path(destination_path) if destination_path else encrypted_path_for(source_path)
    destination_path.write_bytes(encrypt_bytes(source_path.read_bytes(), key=key))
    return destination_path


def decrypt_file_to_bytes(encrypted_path: Path, key: bytes | None = None) -> bytes:
    return decrypt_bytes(Path(encrypted_path).read_bytes(), key=key)


@contextmanager
def decrypt_file_to_temp(encrypted_path: Path, suffix: str = "") -> Iterator[Path]:
    plain_data = decrypt_file_to_bytes(encrypted_path)
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = Path(tmp_file.name)
    try:
        with tmp_file:
            tmp_file.write(plain_data)
        yield tmp_path
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="AES-256-GCM key yardımcı aracı")
    parser.add_argument("--generate-key", action="store_true", help="Yeni base64 AES-256 key üret")
    args = parser.parse_args()

    if args.generate_key:
        print(generate_key_b64())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
