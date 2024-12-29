"""
Generate a gateway API key and print the value + hash to add to tenants.yaml.

Usage:
    uv run python scripts/generate_key.py
"""
import hashlib
import secrets


def generate_key() -> tuple[str, str]:
    raw = "sk-gateway-" + secrets.token_urlsafe(32)
    key_hash = hashlib.sha256(raw.encode()).hexdigest()
    return raw, key_hash


if __name__ == "__main__":
    raw, key_hash = generate_key()
    print(f"API key  : {raw}")
    print(f"SHA-256  : {key_hash}")
    print()
    print("Add to config/tenants.yaml:")
    print(f"  key_hash: {key_hash!r}")
    print()
    print("Use in requests:")
    print(f"  Authorization: Bearer {raw}")
