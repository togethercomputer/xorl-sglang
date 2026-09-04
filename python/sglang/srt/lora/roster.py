"""Stable compact identities for scheduler-to-LoRA roster transport."""

from __future__ import annotations

import hashlib
from typing import Optional


# Keep graph metadata in a fixed-capacity row; larger local rosters use the
# ordinary object-gather fallback.
LORA_ROW_FINGERPRINT_CAPACITY = 32


def lora_uid_fingerprint(uid: Optional[str]) -> int:
    """Return a stable signed int64 identity; zero is reserved for base rows."""

    if uid is None:
        return 0
    value = str(uid).encode("utf-8")
    digest = hashlib.blake2b(
        value,
        digest_size=8,
        person=b"sgl-lora-row-v1",
    ).digest()
    fingerprint = int.from_bytes(digest, "little", signed=True)
    # Keep the base-model sentinel unique in the overwhelmingly unlikely case
    # of a zero digest. The LoRA manager still rejects any resulting collision.
    return 1 if fingerprint == 0 else fingerprint
