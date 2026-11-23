"""Digests estáveis de dataset e de configuração.

O manifesto repete o mesmo hash quando o conteúdo não muda. Ordem de chave e espaço
fora da string não entram. Segredo não é serializado aqui: só o que o chamador passou.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(payload: Any) -> str:
    """JSON compacto, chaves ordenadas, UTF-8 lógico sem escapar acento."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(payload: Any) -> str:
    """SHA-256 hex do JSON canônico."""
    text = canonical_json(payload)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
