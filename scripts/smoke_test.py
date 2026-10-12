from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ui.chatbot import build_runtime, handle_message


def main() -> int:
    query = " ".join(sys.argv[1:]).strip() or "Find me a cozy Japanese restaurant"
    try:
        workflow = build_runtime()
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(exc)
        return 0
    print(handle_message(query, workflow))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
