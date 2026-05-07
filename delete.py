import argparse
import sqlite3

from config import DB_NAME
from database import delete_session


def delete_recent_messages(session_id: str, limit: int, confirm: bool) -> int:
    if not confirm:
        raise SystemExit("Refusing to delete without --yes")

    conn = sqlite3.connect(DB_NAME, timeout=30.0)
    cursor = conn.cursor()
    cursor.execute(
        """
        DELETE FROM messages
        WHERE id IN (
            SELECT id FROM messages
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
        )
        """,
        (session_id, limit),
    )
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    return deleted


def main():
    parser = argparse.ArgumentParser(
        description="Guarded cleanup helper for a session."
    )
    parser.add_argument("session_id")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument(
        "--delete-session",
        action="store_true",
        help="Delete ALL session state (messages, summaries, usage, memory).",
    )
    args = parser.parse_args()

    if args.delete_session:
        if not args.yes:
            raise SystemExit("Refusing to delete session without --yes")
        deleted = delete_session(args.session_id)
        print(f"Deleted session {args.session_id} ({deleted} messages)")
        return

    deleted = delete_recent_messages(args.session_id, args.limit, args.yes)
    print(f"Deleted {deleted} recent rows from session {args.session_id}")


if __name__ == "__main__":
    main()
