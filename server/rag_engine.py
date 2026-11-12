"""Per-game RAG engine with automatic web guide discovery and hybrid search."""
import hashlib
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from bs4 import BeautifulSoup
from ddgs import DDGS
from pypdf import PdfReader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
HTTP_TIMEOUT = 15.0
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


def slugify(name: str) -> str:
    """Normalize a game name into a filesystem and DB safe slug."""
    cleaned = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return cleaned or "default-game"


def clean_html_to_text(html: str) -> tuple[str, str]:
    """Extract readable title and body text from raw HTML."""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "svg"]):
        tag.decompose()
    lines = [ln.strip() for ln in soup.get_text("\n").splitlines()]
    filtered = [ln for ln in lines if len(ln) > 25]
    return title, "\n".join(filtered)


def chunk_text(text: str, chunk_size: int = 850, overlap: int = 120) -> list[str]:
    """Split text into overlapping chunks preferring line boundaries."""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            nl = text.rfind("\n", start + chunk_size // 2, end)
            if nl > start:
                end = nl
        piece = text[start:end].strip()
        if len(piece) >= 60:
            chunks.append(piece)
        if end >= length:
            break
        start = max(end - overlap, start + 1)
    return chunks


class GameRAG:
    """SQLite + Ollama embedding store for per-game guides."""

    def __init__(self, data_dir: Path, guides_dir: Path) -> None:
        self.data_dir = data_dir
        self.guides_dir = guides_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.guides_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "guides.db"
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS games (
                    slug TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS sources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_slug TEXT NOT NULL,
                    url_or_path TEXT NOT NULL,
                    title TEXT,
                    content_hash TEXT NOT NULL,
                    chunk_count INTEGER DEFAULT 0,
                    added_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(game_slug, url_or_path)
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_slug TEXT NOT NULL,
                    source_id INTEGER NOT NULL,
                    title TEXT,
                    content TEXT NOT NULL,
                    embedding BLOB NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_game ON chunks(game_slug);
                """
            )

    def _embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []
        resp = httpx.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": texts},
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return [np.asarray(vec, dtype=np.float32) for vec in data["embeddings"]]

    def ensure_game(self, game_name: str) -> str:
        """Register game if missing and return its slug."""
        slug = slugify(game_name)
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO games(slug, name) VALUES (?, ?)",
                (slug, game_name.strip()),
            )
        (self.guides_dir / slug).mkdir(parents=True, exist_ok=True)
        return slug

    def list_games(self) -> list[dict[str, Any]]:
        """List all indexed games with source and chunk counts."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT g.slug, g.name,
                       COUNT(DISTINCT s.id) AS sources,
                       COUNT(c.id) AS chunks
                FROM games g
                LEFT JOIN sources s ON s.game_slug = g.slug
                LEFT JOIN chunks c ON c.game_slug = g.slug
                GROUP BY g.slug, g.name
                ORDER BY g.created_at DESC
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def ingest_text(self, game_name: str, origin: str, title: str, text: str) -> int:
        """Chunk, embed, and store a document if content changed."""
        slug = self.ensure_game(game_name)
        content_hash = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
        pieces = chunk_text(text)
        if not pieces:
            return 0

        with self._connect() as conn:
            existing = conn.execute(
                "SELECT id, content_hash FROM sources WHERE game_slug=? AND url_or_path=?",
                (slug, origin),
            ).fetchone()
            if existing and existing["content_hash"] == content_hash:
                return 0
            if existing:
                conn.execute("DELETE FROM chunks WHERE source_id=?", (existing["id"],))
                conn.execute("DELETE FROM sources WHERE id=?", (existing["id"],))

        vectors: list[np.ndarray] = []
        batch_size = 16
        for i in range(0, len(pieces), batch_size):
            vectors.extend(self._embed_batch(pieces[i : i + batch_size]))

        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO sources(game_slug, url_or_path, title, content_hash, chunk_count)
                VALUES (?, ?, ?, ?, ?)
                """,
                (slug, origin, title[:200], content_hash, len(pieces)),
            )
            source_id = cur.lastrowid
            conn.executemany(
                "INSERT INTO chunks(game_slug, source_id, title, content, embedding) VALUES (?, ?, ?, ?, ?)",
                [
                    (slug, source_id, title[:200], piece, vec.tobytes())
                    for piece, vec in zip(pieces, vectors)
                ],
            )
        return len(pieces)

    def ingest_local_folder(self, game_name: str) -> dict[str, int]:
        """Ingest all .txt, .md, and .pdf files in the game's guides directory."""
        slug = self.ensure_game(game_name)
        folder = self.guides_dir / slug
        files_done = 0
        chunks_added = 0
        for file_path in sorted(folder.iterdir()):
            if not file_path.is_file():
                continue
            ext = file_path.suffix.lower()
            try:
                if ext in {".txt", ".md"}:
                    text = file_path.read_text(encoding="utf-8", errors="ignore")
                elif ext == ".pdf":
                    reader = PdfReader(str(file_path))
                    text = "\n".join(page.extract_text() or "" for page in reader.pages)
                else:
                    continue
                added = self.ingest_text(game_name, f"file://{file_path.name}", file_path.stem, text)
                if added > 0:
                    files_done += 1
                    chunks_added += added
            except Exception as exc:
                print(f"[WARN] Failed reading {file_path}: {exc}", file=sys.stderr)
        return {"files": files_done, "chunks": chunks_added}

    def auto_learn_game(self, game_name: str, max_pages: int = 10) -> dict[str, Any]:
        """Search web guides/wikis for a game, scrape them, and index into RAG."""
        self.ensure_game(game_name)
        queries = [
            f"{game_name} game wiki guide",
            f"{game_name} beginner guide tips walkthrough",
            f"{game_name} best build tier list mechanics",
        ]
        urls: list[tuple[str, str]] = []
        seen: set[str] = set()

        for query in queries:
            results: list[dict[str, Any]] = []
            for backend in ("auto", "bing", "yahoo"):
                try:
                    results = list(DDGS().text(query, max_results=6, backend=backend))
                    if results:
                        break
                except Exception as exc:
                    print(f"[WARN] Search error ({backend}) for '{query}': {exc}", file=sys.stderr)
            for item in results:
                href = item.get("href") or item.get("url") or ""
                title = item.get("title") or game_name
                if not href.startswith("http") or href in seen:
                    continue
                if any(bad in href.lower() for bad in ("youtube.com", "youtu.be", "tiktok.com", "instagram.com")):
                    continue
                seen.add(href)
                urls.append((href, title))

        scraped = 0
        total_chunks = 0
        errors: list[str] = []
        headers = {"User-Agent": USER_AGENT, "Accept-Language": "es-ES,es;q=0.9,en;q=0.8"}

        with httpx.Client(timeout=HTTP_TIMEOUT, follow_redirects=True, headers=headers) as client:
            for url, fallback_title in urls[:max_pages]:
                try:
                    resp = client.get(url)
                    if resp.status_code >= 400 or "text/html" not in resp.headers.get("content-type", ""):
                        continue
                    title, text = clean_html_to_text(resp.text)
                    if len(text) < 300:
                        continue
                    added = self.ingest_text(game_name, url, title or fallback_title, text)
                    scraped += 1
                    total_chunks += added
                except Exception as exc:
                    errors.append(f"{url}: {exc}")

        local_stats = self.ingest_local_folder(game_name)
        return {
            "game": game_name,
            "slug": slugify(game_name),
            "urls_found": len(urls),
            "pages_scraped": scraped,
            "chunks_indexed": total_chunks + local_stats["chunks"],
            "errors": errors[:5],
        }

    def search(self, game_name: str, query: str, top_k: int = 4) -> list[dict[str, Any]]:
        """Retrieve top_k chunks for a game using cosine similarity + keyword boost."""
        slug = slugify(game_name)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, title, content FROM chunks WHERE game_slug=?",
                (slug,),
            ).fetchall()
            if not rows:
                return []
            emb_rows = conn.execute(
                "SELECT id, embedding FROM chunks WHERE game_slug=?",
                (slug,),
            ).fetchall()

        q_vec = self._embed_batch([query])[0]
        q_norm = np.linalg.norm(q_vec) + 1e-9
        terms = [t for t in re.findall(r"[a-z0-9áéíóúñ]{3,}", query.lower())]

        scored: list[tuple[float, dict[str, Any]]] = []
        for row, erow in zip(rows, emb_rows):
            vec = np.frombuffer(erow["embedding"], dtype=np.float32)
            sim = float(np.dot(q_vec, vec) / (q_norm * (np.linalg.norm(vec) + 1e-9)))
            low = row["content"].lower()
            kw_bonus = sum(0.03 for t in terms if t in low)
            scored.append((sim + kw_bonus, {"title": row["title"], "content": row["content"]}))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]
