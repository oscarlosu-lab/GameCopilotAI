"""Unit tests for GameCopilot AI core logic."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server"))

from rag_engine import GameRAG, chunk_text, clean_html_to_text, slugify  # noqa: E402
from server import detect_switch_game  # noqa: E402


class TestCopilotCore(unittest.TestCase):
    def test_slugify(self) -> None:
        self.assertEqual(slugify("Baldur's Gate 3!"), "baldur-s-gate-3")
        self.assertEqual(slugify("Aniimo"), "aniimo")
        self.assertEqual(slugify("   "), "default-game")

    def test_clean_html_and_chunk(self) -> None:
        html = "<html><head><title>Aniimo Guide</title><style>.x{}</style></head><body><nav>Menu</nav><p>This is a detailed guide paragraph explaining elemental weaknesses and creature capture mechanics in Aniimo.</p></body></html>"
        title, text = clean_html_to_text(html)
        self.assertEqual(title, "Aniimo Guide")
        self.assertIn("elemental weaknesses", text)
        self.assertNotIn("Menu", text)
        chunks = chunk_text(text * 10, chunk_size=300, overlap=50)
        self.assertGreater(len(chunks), 1)

    def test_rag_ingest_and_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rag = GameRAG(Path(tmp) / "data", Path(tmp) / "guides")
            fake_vec = np.ones(8, dtype=np.float32)
            with patch.object(rag, "_embed_batch", side_effect=lambda texts: [fake_vec for _ in texts]):
                added = rag.ingest_text(
                    "Aniimo",
                    "https://example.test/guide",
                    "Starter Guide",
                    "Helion is a fire-type starter Aniimo strong against nature creatures.\n" * 3,
                )
                self.assertGreater(added, 0)
                hits = rag.search("Aniimo", "fire starter Helion", top_k=2)
                self.assertEqual(len(hits), 1)
                self.assertIn("Helion", hits[0]["content"])

    def test_detect_switch_game(self) -> None:
        self.assertEqual(detect_switch_game("vamos a jugar a Elden Ring"), "elden ring")
        self.assertEqual(detect_switch_game("aprende la guía de Aniimo"), "aniimo")
        self.assertIsNone(detect_switch_game("donde esta el jefe final"))


if __name__ == "__main__":
    unittest.main()

