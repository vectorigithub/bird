import re
from pathlib import Path
from functools import lru_cache
from typing import Iterable, List
import torch
from torch.utils.data import Dataset

from runtimes.configure_agent import DATA_DIR, FILE_PATTERNS, BLOCK_SIZE

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

@lru_cache(maxsize=None)
def _compiled_patterns(patterns_key: tuple[str, ...]):
    return tuple(re.compile(p) for p in patterns_key)

def _extract_pdf_text(file_path: Path) -> str:
    """Extract text from PDF files."""
    if not HAS_PDFPLUMBER:
        return ""
    
    try:
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
    except Exception:
        return ""

def _extract_text(file_path: Path) -> str:
    """Extract text from various file formats."""
    suffix = file_path.suffix.lower()
    
    if suffix == ".pdf":
        return _extract_pdf_text(file_path)
    else:
        # Default text extraction for .txt, .py, .js, .json, .md, .csv, etc.
        try:
            return file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""

def match_any_pattern(path: Path, patterns: List[str]) -> bool:
    s = str(path)
    compiled_patterns = _compiled_patterns(tuple(patterns))
    return any(pattern.match(s) for pattern in compiled_patterns)

@lru_cache(maxsize=1)
def load_corpus() -> str:
    if not DATA_DIR.exists():
        return ""

    texts = []
    for p in sorted(DATA_DIR.rglob("*")):
        if p.is_file() and match_any_pattern(p, FILE_PATTERNS):
            text = _extract_text(p)
            if text.strip():
                texts.append(text)
    return "\n\n".join(texts)


# Commented-out helper to run the JavaScript web search leechers.
# This is intentionally disabled by default. To enable, remove the surrounding
# comment block below and ensure Node.js is installed. The leechers script
# `runtimes/leechers/web_search.js` uses the DuckDuckGo Instant Answer API.
#
# Example usage (uncomment to enable):
#
# import subprocess, json
#
# def _run_web_search(query: str, timeout: int = 10) -> list:
#     """Run the Node.js web_search.js helper and return parsed JSON results.
#
#     Requires Node.js. This helper is commented out by default to avoid
#     accidental network requests during training.
#     """
#     try:
#         proc = subprocess.run([
#             "node",
#             str(Path(__file__).resolve().parent / "leechers" / "web_search.js"),
#             query,
#         ], capture_output=True, text=True, timeout=timeout)
#     except Exception:
#         return []
#
#     if proc.returncode != 0:
#         return []
#
#     try:
#         return json.loads(proc.stdout)
#     except Exception:
#         return []


class CharVocab:
    def __init__(self, characters: str | Iterable[str]):
        if isinstance(characters, str):
            chars = sorted(set(characters))
        else:
            chars = list(dict.fromkeys(characters))

        if not chars:
            raise ValueError("CharVocab requires at least one character")

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @classmethod
    def from_stoi(cls, stoi: dict[str, int]):
        vocab = cls(stoi.keys())
        vocab.stoi = dict(stoi)
        vocab.itos = {i: ch for ch, i in stoi.items()}
        return vocab

    def encode(self, s: str):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

class TextDataset(Dataset):
    def __init__(self, text: str, vocab: CharVocab, block_size: int = BLOCK_SIZE):
        self.vocab = vocab
        self.block_size = block_size
        data = torch.tensor(vocab.encode(text), dtype=torch.long)
        if len(data) <= block_size:
            raise ValueError("Corpus is too small for the configured block size")
        self.data = data

    def __len__(self):
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + 1 + self.block_size]
        return x, y