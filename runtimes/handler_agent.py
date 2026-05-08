import os
import re
import logging
from pathlib import Path
from functools import lru_cache
from typing import Iterable, List, Optional
import torch
from torch.utils.data import Dataset

from runtimes.configure_agent import DATA_DIR, FILE_PATTERNS, BLOCK_SIZE

logger = logging.getLogger(__name__)

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

# Mapping of ambiguous extensions to their primary type for consistent handling
AMBIGUOUS_EXTENSIONS = {
    ".ts": "typescript",  # Could be TypeScript or TSV-like
    ".log": "text",
    ".csv": "delimited",
    ".tsv": "delimited",
    ".yaml": "yaml",
    ".yml": "yaml",
}

@lru_cache(maxsize=None)
def _compiled_patterns(patterns_key: tuple[str, ...]):
    return tuple(re.compile(p, re.IGNORECASE) for p in patterns_key)


def _normalize_path(path_str: str) -> Path:
    """Normalize a path string, handling spaces and special characters gracefully.
    
    Args:
        path_str: Raw path string that may contain spaces or need normalization
        
    Returns:
        Normalized Path object
    """
    # Strip leading/trailing whitespace
    path_str = path_str.strip()
    
    # Handle quoted paths (remove surrounding quotes if present)
    if (path_str.startswith('"') and path_str.endswith('"')) or \
       (path_str.startswith("'") and path_str.endswith("'")):
        path_str = path_str[1:-1]
    
    # Expand user home directory (~)
    if path_str.startswith("~"):
        path_str = Path(path_str).expanduser().as_posix()
    
    return Path(path_str)


def _get_file_type(file_path: Path) -> str:
    """Determine file type considering ambiguous extensions.
    
    Handles cases where file extensions might be ambiguous or misleading.
    Uses both extension and content inspection when needed.
    
    Args:
        file_path: Path to the file
        
    Returns:
        String identifier for file type
    """
    suffix = file_path.suffix.lower()
    name_lower = file_path.name.lower()
    
    # Check for compound extensions (e.g., .tar.gz, .test.py)
    stem = file_path.stem.lower()
    
    # Handle ambiguous extensions by checking file content or name patterns
    if suffix == ".ts":
        # Distinguish TypeScript from TSV by checking content or name hints
        if "tab" in name_lower or "tsv" in name_lower:
            return "delimited"
        return "typescript"
    elif suffix in {".yaml", ".yml"}:
        return "yaml"
    elif suffix in {".csv", ".tsv"}:
        return "delimited"
    elif suffix == ".log":
        return "text"
    elif suffix == ".pdf":
        return "pdf"
    else:
        return "text"


def _extract_pdf_text(file_path: Path) -> Optional[str]:
    """Extract text from PDF files with robust error handling.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text or None if extraction fails
    """
    if not HAS_PDFPLUMBER:
        logger.warning(f"pdfplumber not installed, cannot extract text from {file_path}")
        return None
    
    if not file_path.exists():
        logger.warning(f"PDF file does not exist: {file_path}")
        return None
    
    if not file_path.is_file():
        logger.warning(f"Path is not a file: {file_path}")
        return None
    
    try:
        with pdfplumber.open(file_path) as pdf:
            text_parts = []
            for i, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {i+1} in {file_path}: {e}")
                    continue
            
            if not text_parts:
                logger.debug(f"No text extracted from {file_path}")
                return ""
            
            return "\n".join(text_parts)
            
    except PermissionError:
        logger.error(f"Permission denied reading PDF: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error extracting text from PDF {file_path}: {e}")
        return None


def _extract_text(file_path: Path) -> Optional[str]:
    """Extract text from various file formats with robust error handling.
    
    Handles ambiguous file extensions and gracefully manages spaces in paths.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Extracted text or None if extraction fails
    """
    if not file_path.exists():
        logger.warning(f"File does not exist: {file_path}")
        return None
    
    if not file_path.is_file():
        logger.warning(f"Path is not a file: {file_path}")
        return None
    
    # Check read permissions
    try:
        if not os.access(file_path, os.R_OK):
            logger.error(f"No read permission for file: {file_path}")
            return None
    except Exception:
        pass  # Continue anyway, let the read attempt fail naturally
    
    file_type = _get_file_type(file_path)
    
    if file_type == "pdf":
        return _extract_pdf_text(file_path)
    
    # For text-based files, try multiple encodings
    encodings_to_try = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    
    for encoding in encodings_to_try:
        try:
            content = file_path.read_text(encoding=encoding, errors="strict")
            # Verify we got some readable content
            if content.strip():
                return content
            # Empty file is still valid, just return empty string
            return content
        except UnicodeDecodeError:
            continue
        except PermissionError:
            logger.error(f"Permission denied reading file: {file_path}")
            return None
        except Exception as e:
            logger.warning(f"Error reading file {file_path} with {encoding}: {e}")
            continue
    
    # Last resort: read with errors ignored
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        logger.debug(f"Read {file_path} with UTF-8 ignoring errors")
        return content
    except Exception as e:
        logger.error(f"Failed to read file {file_path} after all attempts: {e}")
        return None

def match_any_pattern(path: Path, patterns: List[str]) -> bool:
    """Check if a path matches any of the given regex patterns.
    
    Handles paths with spaces by using the full path string representation.
    
    Args:
        path: Path object to check
        patterns: List of regex pattern strings
        
    Returns:
        True if path matches any pattern, False otherwise
    """
    # Use both absolute and relative path representations for matching
    path_str = str(path)
    abs_path_str = str(path.absolute()) if path.is_absolute() else str(path.absolute())
    
    compiled_patterns = _compiled_patterns(tuple(patterns))
    return any(pattern.match(path_str) or pattern.match(abs_path_str) 
               for pattern in compiled_patterns)


@lru_cache(maxsize=1)
def load_corpus() -> str:
    """Load and aggregate text from all matching files in DATA_DIR.
    
    Gracefully handles:
    - Missing DATA_DIR
    - Files with spaces in names
    - Ambiguous file extensions
    - Permission errors
    - Corrupted or unreadable files
    
    Returns:
        Concatenated text from all successfully processed files
    """
    if not DATA_DIR:
        logger.warning("DATA_DIR is not configured")
        return ""
    
    if not DATA_DIR.exists():
        logger.warning(f"DATA_DIR does not exist: {DATA_DIR}")
        return ""
    
    if not DATA_DIR.is_dir():
        logger.error(f"DATA_DIR is not a directory: {DATA_DIR}")
        return ""
    
    texts = []
    processed_count = 0
    error_count = 0
    
    try:
        all_files = sorted(DATA_DIR.rglob("*"))
    except PermissionError as e:
        logger.error(f"Permission denied accessing DATA_DIR {DATA_DIR}: {e}")
        return ""
    except Exception as e:
        logger.error(f"Error traversing DATA_DIR {DATA_DIR}: {e}")
        return ""
    
    for p in all_files:
        try:
            if not p.is_file():
                continue
                
            if not match_any_pattern(p, FILE_PATTERNS):
                continue
            
            text = _extract_text(p)
            
            if text is None:
                logger.warning(f"Skipping file due to extraction failure: {p}")
                error_count += 1
                continue
            
            if text.strip():
                texts.append(text)
                processed_count += 1
                logger.debug(f"Successfully processed: {p}")
            else:
                logger.debug(f"File is empty or whitespace only: {p}")
                
        except Exception as e:
            logger.error(f"Unexpected error processing file {p}: {e}")
            error_count += 1
            continue
    
    if processed_count == 0:
        logger.warning(f"No valid text extracted from {len(all_files)} files in {DATA_DIR}")
    elif error_count > 0:
        logger.info(f"Processed {processed_count} files with {error_count} errors")
    
    if not texts:
        return ""
    
    result = "\n\n".join(texts)
    logger.info(f"Loaded corpus: {len(result)} characters from {processed_count} files")
    return result


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