"""Split a streaming LLM reply into speakable chunks.

Speech starts as soon as the first sentence is complete instead of waiting for
the whole reply. Chunks are kept within a size window that TTS models handle
well (very short fragments sound choppy, very long ones hurt latency and can
exceed model limits).
"""

from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"""([.!?…。！？]+["')\]]*)(\s+|$)|(\n\s*\n|\n)""")
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e",
    "approx", "dept", "est", "fig", "inc", "ltd", "no", "vol", "sra", "srta", "ud", "uds",
}

_MD_PATTERNS = [
    (re.compile(r"```.*?```", re.S), " "),          # fenced code
    (re.compile(r"`([^`]*)`"), r"\1"),               # inline code
    (re.compile(r"!\[([^\]]*)\]\([^)]*\)"), r"\1"),  # images
    (re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"),   # links -> text
    (re.compile(r"https?://\S+"), "a link"),
    (re.compile(r"^\s{0,3}#{1,6}\s*", re.M), ""),    # headings
    (re.compile(r"^\s*[-*+•]\s+", re.M), ""),        # bullets
    (re.compile(r"^\s*\d+[.)]\s+", re.M), ""),       # numbered lists
    (re.compile(r"(\*\*|__|\*|_|~~)(?=\S)(.+?)(?<=\S)\1"), r"\2"),  # emphasis
    (re.compile(r"^\s*>\s?", re.M), ""),             # quotes
    (re.compile(r"\|"), " "),                        # tables
]
_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF\U0001F900-\U0001F9FF️‍]+"
)


def clean_for_speech(text: str) -> str:
    for pat, repl in _MD_PATTERNS:
        text = pat.sub(repl, text)
    text = _EMOJI.sub("", text)
    text = text.replace("*", "").replace("#", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _ends_with_abbreviation(text: str) -> bool:
    m = re.search(r"(\b[\w.]+)\.$", text.rstrip("\"')] "))
    if not m:
        return False
    word = m.group(1).lower()
    return word in _ABBREVIATIONS or (len(word) == 1 and word.isalpha())


class SentenceChunker:
    def __init__(self, first_min_chars: int = 20, min_chars: int = 60, max_chars: int = 280):
        self.first_min_chars = first_min_chars
        self.min_chars = min_chars
        self.max_chars = max_chars
        self._buf = ""
        self._pending = ""  # complete sentences waiting to reach min length
        self._emitted = 0

    def _min(self) -> int:
        return self.first_min_chars if self._emitted == 0 else self.min_chars

    def feed(self, delta: str) -> list[str]:
        self._buf += delta
        out: list[str] = []
        while True:
            m = self._find_boundary(self._buf)
            if m is None:
                break
            sentence, self._buf = self._buf[:m], self._buf[m:]
            self._pending = (self._pending + " " + sentence).strip() if self._pending else sentence.strip()
            if len(self._pending) >= self._min():
                out.extend(self._emit(self._pending))
                self._pending = ""
        # A very long run without punctuation: split at a soft boundary.
        if len(self._pending) + len(self._buf) > self.max_chars * 1.5:
            combined = (self._pending + " " + self._buf).strip()
            cut = self._soft_cut(combined)
            out.extend(self._emit(combined[:cut]))
            self._pending, self._buf = "", combined[cut:]
        return [c for c in out if c]

    def flush(self) -> list[str]:
        rest = (self._pending + " " + self._buf).strip()
        self._pending, self._buf = "", ""
        return [c for c in self._emit(rest) if c] if rest else []

    def _find_boundary(self, text: str) -> int | None:
        for m in _SENTENCE_END.finditer(text):
            end = m.end()
            if m.group(3) is None:  # punctuation boundary
                if m.group(2) == "" and m.end() == len(text):
                    return None  # might be mid-token (e.g. "3." of "3.5"); wait for more
                if _ends_with_abbreviation(text[: m.end(1)]):
                    continue
            return end
        return None

    def _soft_cut(self, text: str) -> int:
        window = text[: self.max_chars]
        for sep in (", ", "; ", ": ", " - ", " "):
            i = window.rfind(sep)
            if i > self.max_chars // 3:
                return i + len(sep)
        return len(window)

    def _emit(self, text: str) -> list[str]:
        cleaned = clean_for_speech(text)
        if not cleaned or not re.search(r"\w", cleaned):
            return []
        parts = []
        while len(cleaned) > self.max_chars:
            cut = self._soft_cut(cleaned)
            parts.append(cleaned[:cut].strip())
            cleaned = cleaned[cut:].strip()
        if cleaned:
            parts.append(cleaned)
        self._emitted += len(parts)
        return parts


def split_text(text: str, **kw) -> list[str]:
    """Chunk a complete text (for the 'speak this' box)."""
    c = SentenceChunker(**kw)
    return c.feed(text) + c.flush()
