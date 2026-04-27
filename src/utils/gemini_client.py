"""
Shared Gemini 2.5 Flash client for EVE-RAG.

Usage from a notebook:
    from src.utils.gemini_client import GeminiClient
    gc = GeminiClient()
    text = gc.generate(prompt, image_path=Path("x.jpg"))

Features:
- Disk-backed cache (`cache/mllm/gemini_responses/`) keyed by hash(prompt + image_bytes + model).
- Soft rate limit: at most N_RPM requests per minute (default 10).
- Retries on transient errors (429, 500, network).
- Returns plain text; raises only on fatal config / auth failures.
"""
from __future__ import annotations
import os, json, time, hashlib
from pathlib import Path
from dataclasses import dataclass

import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv


@dataclass
class GeminiClient:
    model_name: str = "models/gemini-2.5-flash"
    rpm_limit: int = 10
    cache_dir: Path = None
    image_max_side: int = 512
    max_retries: int = 4
    request_log: list = None

    def __post_init__(self):
        # Locate repo root
        here = Path(__file__).resolve()
        for p in [here] + list(here.parents):
            if (p / "src" / "utils" / "paths.py").exists():
                self.repo_root = p
                break
        else:
            self.repo_root = Path.cwd()
        # API key
        load_dotenv(self.repo_root / ".env", override=True)
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not in environment. Put it in .env at repo root.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.model_name)
        # Cache
        if self.cache_dir is None:
            self.cache_dir = self.repo_root / "cache" / "mllm" / "gemini_responses"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Rate limit state
        self.request_log = []

    # --- cache key ---
    @staticmethod
    def _key(prompt: str, image_path: Path | None, model_name: str) -> str:
        h = hashlib.sha256()
        h.update(model_name.encode())
        h.update(b"||")
        h.update(prompt.encode("utf-8"))
        h.update(b"||")
        if image_path is not None and Path(image_path).exists():
            h.update(Path(image_path).read_bytes())
        return h.hexdigest()[:32]

    # --- soft rate limit ---
    def _wait_if_needed(self):
        now = time.time()
        self.request_log = [t for t in self.request_log if now - t < 60]
        if len(self.request_log) >= self.rpm_limit:
            sleep_for = 60 - (now - self.request_log[0]) + 0.1
            if sleep_for > 0:
                time.sleep(sleep_for)
                self.request_log = [t for t in self.request_log if time.time() - t < 60]

    def generate(self, prompt: str, image_path: Path | None = None,
                 use_cache: bool = True, temperature: float = 0.2,
                 max_output_tokens: int = 512) -> str:
        """Return Gemini's text response. Caches by hash of (prompt, image, model)."""
        key = self._key(prompt, image_path, self.model_name)
        cache_file = self.cache_dir / f"{key}.json"

        if use_cache and cache_file.exists():
            try:
                d = json.loads(cache_file.read_text(encoding="utf-8"))
                return d["text"]
            except Exception:
                pass

        self._wait_if_needed()
        parts: list = [prompt]
        if image_path is not None:
            img = Image.open(image_path).convert("RGB")
            img.thumbnail((self.image_max_side, self.image_max_side))
            parts.append(img)

        gen_cfg = {"temperature": temperature, "max_output_tokens": max_output_tokens}
        last_err = None
        for attempt in range(self.max_retries):
            try:
                self.request_log.append(time.time())
                resp = self.model.generate_content(parts, generation_config=gen_cfg)
                text = (resp.text or "").strip()
                cache_file.write_text(json.dumps({
                    "text": text, "prompt": prompt[:2000],
                    "image": str(image_path) if image_path else None,
                    "model": self.model_name, "ts": time.time(),
                }), encoding="utf-8")
                return text
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                if any(k in msg for k in ("429", "quota", "rate", "deadline", "503", "500", "unavailable")):
                    backoff = min(60.0, 2 ** attempt * 5)
                    time.sleep(backoff)
                    continue
                raise
        raise RuntimeError(f"Gemini call failed after {self.max_retries} retries: {last_err}")
