"""
Gemini 2.5 Flash-Lite client for EVE-RAG.

Used by:
    notebooks/06_mllm/mllm_inference.ipynb
    notebooks/06_mllm/chain_of_thought.ipynb

Features:
- Disk cache keyed by hash(prompt + image bytes + model). Free re-runs.
- Soft RPM rate limit (default 14, headroom under 15 RPM Flash-Lite limit).
- Daily request counter so we stop cleanly before hitting 1000 RPD.
- Retries on 429 / 500 / 503 with exponential backoff.
- Falls back from Flash-Lite to Flash (if you set fallback_model).
"""
from __future__ import annotations
import os, json, time, hashlib
from pathlib import Path
from dataclasses import dataclass, field
from datetime import date

import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv


@dataclass
class MLLMClient:
    model_name: str = "models/gemini-2.5-flash-lite"
    fallback_model: str | None = "models/gemini-2.5-flash"
    rpm_limit: int = 14            # headroom under 15 RPM
    rpd_limit: int = 950           # headroom under 1000 RPD
    cache_dir: Path | None = None
    image_max_side: int = 512
    max_retries: int = 5
    request_log: list = field(default_factory=list)

    def __post_init__(self):
        here = Path(__file__).resolve()
        for p in [here] + list(here.parents):
            if (p / "src" / "utils" / "paths.py").exists():
                self.repo_root = p; break
        else:
            self.repo_root = Path.cwd()

        load_dotenv(self.repo_root / ".env", override=True)
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not in .env (repo root).")
        genai.configure(api_key=api_key)

        self._primary = genai.GenerativeModel(self.model_name)
        self._fallback = genai.GenerativeModel(self.fallback_model) if self.fallback_model else None

        if self.cache_dir is None:
            self.cache_dir = self.repo_root / "cache" / "mllm" / "gemini_responses"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Daily counter: tiny json file keyed by today's date
        self._counter_path = self.repo_root / "cache" / "mllm" / "daily_counter.json"
        self._load_counter()

    # ------- counter -------
    def _load_counter(self):
        today = str(date.today())
        if self._counter_path.exists():
            try:
                d = json.loads(self._counter_path.read_text())
                if d.get("date") == today:
                    self._daily_count = int(d.get("count", 0))
                    return
            except Exception:
                pass
        self._daily_count = 0
        self._save_counter()

    def _save_counter(self):
        self._counter_path.write_text(json.dumps({
            "date": str(date.today()),
            "count": self._daily_count,
        }))

    def _bump_counter(self):
        self._daily_count += 1
        self._save_counter()

    def daily_remaining(self) -> int:
        return max(0, self.rpd_limit - self._daily_count)

    # ------- cache key -------
    @staticmethod
    def _key(prompt: str, image_path, model_name: str) -> str:
        h = hashlib.sha256()
        h.update(model_name.encode()); h.update(b"||")
        h.update(prompt.encode("utf-8")); h.update(b"||")
        if image_path is not None and Path(image_path).exists():
            h.update(Path(image_path).read_bytes())
        return h.hexdigest()[:32]

    # ------- rate limit -------
    def _wait_if_needed(self):
        now = time.time()
        self.request_log = [t for t in self.request_log if now - t < 60]
        if len(self.request_log) >= self.rpm_limit:
            sleep_for = 60 - (now - self.request_log[0]) + 0.5
            if sleep_for > 0:
                time.sleep(sleep_for)
                self.request_log = [t for t in self.request_log if time.time() - t < 60]

    # ------- main API -------
    def generate(self, prompt: str, image_path=None,
                 use_cache: bool = True, temperature: float = 0.2,
                 max_output_tokens: int = 512) -> str:
        """Return Gemini text response. Cached, rate-limited, restartable."""
        key = self._key(prompt, image_path, self.model_name)
        cache_file = self.cache_dir / f"{key}.json"

        if use_cache and cache_file.exists():
            try:
                return json.loads(cache_file.read_text(encoding="utf-8"))["text"]
            except Exception:
                pass

        if self._daily_count >= self.rpd_limit:
            raise RuntimeError(
                f"Daily quota of {self.rpd_limit} reached. Resume tomorrow "
                f"(quota resets at midnight Pacific). Cache is preserved."
            )

        self._wait_if_needed()
        parts = [prompt]
        if image_path is not None:
            img = Image.open(image_path).convert("RGB")
            img.thumbnail((self.image_max_side, self.image_max_side))
            parts.append(img)

        gen_cfg = {"temperature": temperature, "max_output_tokens": max_output_tokens}

        last_err = None
        models_to_try = [self._primary] + ([self._fallback] if self._fallback else [])
        for mdl in models_to_try:
            for attempt in range(self.max_retries):
                try:
                    self.request_log.append(time.time())
                    resp = mdl.generate_content(parts, generation_config=gen_cfg)
                    text = (resp.text or "").strip()
                    cache_file.write_text(json.dumps({
                        "text": text,
                        "prompt": prompt[:2000],
                        "image": str(image_path) if image_path else None,
                        "model": mdl.model_name,
                        "ts": time.time(),
                    }), encoding="utf-8")
                    self._bump_counter()
                    return text
                except Exception as e:
                    last_err = e
                    msg = str(e).lower()
                    transient = any(k in msg for k in
                        ("429", "quota", "rate", "deadline", "503", "500", "unavailable", "internal"))
                    if transient:
                        backoff = min(60.0, 2 ** attempt * 5)
                        time.sleep(backoff)
                        continue
                    # Non-transient — try fallback model
                    break
        raise RuntimeError(f"Gemini call failed after retries (and fallback): {last_err}")
