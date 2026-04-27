"""
Shared OpenRouter vision client for EVE-RAG.

Usage:
    from src.utils.mllm_client import MLLMClient
    gc = MLLMClient()
    text = gc.generate(prompt, image_path=Path("x.jpg"))

Features:
- Disk-backed cache (cache/mllm/openrouter_responses/) keyed by hash(prompt + image_bytes + model).
- Soft rate limit: at most rpm_limit requests per minute.
- Retries on transient errors (429, 500, 502, 503).
- Returns plain text; raises only on fatal config / auth failures.
"""
from __future__ import annotations
import os, json, time, hashlib, base64
from pathlib import Path
from dataclasses import dataclass, field
from io import BytesIO

from openai import OpenAI
from PIL import Image
from dotenv import load_dotenv


@dataclass
class MLLMClient:
    model_name: str = "google/gemma-4-31b-it:free"
    rpm_limit: int = 18
    cache_dir: Path = None
    image_max_side: int = 512
    max_retries: int = 5
    request_log: list = field(default_factory=list)

    def __post_init__(self):
        here = Path(__file__).resolve()
        for p in [here] + list(here.parents):
            if (p / "src" / "utils" / "paths.py").exists():
                self.repo_root = p
                break
        else:
            self.repo_root = Path.cwd()
        load_dotenv(self.repo_root / ".env", override=True)
        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not in .env at repo root.")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        if self.cache_dir is None:
            self.cache_dir = self.repo_root / "cache" / "mllm" / "openrouter_responses"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _key(prompt: str, image_path, model_name: str) -> str:
        h = hashlib.sha256()
        h.update(model_name.encode())
        h.update(b"||")
        h.update(prompt.encode("utf-8"))
        h.update(b"||")
        if image_path is not None and Path(image_path).exists():
            h.update(Path(image_path).read_bytes())
        return h.hexdigest()[:32]

    def _wait_if_needed(self):
        now = time.time()
        self.request_log = [t for t in self.request_log if now - t < 60]
        if len(self.request_log) >= self.rpm_limit:
            sleep_for = 60 - (now - self.request_log[0]) + 0.5
            if sleep_for > 0:
                time.sleep(sleep_for)
                self.request_log = [t for t in self.request_log if time.time() - t < 60]

    def generate(self, prompt: str, image_path=None,
                 use_cache: bool = True, temperature: float = 0.2,
                 max_output_tokens: int = 512) -> str:
        key = self._key(prompt, image_path, self.model_name)
        cache_file = self.cache_dir / f"{key}.json"

        if use_cache and cache_file.exists():
            try:
                return json.loads(cache_file.read_text(encoding="utf-8"))["text"]
            except Exception:
                pass

        self._wait_if_needed()
        content = [{"type": "text", "text": prompt}]
        if image_path is not None:
            img = Image.open(image_path).convert("RGB")
            img.thumbnail((self.image_max_side, self.image_max_side))
            buf = BytesIO()
            img.save(buf, format="JPEG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

        last_err = None
        for attempt in range(self.max_retries):
            try:
                self.request_log.append(time.time())
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": content}],
                    temperature=temperature,
                    max_tokens=max_output_tokens,
                )
                text = (resp.choices[0].message.content or "").strip()
                cache_file.write_text(json.dumps({
                    "text": text, "prompt": prompt[:2000],
                    "image": str(image_path) if image_path else None,
                    "model": self.model_name, "ts": time.time(),
                }), encoding="utf-8")
                return text
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                if any(k in msg for k in ("429", "quota", "rate", "502", "503", "500", "unavailable")):
                    backoff = min(60.0, 2 ** attempt * 5)
                    time.sleep(backoff)
                    continue
                raise
        raise RuntimeError(f"MLLMClient failed after {self.max_retries} retries: {last_err}")