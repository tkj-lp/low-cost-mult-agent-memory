from __future__ import annotations

from pathlib import Path


class SentenceTransformerEncoder:
    """Local, explicitly configured encoder; no model download or hash fallback."""

    def __init__(
        self, model_path: str | Path, *, model_id: str, device: str = "cpu"
    ) -> None:
        if not model_id.strip():
            raise ValueError(
                "model_id must include the model revision and pooling configuration"
            )
        path = Path(model_path).resolve()
        if not path.is_dir():
            raise FileNotFoundError(
                f"Local embedding model directory does not exist: {path}"
            )
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required in the active Python environment"
            ) from exc
        self.model_id = model_id
        self.model = SentenceTransformer(
            str(path), device=device, local_files_only=True, trust_remote_code=False
        )

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).tolist()
