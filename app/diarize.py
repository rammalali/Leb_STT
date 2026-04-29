"""Open-source speaker diarization using SpeechBrain ECAPA-TDNN + clustering.

No gated models, no HF token required. Pipeline:
  1. Group Whisper word-timestamps into utterances by silence gaps.
  2. Slice the audio per utterance and compute an ECAPA-TDNN voice embedding.
  3. Cluster embeddings (agglomerative cosine) with optional num_speakers pin.
  4. Merge consecutive same-speaker utterances into lines.
"""
import logging
import threading
import traceback
from typing import Any

from .config import Settings

logger = logging.getLogger("leb_stt.diarize")

ENCODER_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
GAP_THRESHOLD_S = 0.6
MIN_UTT_S = 0.3
DEFAULT_DISTANCE_THRESHOLD = 0.7


class DiarizationError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


class Diarizer:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._encoder: Any = None
        self._device: str = "cpu"
        self._lock = threading.Lock()

    def _build(self) -> Any:
        try:
            import torch  # type: ignore
            from speechbrain.inference.speaker import EncoderClassifier  # type: ignore
        except ImportError as e:
            raise DiarizationError(
                500,
                "Diarization needs speechbrain. Run: pip install -r requirements.txt",
            ) from e

        requested = self._settings.device
        if requested == "auto":
            use_cuda = torch.cuda.is_available()
        elif requested == "cuda":
            use_cuda = True
        else:
            use_cuda = False

        self._device = "cuda" if use_cuda else "cpu"

        kwargs: dict[str, Any] = {
            "source": ENCODER_MODEL,
            "savedir": "pretrained_models/spkrec-ecapa-voxceleb",
            "run_opts": {"device": self._device},
        }
        try:
            from speechbrain.utils.fetching import LocalStrategy  # type: ignore
            kwargs["local_strategy"] = LocalStrategy.COPY
        except ImportError:
            pass

        try:
            encoder = EncoderClassifier.from_hparams(**kwargs)
        except Exception as e:
            logger.error("SpeechBrain load failed:\n%s", traceback.format_exc())
            raise DiarizationError(500, f"Could not load speaker encoder: {e}") from e
        return encoder

    def ensure_loaded(self) -> Any:
        if self._encoder is not None:
            return self._encoder
        with self._lock:
            if self._encoder is None:
                self._encoder = self._build()
        return self._encoder

    def diarize_chunks(
        self,
        audio_array,
        sampling_rate: int,
        whisper_chunks: list[dict],
        num_speakers: int | None,
        min_speakers: int | None,
        max_speakers: int | None,
    ) -> tuple[list[dict], list[dict]]:
        """Run diarization. Returns (lines, raw_segments).

        lines:           [{speaker: "SPEAKER_00", start, end, text}, ...]
        raw_segments:    same shape as lines, used for audit.
        """
        import numpy as np  # type: ignore
        import torch  # type: ignore

        encoder = self.ensure_loaded()

        utterances = self._group_into_utterances(whisper_chunks, GAP_THRESHOLD_S)
        if not utterances:
            return [], []
        if len(utterances) == 1:
            u = utterances[0]
            line = {"speaker": "SPEAKER_00", "start": u["start"], "end": u["end"], "text": u["text"]}
            return [line], [line]

        embeddings = []
        for utt in utterances:
            s = int(utt["start"] * sampling_rate)
            e = int(utt["end"] * sampling_rate)
            slice_arr = audio_array[s:e]
            min_len = int(MIN_UTT_S * sampling_rate)
            if len(slice_arr) < min_len:
                slice_arr = np.pad(slice_arr, (0, min_len - len(slice_arr)))
            tensor = torch.tensor(slice_arr, dtype=torch.float32).unsqueeze(0).to(self._device)
            with torch.no_grad():
                emb = encoder.encode_batch(tensor).squeeze().cpu().numpy()
            embeddings.append(emb)

        emb_matrix = np.stack(embeddings)
        emb_matrix = emb_matrix / (np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-9)

        labels = self._cluster(emb_matrix, num_speakers, min_speakers, max_speakers)

        raw_segments = [
            {
                "speaker": f"SPEAKER_{int(label):02d}",
                "start": utt["start"],
                "end": utt["end"],
                "text": utt["text"],
            }
            for utt, label in zip(utterances, labels)
        ]

        lines: list[dict] = []
        for seg in raw_segments:
            if lines and lines[-1]["speaker"] == seg["speaker"]:
                lines[-1]["end"] = seg["end"]
                lines[-1]["text"] = (lines[-1]["text"] + " " + seg["text"]).strip()
            else:
                lines.append(dict(seg))

        return lines, raw_segments

    @staticmethod
    def _cluster(
        embeddings,
        num_speakers: int | None,
        min_speakers: int | None,
        max_speakers: int | None,
    ):
        from sklearn.cluster import AgglomerativeClustering  # type: ignore

        n_samples = len(embeddings)
        if num_speakers is not None:
            k = max(1, min(num_speakers, n_samples))
            clusterer = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
            return clusterer.fit_predict(embeddings)

        clusterer = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=DEFAULT_DISTANCE_THRESHOLD,
        )
        labels = clusterer.fit_predict(embeddings)
        n_found = len(set(labels))

        if min_speakers and n_found < min_speakers:
            k = min(min_speakers, n_samples)
            return AgglomerativeClustering(
                n_clusters=k, metric="cosine", linkage="average"
            ).fit_predict(embeddings)
        if max_speakers and n_found > max_speakers:
            k = min(max_speakers, n_samples)
            return AgglomerativeClustering(
                n_clusters=k, metric="cosine", linkage="average"
            ).fit_predict(embeddings)
        return labels

    @staticmethod
    def _group_into_utterances(chunks: list[dict], gap_threshold: float) -> list[dict]:
        utterances: list[dict] = []
        for chunk in chunks:
            ts = chunk.get("timestamp")
            if not ts or ts[0] is None:
                continue
            start = float(ts[0])
            end = float(ts[1]) if ts[1] is not None else start
            text = chunk.get("text", "")
            if utterances and start - utterances[-1]["end"] < gap_threshold:
                utterances[-1]["end"] = end
                utterances[-1]["text"] += text
            else:
                utterances.append({"start": start, "end": end, "text": text})

        out = []
        for u in utterances:
            u["text"] = u["text"].strip()
            if u["text"]:
                out.append(u)
        return out

    def info(self) -> dict:
        return {
            "diarization_backend": "speechbrain-ecapa",
            "encoder": ENCODER_MODEL,
            "loaded": self._encoder is not None,
            "device": self._device if self._encoder is not None else None,
        }


def relabel(lines: list[dict]) -> list[dict]:
    mapping: dict[str, str] = {}
    out = []
    for line in lines:
        raw = line["speaker"]
        if raw not in mapping:
            mapping[raw] = f"Speaker {len(mapping) + 1}"
        out.append({**line, "speaker": mapping[raw]})
    return out


def render_text(lines: list[dict]) -> str:
    return "\n".join(f"{line['speaker']}: {line['text']}" for line in lines)
