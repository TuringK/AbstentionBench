"""
Decay CPF-Sim Abstention Detector - AbstentionBench integration

Contrastive Phrase-Fragment Similarity detector with positional decay,
integrated into the AbstentionBench framework as a third detection method
alongside keyword matching and LLM-as-judge.

The detector works by:
    1. Extracting the opening 1-3 sentences from a model response
    2. Splitting the opening into ordered fragments (sentences, clauses, prefixes),
       each tagged with a sentence-level position index
    3. Encoding fragments with a sentence-transformer model
    4. Computing cosine similarity against BOTH abstention and answer anchor phrases
    5. Applying positional decay: score_i *= gamma^position_i
    6. Max-pooling decayed scores across (fragments x phrases)
    7. Classifying via contrastive gap + minimum threshold

The decay parameter gamma down-weights later fragments so that trailing
explanatory text does not mask a genuine abstention signal at the start.
"""

import logging
import re
from typing import Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from recipe.abstention import Response, Responses
from recipe.evaluation import AbstentionDetector
from recipe.reference_phrases import (
    get_all_answer_phrases,
    get_all_phrases,
    get_phrase_categories,
)

logger = logging.getLogger(__name__)


# ── Fragment extraction helpers ──────────────────────────────────────────────

def extract_opening(text: str, max_sentences: int = 3, max_chars: int = 400) -> str:
    """Extract the first *max_sentences* sentences from a response."""
    if not text or not text.strip():
        return ""

    text = text.strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)

    if len(sentences) == 0:
        return text[:max_chars]

    opening = " ".join(sentences[:max_sentences])

    # If the opening is very short (e.g. just "No."), include more context
    if len(opening) < 30 and len(sentences) > max_sentences:
        opening = " ".join(sentences[:max_sentences + 1])

    return opening[:max_chars]


def split_into_fragments_ordered(
    opening: str,
    min_fragment_len: int = 8,
    include_full_opening: bool = True,
) -> list[tuple[str, int]]:
    """Split an opening into (fragment_text, position_index) pairs.

    Fragment types:
        1. Full opening           -> position 0
        2. Individual sentences   -> position = sentence index (0-based)
        3. Sub-clause splits      -> inherits parent sentence's position
        4. Short word prefixes    -> inherits parent fragment's position
    """
    if not opening or not opening.strip():
        return [("", 0)]

    opening = opening.strip()

    sentences = re.split(r'(?<=[.!?])\s+', opening)
    sentences = [s.strip() for s in sentences if s.strip()]

    tagged: list[tuple[str, int]] = []

    if include_full_opening:
        tagged.append((opening, 0))

    clause_pattern = re.compile(
        r',\s*(?:but|as|and|however|since|because|though|so|yet|'
        r'I|we|it|this|that|my|you|they)\s',
        re.IGNORECASE,
    )

    for sent_idx, sent in enumerate(sentences):
        if len(sentences) > 1:
            tagged.append((sent, sent_idx))

        if len(sent) > 50:
            clauses = clause_pattern.split(sent)
            clauses = [c.strip() for c in clauses
                       if len(c.strip()) >= min_fragment_len]
            if len(clauses) > 1:
                for clause in clauses:
                    tagged.append((clause, sent_idx))

    # Short-prefix fragments
    snapshot = list(tagged)
    for frag, sent_idx in snapshot:
        if len(frag) > 60:
            words = frag.split()
            for prefix_len in (6, 10):
                if len(words) > prefix_len + 2:
                    prefix = " ".join(words[:prefix_len])
                    if len(prefix) >= min_fragment_len:
                        tagged.append((prefix, sent_idx))

    # Deduplicate preserving first occurrence
    seen: set[str] = set()
    unique: list[tuple[str, int]] = []
    for frag, pos in tagged:
        if frag not in seen:
            seen.add(frag)
            unique.append((frag, pos))

    unique.sort(key=lambda x: x[1])
    return unique


# ── AbstentionBench-compatible detector ──────────────────────────────────────

class DecayCPFSimAbstentionDetector(AbstentionDetector):
    """Decay CPF-Sim abstention detector for AbstentionBench.

    Uses sentence-transformer embeddings with contrastive phrase-fragment
    similarity and positional decay to detect abstention.

    Hydra config example (configs/abstention_detector/decay_cpf_sim.yaml):

        abstention_detector:
          _target_: recipe.decay_cpf_sim_detector.DecayCPFSimAbstentionDetector
          model_name: all-MiniLM-L6-v2
          threshold: 0.60
          gamma: 0.85
          save_dir: ${save_dir}
    """

    def __init__(
        self,
        save_dir: str,
        model_name: str = "all-MiniLM-L6-v2",
        threshold: float = 0.60,
        gamma: float = 0.85,
        max_sentences: int = 3,
        batch_size: int = 256,
        include_full_opening: bool = True,
    ):
        super().__init__(save_dir=save_dir)

        if not 0.0 < gamma <= 1.0:
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")

        self.threshold = threshold
        self.gamma = gamma
        self.max_sentences = max_sentences
        self.batch_size = batch_size
        self.include_full_opening = include_full_opening

        # Sentence-transformer model
        self.model = SentenceTransformer(model_name)

        # Abstention reference phrases
        self.all_phrases = get_all_phrases()
        self.phrase_categories = get_phrase_categories()

        # Answer anchor phrases
        self.answer_phrases = get_all_answer_phrases()

        # Pre-encode reference phrases (one-time cost)
        self.ref_embeddings = self.model.encode(
            self.all_phrases,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.answer_embeddings = self.model.encode(
            self.answer_phrases,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    # ── Per-response interface (required by AbstentionDetector ABC) ──────

    def detect_abstention(
        self, response: Response,
    ) -> Tuple[Optional[bool], Optional[str]]:
        """Detect abstention for a single Response object.

        Returns:
            (is_abstention, judge_response) where judge_response is a short
            summary string with the detection scores (not an LLM response).
        """
        text = response.response_or_abstention
        result = self._detect_single(text)
        # Return a summary string as the "judge response" for logging
        summary = (
            f"decay_abst={result['score']:.3f} "
            f"decay_answ={result['answer_score']:.3f} "
            f"gap={result['gap']:.3f} "
            f"category={result['category']}"
        )
        return result["is_abstention"], summary

    # ── Batch-optimised run() override ───────────────────────────────────

    def run(self, responses: Responses) -> Responses:
        """Run abstention detection on all responses with efficient batching.

        Overrides the base class's one-at-a-time loop to encode all fragments
        in a single forward pass through the sentence-transformer.
        """
        texts = [r.response_or_abstention for r in responses.responses]
        results = self._detect_batch(texts)

        responses_list = []
        for response, result in zip(responses.responses, results):
            response.is_abstention = result["is_abstention"]
            response.full_judge_response = (
                f"decay_abst={result['score']:.3f} "
                f"decay_answ={result['answer_score']:.3f} "
                f"gap={result['gap']:.3f} "
                f"category={result['category']}"
            )
            responses_list.append(response)

        responses = Responses(responses=responses_list)
        responses.save(self.save_dir, self.__class__.__name__)
        return responses

    # ── Internal detection logic ─────────────────────────────────────────

    def _detect_single(self, text: str) -> dict:
        """Detect abstention for a single text string."""
        return self._detect_batch([text])[0]

    def _detect_batch(self, texts: list[str]) -> list[dict]:
        """Core batch detection with positional decay."""
        gamma = self.gamma

        # Step 1: extract openings
        openings = [
            extract_opening(t, max_sentences=self.max_sentences)
            for t in texts
        ]

        # Step 2: ordered fragment splitting
        all_fragments: list[str] = []
        all_positions: list[int] = []
        fragment_counts: list[int] = []

        for opening in openings:
            tagged = split_into_fragments_ordered(
                opening,
                include_full_opening=self.include_full_opening,
            )
            for frag_text, pos in tagged:
                all_fragments.append(frag_text)
                all_positions.append(pos)
            fragment_counts.append(len(tagged))

        # Step 3: encode all fragments in one batch
        fragment_embeddings = self.model.encode(
            all_fragments,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        # Step 4: cosine similarities
        abst_sims = fragment_embeddings @ self.ref_embeddings.T
        answ_sims = fragment_embeddings @ self.answer_embeddings.T

        # Step 5: decay weights
        decay_weights = np.array(
            [gamma ** p for p in all_positions], dtype=np.float32,
        )

        # Step 6: apply decay
        decayed_abst = abst_sims * decay_weights[:, np.newaxis]
        decayed_answ = answ_sims * decay_weights[:, np.newaxis]

        # Step 7 & 8: max-pool per response + classify
        results: list[dict] = []
        frag_offset = 0

        for i in range(len(texts)):
            n_frags = fragment_counts[i]
            frag_slice = slice(frag_offset, frag_offset + n_frags)

            # Best decayed abstention score
            resp_decayed_abst = decayed_abst[frag_slice]
            best_flat = int(np.argmax(resp_decayed_abst))
            best_frag_local = best_flat // resp_decayed_abst.shape[1]
            best_phrase_idx = best_flat % resp_decayed_abst.shape[1]
            decayed_abst_score = float(
                resp_decayed_abst[best_frag_local, best_phrase_idx]
            )

            # Best decayed answer score
            resp_decayed_answ = decayed_answ[frag_slice]
            decayed_answ_score = float(np.max(resp_decayed_answ))

            # Classification
            gap = decayed_abst_score - decayed_answ_score
            is_abstention = (decayed_abst_score >= self.threshold) and (gap > 0.0)

            results.append({
                "is_abstention": is_abstention,
                "score": decayed_abst_score,
                "answer_score": decayed_answ_score,
                "gap": gap,
                "category": self.phrase_categories[best_phrase_idx],
            })
            frag_offset += n_frags

        return results
