"""Local sentence-embedding redundancy; similarities are evidence, not truth."""

import re
from pathlib import Path

import numpy as np

from qwenlean.utils.io import digest

MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
MODEL_LICENSE = 'Apache-2.0'
METHOD = 'minilm_chunk_cosine_v1'


def chunks(text, min_words=5, max_words=64):
    result = []
    for part in re.split(r'\n+|(?<=[.!?])\s+', text):
        ws = part.split()
        for i in range(0, len(ws), max_words):
            content = ' '.join(ws[i:i+max_words])
            if len(content.split()) >= min_words:
                result.append(content)
    return result


def semantic_statistics(texts, vectors, threshold=.90):
    if len(texts) != len(vectors):
        raise ValueError('Chunk/vector alignment mismatch')
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) if len(vectors) else np.empty((0, 1))
    normalized = vectors / np.maximum(norms, 1e-12)
    pairs, repeated_words = [], 0
    for i in range(1, len(texts)):
        sims = normalized[:i] @ normalized[i]
        j = int(np.argmax(sims))
        if sims[j] >= threshold:
            pairs.append({'earlier_chunk': j, 'later_chunk': i, 'cosine': float(sims[j]),
                          'numeric_literals_match': re.findall(r'-?\d+(?:\.\d+)?', texts[i]) ==
                                                    re.findall(r'-?\d+(?:\.\d+)?', texts[j])})
            repeated_words += len(texts[i].split())
    return {'method': METHOD, 'embedding_model': MODEL, 'embedding_revision': REVISION,
            'threshold': threshold, 'eligible_chunks': len(texts), 'near_duplicate_chunks': len(pairs),
            'density': repeated_words/max(1, sum(len(t.split()) for t in texts)),
            'pairs': pairs, 'chunk_texts': texts,
            'interpretation': 'Near-duplicate representations can include useful verification or numerically different claims. Numeric agreement is reported separately; no oracle of unnecessary computation.'}


class LocalSemanticMetric:
    """Caller must hold the repository model lease; CPU float32, one thread."""

    def __init__(self, cache='.cache/semantic', threshold=.90):
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        self.model = SentenceTransformer(MODEL, revision=REVISION, device='cpu')
        self.model.eval()
        self.threshold = threshold
        self.cache = Path(cache) / REVISION
        self.cache.mkdir(parents=True, exist_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        import gc

        self.model = None
        gc.collect()

    def bound_chunks(self, texts):
        # No source trace truncation. Recursively subdivide embedding-only units
        # until all text fits the encoder's advertised 256-wordpiece window.
        result = []
        for text in texts:
            ids = self.model.tokenizer.encode(text, add_special_tokens=True)
            if len(ids) <= self.model.max_seq_length:
                result.append(text)
            else:
                ws = text.split()
                if len(ws) < 2:
                    raise ValueError('An unsplittable embedding unit exceeds encoder limit')
                middle = len(ws)//2
                result.extend(self.bound_chunks([' '.join(ws[:middle]), ' '.join(ws[middle:])]))
        return result

    def embeddings(self, texts):
        paths = [self.cache / (digest({'method': METHOD, 'text': t}) + '.npy') for t in texts]
        missing = [i for i, p in enumerate(paths) if not p.exists()]
        if missing:
            vecs = self.model.encode([texts[i] for i in missing], batch_size=16,
                                     normalize_embeddings=True, convert_to_numpy=True,
                                     show_progress_bar=False)
            for i, v in zip(missing, vecs):
                np.save(paths[i], v.astype(np.float32))
        return np.stack([np.load(p, allow_pickle=False) for p in paths]) if texts else np.empty((0, 384))

    def measure(self, text):
        units = self.bound_chunks(chunks(text))
        return semantic_statistics(units, self.embeddings(units), self.threshold)
