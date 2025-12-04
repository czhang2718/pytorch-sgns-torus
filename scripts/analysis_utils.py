import os
import pickle
import torch as t
from abc import ABC, abstractmethod
from model import Word2Vec, SGNS


class EmbeddingAnalysis(ABC):
    """Base class for embedding analysis with pluggable distance functions."""
    
    def __init__(
        self,
        run_path: str,
        data_dir: str = "/cache/openwebtext/train",
        device='cuda' if t.cuda.is_available() else 'cpu',
    ):
        self.run_path = run_path
        self.data_dir = data_dir
        self.device = t.device(device)
        self.load_data()

    def load_data(self):
        # Load BOTH input and output embeddings
        idx2ivec_path = f"{self.run_path}/idx2ivec.dat"
        idx2ovec_path = f"{self.run_path}/idx2ovec.dat"

        self.idx2ivec = t.tensor(pickle.load(open(idx2ivec_path, 'rb')), device=self.device)
        self.idx2ovec = t.tensor(pickle.load(open(idx2ovec_path, 'rb')), device=self.device)

        self.e_dim = self.idx2ivec[0].shape[0]

        print(f"Loaded idx2ivec (input vectors) with {len(self.idx2ivec)} embeddings")
        print(f"Loaded idx2ovec (output vectors) with {len(self.idx2ovec)} embeddings")
        print(f"Embedding dimension: {self.e_dim}")

        # Load vocab
        self.words = pickle.load(open(os.path.join(self.data_dir, 'idx2word.dat'), 'rb'))
        self.vocab_size = len(self.words)

        print(f"Vocab size: {self.vocab_size}")

        self.word2ivec = {self.words[idx]: self.idx2ivec[idx] for idx in range(len(self.words))}
        self.word2ovec = {self.words[idx]: self.idx2ovec[idx] for idx in range(len(self.words))}

        print(f"Created word2ivec dict with {len(self.word2ivec)} entries")
        print(f"Created word2ovec dict with {len(self.word2ovec)} entries")

    def _resolve_ivec(self, word: int | str | t.Tensor) -> t.Tensor:
        if isinstance(word, int):
            return self.idx2ivec[word]
        elif isinstance(word, str):
            return self.word2ivec[word]
        return word

    def _resolve_ovec(self, word: int | str | t.Tensor) -> t.Tensor:
        if isinstance(word, int):
            return self.idx2ovec[word]
        elif isinstance(word, str):
            return self.word2ovec[word]
        return word

    @abstractmethod
    def similarity(
        self,
        word1: int | str | t.Tensor,
        word2: int | str | t.Tensor,
    ) -> t.Tensor:
        """Compute similarity between word1 (input) and word2 (output)."""
        pass

    @abstractmethod
    def batch_similarity(
        self,
        word1: list[int | str | t.Tensor],
        word2: list[int | str | t.Tensor],
    ) -> t.Tensor:
        """Compute similarity for batches of words."""
        pass

    def get_closest_words(
        self,
        word: int | str | t.Tensor,
        closest_ovec=True,
        n=10,
    ):
        """Find n closest words using the similarity function."""
        if isinstance(word, int):
            word = self.idx2ivec[word] if closest_ovec else self.idx2ovec[word]
        elif isinstance(word, str):
            word = self.word2ivec[word] if closest_ovec else self.word2ovec[word]
            
        sim = []
        for idx in range(self.vocab_size):
            if closest_ovec:
                h = self.similarity(word, idx).item()
            else:
                h = self.similarity(idx, word).item()
            sim.append((h, self.words[idx]))
        sim = sorted(sim, reverse=True)
        return sim[:n]

    # def get_T_closest_words(
    #     self,
    #     word: int | str | t.Tensor,
    #     n=10,
    # ):
    #     """Find n closest words using toric distance: sum_i 


class ToricEmbeddingAnalysis(EmbeddingAnalysis):
    """Analysis for toric (circular) embeddings using weighted cosine distance."""

    def load_data(self):
        super().load_data()
        
        # Load model to get coord_weights
        model_path = f"{self.run_path}/sgns.pt"
        print(f"Loading model from {model_path}...")
        model = Word2Vec(vocab_size=self.vocab_size, embedding_size=self.e_dim, torus=True)
        sgns = SGNS(embedding=model, vocab_size=self.vocab_size, n_negs=20, torus=True)
        sgns.load_state_dict(t.load(model_path, map_location='cpu'))

        if not hasattr(sgns, 'coord_weights'):
            self.coord_weights = t.eye(self.e_dim).to(self.device)
        else:
            self.coord_weights = sgns.coord_weights.to(self.device)
            print(f"Loaded coord_weights: shape={self.coord_weights.shape}, mean={self.coord_weights.mean().item():.4f}")

        # Summary
        print(f"\nSummary (Toric):")
        print(f"  Embedding dimension: {self.e_dim}")
        print(f"  Vocab size: {self.vocab_size}")
        print(f"  Coord weights available: {self.coord_weights is not None}")
        print(f"  Using device: {self.device}")

    def similarity(
        self,
        word1: int | str | t.Tensor,
        word2: int | str | t.Tensor,
    ) -> t.Tensor:
        """
        Toric similarity: sum(coord_weights * cos(π * (ivec - ovec)))
        """
        ivec = self._resolve_ivec(word1)
        ovec = self._resolve_ovec(word2)
        return (self.coord_weights * t.cos(t.pi * (ivec - ovec))).sum(dim=-1)

    def batch_similarity(
        self,
        word1: list[int | str | t.Tensor],
        word2: list[int | str | t.Tensor],
    ) -> t.Tensor:
        """
        Batch toric similarity: sum(coord_weights * cos(π * (ivec[word1] - ovec[word2])))
        """
        if isinstance(word1[0], str):
            word1 = [self.words.index(word) for word in word1]
        if isinstance(word2[0], str):
            word2 = [self.words.index(word) for word in word2]
        ivecs = t.stack([self.idx2ivec[word] for word in word1])
        ovecs = t.stack([self.idx2ovec[word] for word in word2])
        return (self.coord_weights * t.cos(t.pi * (ivecs - ovecs))).sum(dim=-1)

    def project_T_to_R(
        self,
        X: t.Tensor,
        weighted: bool = True
    ) -> t.Tensor:
        """Project toric coordinates to real space via cos/sin."""
        return t.cat([t.cos(t.pi * X), t.sin(t.pi * X)], dim=-1)

    def steer_torus_point(
        self,
        point: t.Tensor,
        t1: t.Tensor,
        t2: t.Tensor,
    ) -> t.Tensor:
        """
        Steer a T^d point using a steering vector (t2 - t1).
        
        For each dimension:
        1. Convert the point angle and steering endpoints to S^1 (unit circle)
        2. Compute the steering direction as the 2D vector from t1 to t2 on S^1
        3. Find where the ray from the point in that direction intersects S^1
        4. Convert the intersection back to an angle in [-1, 1]
        
        Args:
            point: T^d point with angles in [-1, 1], shape (d,)
            t1: Start of steering vector (steer FROM here), shape (d,)
            t2: End of steering vector (steer TO here), shape (d,)
            
        Returns:
            Projected T^d point with angles in [-1, 1], shape (d,)
        """
        # Convert angles to S^1 coordinates: angle θ ∈ [-1,1] → (cos(πθ), sin(πθ))
        px = t.cos(t.pi * point)  # x-coord of point on each S^1
        py = t.sin(t.pi * point)  # y-coord of point on each S^1
        
        # Steering direction: from t1 to t2 on S^1 (i.e., t2 - t1)
        dx = t.cos(t.pi * t2) - t.cos(t.pi * t1)
        dy = t.sin(t.pi * t2) - t.sin(t.pi * t1)
        
        # Normalize direction (avoid division by zero)
        dir_norm = t.sqrt(dx**2 + dy**2).clamp(min=1e-8)
        dx = dx / dir_norm
        dy = dy / dir_norm
        
        # Ray-circle intersection: find s where |P + s*D| = 1
        # Since P is on circle: |P|² = 1
        # Expanding |P + sD|² = 1:
        #   1 + 2s(P·D) + s²|D|² = 1
        #   s(2(P·D) + s|D|²) = 0
        # Since |D| = 1 (normalized): s = -2(P·D) or s = 0
        # We want the non-zero solution (the "other" intersection)
        
        dot_pd = px * dx + py * dy  # P · D for each dimension
        s_intersect = -2.0 * dot_pd  # Since |D|² = 1
        
        # Compute intersection point
        new_x = px + s_intersect * dx
        new_y = py + s_intersect * dy
        
        # Convert back to angle: atan2 gives angle in [-π, π], normalize to [-1, 1]
        new_angle = t.atan2(new_y, new_x) / t.pi
        
        return new_angle


class RealEmbeddingAnalysis(EmbeddingAnalysis):
    """Analysis for real (Euclidean) embeddings using dot product similarity."""

    def load_data(self):
        super().load_data()

        # Summary
        print(f"\nSummary (Real):")
        print(f"  Embedding dimension: {self.e_dim}")
        print(f"  Vocab size: {self.vocab_size}")
        print(f"  Using device: {self.device}")

    def similarity(
        self,
        word1: int | str | t.Tensor,
        word2: int | str | t.Tensor,
    ) -> t.Tensor:
        """
        Real similarity: cosine similarity of ivec and ovec.
        """
        # print("cosine")
        ivec = self._resolve_ivec(word1)
        ovec = self._resolve_ovec(word2)
        # Cosine similarity: (a · b) / (||a|| * ||b||)
        dot_product = t.dot(ivec, ovec)
        norm_product = t.norm(ivec) * t.norm(ovec)
        return dot_product / norm_product.clamp(min=1e-8)

    def h(
        self,
        word1: int | str | t.Tensor,
        word2: int | str | t.Tensor,
    ) -> t.Tensor:
        """
        Real similarity: dot product of ivec and ovec.
        """
        ivec = self._resolve_ivec(word1)
        ovec = self._resolve_ovec(word2)
        return t.dot(ivec, ovec)

    def batch_similarity(
        self,
        word1: list[int | str | t.Tensor],
        word2: list[int | str | t.Tensor],
    ) -> t.Tensor:
        """
        Batch real similarity: cosine similarity for each pair.
        """
        if isinstance(word1[0], str):
            word1 = [self.words.index(word) for word in word1]
        if isinstance(word2[0], str):
            word2 = [self.words.index(word) for word in word2]
        ivecs = t.stack([self.idx2ivec[word] for word in word1])
        ovecs = t.stack([self.idx2ovec[word] for word in word2])
        # Cosine similarity: (a · b) / (||a|| * ||b||)
        dot_products = (ivecs * ovecs).sum(dim=-1)
        norms = t.norm(ivecs, dim=-1) * t.norm(ovecs, dim=-1)
        return dot_products / norms.clamp(min=1e-8)
