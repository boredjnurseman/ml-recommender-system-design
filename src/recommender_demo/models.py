from dataclasses import dataclass, field


@dataclass
class Candidate:
    """Store the attributes used to filter and rerank one film candidate.

    Attributes:
        item_id: Stable identifier for the synthetic film.
        relevance: Base recommender score on the interval [0, 1].
        genre: Coarse film genre used for inspection only.
        popularity: Normalised popularity on the interval [0, 1].
        recently_seen: Whether the user has previously consumed the film.
        recently_recommended: Whether the film appeared in the user's recent lists.
        regional_availability: Region codes in which the film may be served.
    """

    item_id: int
    relevance: float
    genre: str
    popularity: float
    recently_seen: bool
    recently_recommended: bool
    regional_availability: list[str] = field(default_factory=list)


@dataclass
class CandidateSnapshot:
    """Store one versioned, per-user output from the offline recommender.

    Attributes:
        user_id: Stable pseudonymous user identifier.
        model_version: Version of the model that produced the candidates.
        candidate_refresh_id: Identifier for the offline refresh run.
        top_25_candidates: Candidates ordered by the base relevance score.
    """

    user_id: str
    model_version: str
    candidate_refresh_id: str
    top_25_candidates: list[Candidate] = field(default_factory=list)
