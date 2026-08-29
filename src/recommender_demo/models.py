from dataclasses import dataclass, field


@dataclass
class Candidate:
    item_id: int
    relevance: float
    genre: str
    popularity: float
    recently_seen: bool
    recently_recommended: bool
    regional_availability: list[str] = field(default_factory=list)


@dataclass
class CandidateSnapshot:
    user_id: str
    model_version: str
    candidate_refresh_id: str
    top_25_candidates: list[Candidate] = field(default_factory=list)