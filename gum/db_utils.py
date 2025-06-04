# db_utils.py

from __future__ import annotations

import math
from datetime import datetime, timezone
import re
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sqlalchemy import MetaData, Table, literal_column, select, text, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from .models import (
    Observation,
    Proposition,
    proposition_parent,
    observation_proposition,
)

def build_fts_query(raw: str, mode: str = "OR") -> str:
    tokens = re.findall(r"\w+", raw.lower())
    if not tokens:
        return ""
    if mode == "PHRASE":
        return f'"{" ".join(tokens)}"'
    elif mode == "OR":
        return " OR ".join(tokens)
    else:                              # implicit AND
        return " ".join(tokens)

def _has_child_subquery() -> select:
    return (
        select(literal_column("1"))
        .select_from(proposition_parent)
        .where(proposition_parent.c.parent_id == Proposition.id)
        .exists()
    )

# constants
K_DECAY = 2.0     # decay rate for recency adjustment
LAMBDA = 0.5      # trade-off for MMR

async def search_propositions_bm25(
    session: AsyncSession,
    user_query: str,
    *,
    limit: int = 3,
    mode: str = "OR",
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> list[tuple[Proposition, float]]:

    q = build_fts_query(user_query, mode)
    has_query = bool(q)          # <- remember whether we really have an FTS query

    # --------------------------------------------------------
    # 1  Build candidate list
    # --------------------------------------------------------
    candidate_pool = max(limit * 10, limit)
    has_child = _has_child_subquery()

    # ----------  1-a.  With a real query  ----------
    if has_query:
        fts_prop = Table("propositions_fts", MetaData())
        fts_obs  = Table("observations_fts",  MetaData())

        bm25_p = literal_column("bm25(propositions_fts)").label("score")
        bm25_o = literal_column("bm25(observations_fts)").label("score")

        sub_p = (
            select(Proposition.id.label("pid"), bm25_p)
            .select_from(fts_prop.join(Proposition,
                                       literal_column("propositions_fts.rowid") == Proposition.id))
            .where(text("propositions_fts MATCH :q"))
        )

        sub_o = (
            select(observation_proposition.c.proposition_id.label("pid"), bm25_o)
            .select_from(
                fts_obs
                .join(Observation,
                      literal_column("observations_fts.rowid") == Observation.id)
                .join(observation_proposition,
                      observation_proposition.c.observation_id == Observation.id)
            )
            .where(text("observations_fts MATCH :q"))
        )

        union_sub  = sub_p.union_all(sub_o).subquery()
        best_scores = (
            select(union_sub.c.pid,
                   func.min(union_sub.c.score).label("bm25"))
            .group_by(union_sub.c.pid)
            .subquery()
        )

        stmt = (
            select(Proposition, best_scores.c.bm25)
            .join(best_scores, best_scores.c.pid == Proposition.id)
            .where(~has_child)
        )

    # ----------  1-b.  No query – return “something” ----------
    else:
        # Give every row the same dummy BM25 so later code keeps working.
        stmt = (
            select(Proposition,
                   literal_column("0.0").label("bm25"))
            .where(~has_child)
            # In the no-query case you probably want a different sort;
            # here we simply return most recent first:
            .order_by(Proposition.created_at.desc())
        )

    # --------------------------------------------------------
    # 2  Time filtering and limit
    # --------------------------------------------------------
    if end_time is None:
        end_time = datetime.now(timezone.utc)
    if start_time is not None and start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)
    if end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=timezone.utc)

    if start_time is not None:
        stmt = stmt.where(Proposition.created_at >= start_time)
    stmt = stmt.where(Proposition.created_at <= end_time)

    stmt = (
        stmt.options(selectinload(Proposition.observations))
        .limit(candidate_pool)
    )

    # Supply :q only when there is a real query
    bind = {"q": q} if has_query else {}
    raw  = await session.execute(stmt, bind)
    rows = raw.all()
    if not rows:
        return []

    now = datetime.now(timezone.utc)
    rel_scores: List[float] = []
    for prop, raw_score in rows:
        # Ensure tz‑aware timestamp
        dt = prop.created_at
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        age_days = max((now - dt).total_seconds() / 86400, 0.0)
        alpha = prop.decay if prop.decay is not None else 0.0
        gamma = math.exp(-alpha * K_DECAY * age_days)

        r_eff = -raw_score * gamma  # BM25: lower is better → negate
        rel_scores.append(r_eff)

    docs: list[str] = []
    for p, _ in rows:
        obs_concat = " ".join(o.content for o in list(p.observations)[:10])
        docs.append(f"{p.text} {p.reasoning} {obs_concat}")

    vecs = TfidfVectorizer().fit_transform(docs)

    selected_idxs: List[int] = []
    final_scores:  List[float] = []

    while len(selected_idxs) < min(limit, len(rows)):
        if not selected_idxs:
            idx = int(np.argmax(rel_scores))
        else:
            sims = cosine_similarity(vecs, vecs[selected_idxs]).max(axis=1)
            mmr = LAMBDA * np.array(rel_scores) - (1 - LAMBDA) * sims
            mmr[selected_idxs] = -np.inf  # don’t repeat
            idx = int(np.argmax(mmr))
        selected_idxs.append(idx)
        final_scores.append(rel_scores[idx])

    return [(rows[i][0], final_scores[pos]) for pos, i in enumerate(selected_idxs)]

async def get_related_observations(
    session: AsyncSession,
    proposition_id: int,
    *,  # Force keyword arguments for optional parameters
    limit: int = 5,
) -> List[Observation]:

    stmt = (
        select(Observation)
        .join(Observation.propositions)
        .where(Proposition.id == proposition_id)
        .order_by(Observation.created_at.desc())
        .limit(limit)  # Use the limit parameter
    )
    result = await session.execute(stmt)
    return result.scalars().all()