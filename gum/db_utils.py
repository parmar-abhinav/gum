# db_utils.py

from __future__ import annotations

import math
from datetime import datetime, timezone
import re
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sqlalchemy import MetaData, Table, literal_column, select, text, func, union_all
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

import math
import numpy as np
from datetime import datetime, timezone
from typing import List

from sqlalchemy import (
    MetaData,
    Table,
    select,
    literal_column,
    literal,
    text,
    func,
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

K_DECAY  = 2      # whatever you used
LAMBDA   = 0.5    # ditto
EPS      = 1e-12  # protects log(0)

async def search_propositions_bm25(
    session: AsyncSession,
    user_query: str,
    *,
    limit: int = 3,
    mode: str = "OR",
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    include_observations: bool = True,
    enable_decay: bool = True,
    enable_mmr: bool = True,
) -> list[tuple["Proposition", float]]:

    q = build_fts_query(user_query, mode)
    has_query = bool(q)

    # --------------------------------------------------------
    # 1  Build candidate list
    # --------------------------------------------------------
    candidate_pool = limit * 10 if enable_mmr else limit
    has_child      = _has_child_subquery()

    if has_query:
        fts_prop = Table("propositions_fts", MetaData())

        if include_observations:
            # --- 1-a-1  WITH observations --------------------
            fts_obs  = Table("observations_fts", MetaData())

            bm25_p   = literal_column("bm25(propositions_fts)").label("score")
            bm25_o   = literal_column("bm25(observations_fts)").label("score")

            sub_p = (
                select(Proposition.id.label("pid"), bm25_p)
                .select_from(
                    fts_prop.join(
                        Proposition,
                        literal_column("propositions_fts.rowid") == Proposition.id,
                    )
                )
                .where(text("propositions_fts MATCH :q"))
            )

            sub_o = (
                select(observation_proposition.c.proposition_id.label("pid"), bm25_o)
                .select_from(
                    fts_obs
                    .join(
                        Observation,
                        literal_column("observations_fts.rowid") == Observation.id,
                    )
                    .join(
                        observation_proposition,
                        observation_proposition.c.observation_id == Observation.id,
                    )
                )
                .where(text("observations_fts MATCH :q"))
            )

            union_sub = sub_p.union_all(sub_o).subquery()

            best_scores = (
                select(
                    union_sub.c.pid,
                    func.min(union_sub.c.score).label("bm25"),
                )
                .group_by(union_sub.c.pid)
                .subquery()
            )
        else:
            # --- 1-a-2  WITHOUT observations -----------------
            best_scores = (
                select(
                    Proposition.id.label("pid"),
                    literal_column("bm25(propositions_fts)").label("bm25"),
                )
                .select_from(
                    fts_prop.join(
                        Proposition,
                        literal_column("propositions_fts.rowid") == Proposition.id,
                    )
                )
                .where(text("propositions_fts MATCH :q"))
                .subquery()
            )

        stmt = (
            select(Proposition, best_scores.c.bm25)
            .join(best_scores, best_scores.c.pid == Proposition.id)
            .where(~has_child)
            .order_by(best_scores.c.bm25.asc())          # smallest→best
        )
    else:
        # --- 1-b  No user query ------------------------------
        stmt = (
            select(Proposition, literal_column("0.0").label("bm25"))
            .where(~has_child)
            .order_by(Proposition.created_at.desc())
        )

    # --------------------------------------------------------
    # 2  Time filtering & eager-load
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

    if include_observations:
        stmt = stmt.options(selectinload(Proposition.observations))

    stmt = stmt.limit(candidate_pool)

    # --------------------------------------------------------
    # 3  Execute & score in log-space
    # --------------------------------------------------------
    bind = {"q": q} if has_query else {}
    rows = (await session.execute(stmt, bind)).all()
    if not rows:
        return []

    rel_scores: list[float] = []
    now = datetime.now(timezone.utc)

    for prop, raw_score in rows:
        # -------- 3-a  log(BM25) (raw_score > 0) ------------
        ln_raw = math.log(max(raw_score, EPS))          # smaller→more-negative

        # -------- 3-b  log(time-decay) ----------------------
        ln_gamma = 0.0                                  # ln(1) → no decay
        if enable_decay:
            dt        = prop.created_at.replace(tzinfo=timezone.utc)
            age_days  = max((now - dt).total_seconds() / 86_400, 0.0)
            alpha     = prop.decay if prop.decay is not None else 0.0
            ln_gamma  = -alpha * K_DECAY * age_days     # γ = e^(…);  ln γ ≤ 0

        # -------- 3-c  final score (larger-is-better) ------
        # score = −ln(raw_score) + ln_gamma
        score = -ln_raw + ln_gamma
        rel_scores.append(score)

    # --------------------------------------------------------
    # 4  Optional MMR diversification
    # --------------------------------------------------------
    if enable_mmr and len(rows) > 1:
        docs: list[str] = []
        for p, _ in rows:
            obs_concat = (
                " ".join(o.content for o in p.observations[:10])
                if include_observations else ""
            )
            docs.append(f"{p.text} {p.reasoning} {obs_concat}")

        vecs = TfidfVectorizer().fit_transform(docs)

        selected_idxs, final_scores = [], []
        while len(selected_idxs) < min(limit, len(rows)):
            if not selected_idxs:
                idx = int(np.argmax(rel_scores))
            else:
                sims = cosine_similarity(vecs, vecs[selected_idxs]).max(axis=1)
                mmr  = LAMBDA * np.array(rel_scores) - (1 - LAMBDA) * sims
                mmr[selected_idxs] = -np.inf
                idx = int(np.argmax(mmr))
            selected_idxs.append(idx)
            final_scores.append(rel_scores[idx])

    else:
        # no MMR → pick rows by simple order
        if has_query:                                 # real query → sort by score
            idxs = np.argsort(rel_scores)[::-1][:limit]
        else:                                         # no query → SQL already sorted
            idxs = list(range(min(limit, len(rows))))
        selected_idxs   = idxs
        final_scores    = [rel_scores[i] for i in idxs]

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