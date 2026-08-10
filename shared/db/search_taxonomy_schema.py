"""검색 의미 사전과 공고 연결을 저장하는 SQLite 스키마."""

SEARCH_TAXONOMY_SCHEMA = """
CREATE TABLE IF NOT EXISTS taxonomy_sources (
    source_key      TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    version         TEXT NOT NULL,
    source_url      TEXT NOT NULL,
    license         TEXT,
    downloaded_at   TEXT,
    imported_at     TEXT NOT NULL,
    metadata_json   TEXT
);

CREATE TABLE IF NOT EXISTS search_concepts (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    concept_key         TEXT NOT NULL UNIQUE,
    concept_type        TEXT NOT NULL,
    preferred_label_ko  TEXT,
    preferred_label_en  TEXT,
    definition          TEXT,
    status              TEXT NOT NULL DEFAULT 'active',
    source_key          TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL,
    FOREIGN KEY(source_key) REFERENCES taxonomy_sources(source_key),
    CHECK(concept_type IN ('occupation', 'skill', 'domain', 'employment_type', 'experience_level')),
    CHECK(status IN ('candidate', 'active', 'deprecated'))
);

CREATE INDEX IF NOT EXISTS idx_search_concepts_type
ON search_concepts(concept_type, status);

CREATE TABLE IF NOT EXISTS search_aliases (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    concept_id          INTEGER NOT NULL,
    alias               TEXT NOT NULL,
    normalized_alias    TEXT NOT NULL,
    language            TEXT NOT NULL DEFAULT 'und',
    alias_type          TEXT NOT NULL DEFAULT 'exact',
    source_key          TEXT NOT NULL,
    active              INTEGER NOT NULL DEFAULT 1,
    created_at          TEXT NOT NULL,
    UNIQUE(concept_id, normalized_alias, language),
    FOREIGN KEY(concept_id) REFERENCES search_concepts(id) ON DELETE CASCADE,
    FOREIGN KEY(source_key) REFERENCES taxonomy_sources(source_key),
    CHECK(alias_type IN ('preferred', 'exact', 'abbreviation', 'hidden')),
    CHECK(active IN (0, 1))
);

CREATE INDEX IF NOT EXISTS idx_search_aliases_normalized
ON search_aliases(normalized_alias, active);

CREATE TABLE IF NOT EXISTS search_concept_relations (
    source_concept_id   INTEGER NOT NULL,
    target_concept_id   INTEGER NOT NULL,
    relation_type       TEXT NOT NULL,
    source_key          TEXT NOT NULL,
    metadata_json       TEXT,
    created_at          TEXT NOT NULL,
    PRIMARY KEY(source_concept_id, target_concept_id, relation_type, source_key),
    FOREIGN KEY(source_concept_id) REFERENCES search_concepts(id) ON DELETE CASCADE,
    FOREIGN KEY(target_concept_id) REFERENCES search_concepts(id) ON DELETE CASCADE,
    FOREIGN KEY(source_key) REFERENCES taxonomy_sources(source_key),
    CHECK(relation_type IN ('broader', 'related', 'occupation_skill'))
);

CREATE INDEX IF NOT EXISTS idx_search_relations_target
ON search_concept_relations(target_concept_id, relation_type);

CREATE TABLE IF NOT EXISTS job_concept_links (
    job_id              INTEGER NOT NULL,
    concept_id          INTEGER NOT NULL,
    link_type           TEXT NOT NULL,
    evidence_field      TEXT NOT NULL,
    evidence_text       TEXT NOT NULL,
    requirement_type    TEXT NOT NULL DEFAULT 'mentioned',
    minimum_months      INTEGER,
    confidence          REAL NOT NULL,
    linked_by           TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL,
    PRIMARY KEY(job_id, concept_id, link_type, evidence_field),
    FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE,
    FOREIGN KEY(concept_id) REFERENCES search_concepts(id) ON DELETE CASCADE,
    CHECK(link_type IN ('occupation', 'skill', 'domain')),
    CHECK(requirement_type IN ('required', 'preferred', 'mentioned')),
    CHECK(confidence >= 0.0 AND confidence <= 1.0)
);

CREATE INDEX IF NOT EXISTS idx_job_concept_links_concept
ON job_concept_links(concept_id, link_type, job_id);
"""


__all__ = ["SEARCH_TAXONOMY_SCHEMA"]
