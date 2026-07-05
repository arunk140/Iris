CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS videos (
    id           UUID PRIMARY KEY,
    filename     TEXT NOT NULL,
    source_path  TEXT NOT NULL,
    content_hash TEXT,
    size_bytes   BIGINT,
    duration_s   DOUBLE PRECISION,
    width        INT,
    height       INT,
    original_fps DOUBLE PRECISION,
    status       TEXT NOT NULL DEFAULT 'pending'
                     CHECK (status IN ('pending', 'processing', 'done', 'error')),
    error_msg    TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS frames (
    id          BIGSERIAL PRIMARY KEY,
    video_id    UUID NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    idx         INT NOT NULL,
    timestamp_s DOUBLE PRECISION NOT NULL,
    embedding   vector(512) NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (video_id, idx)
);

CREATE INDEX IF NOT EXISTS idx_frames_embedding ON frames
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);

CREATE INDEX IF NOT EXISTS idx_frames_video_id ON frames (video_id);

CREATE INDEX IF NOT EXISTS idx_videos_status ON videos (status);

-- Migrations for existing databases
ALTER TABLE videos ADD COLUMN IF NOT EXISTS content_hash TEXT;
ALTER TABLE videos DROP CONSTRAINT IF EXISTS videos_content_hash_key;
