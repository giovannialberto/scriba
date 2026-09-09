-- SCRIBA v0.2.0 Database Schema
-- Designed for future AI-powered knowledge base capabilities

CREATE TABLE IF NOT EXISTS recordings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    directory_name TEXT NOT NULL UNIQUE,  -- e.g., "2025-08-10_14-30-25_meeting-notes"
    display_name TEXT,                     -- User-friendly name
    created_at DATETIME NOT NULL,          -- When recording was created
    updated_at DATETIME NOT NULL,          -- Last modified
    duration_seconds INTEGER,              -- Audio duration in seconds
    file_size_bytes INTEGER,              -- Size of audio file
    audio_format TEXT DEFAULT 'wav',      -- Audio file format
    sample_rate INTEGER DEFAULT 48000,    -- Audio sample rate
    channels INTEGER DEFAULT 1,           -- Mono/Stereo
    has_transcript BOOLEAN DEFAULT 0,     -- Whether transcript exists
    transcript_status TEXT DEFAULT 'pending', -- pending, processing, completed, failed
    language_code TEXT DEFAULT 'auto',    -- Language used for transcription
    model_used TEXT DEFAULT 'whisper-1',  -- AI model used for transcription

    -- Future AI features
    tags TEXT,                            -- JSON array of tags
    summary TEXT,                         -- AI-generated summary
    key_points TEXT,                      -- JSON array of key points
    action_items TEXT,                    -- JSON array of action items
    speakers TEXT,                        -- JSON array of detected speakers
    sentiment_score REAL,                 -- Overall sentiment (-1 to 1)

    -- Metadata for search and organization
    search_index TEXT,                    -- Full-text search index
    categories TEXT,                      -- JSON array of categories
    confidence_score REAL,               -- Transcription confidence (0-1)

    -- File paths (relative to base directory)
    audio_path TEXT NOT NULL,            -- Path to audio file
    transcript_path TEXT,                -- Path to transcript file
    transcript_error TEXT                -- Why the last transcription failed (transcript_status = 'failed')
);

CREATE TABLE IF NOT EXISTS transcripts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recording_id INTEGER NOT NULL,
    content TEXT NOT NULL,               -- Full transcript text
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,

    -- Transcript metadata
    word_count INTEGER,
    character_count INTEGER,
    language_detected TEXT,
    confidence_scores TEXT,              -- JSON array of per-segment confidence

    -- Segmentation (for future features like speaker diarization)
    segments TEXT,                       -- JSON array of time-stamped segments

    -- AI processing results
    entities TEXT,                       -- JSON array of named entities (people, places, etc.)
    topics TEXT,                        -- JSON array of detected topics

    FOREIGN KEY (recording_id) REFERENCES recordings(id) ON DELETE CASCADE
);

-- Table for AI-powered search and knowledge base
CREATE TABLE IF NOT EXISTS knowledge_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recording_id INTEGER NOT NULL,
    transcript_id INTEGER,

    -- Knowledge extraction
    item_type TEXT NOT NULL,             -- 'fact', 'action_item', 'decision', 'question', etc.
    content TEXT NOT NULL,               -- The extracted knowledge
    context TEXT,                        -- Surrounding context
    confidence REAL,                     -- Extraction confidence

    -- Temporal info
    timestamp_start REAL,                -- Start time in recording (seconds)
    timestamp_end REAL,                  -- End time in recording (seconds)
    created_at DATETIME NOT NULL,

    -- Relationships and references
    related_items TEXT,                  -- JSON array of related knowledge item IDs
    entities_mentioned TEXT,             -- JSON array of people/places/things mentioned

    FOREIGN KEY (recording_id) REFERENCES recordings(id) ON DELETE CASCADE,
    FOREIGN KEY (transcript_id) REFERENCES transcripts(id) ON DELETE CASCADE
);

-- Table for tagging and categorization
CREATE TABLE IF NOT EXISTS tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    color TEXT DEFAULT '#6366f1',       -- Hex color for UI
    created_at DATETIME NOT NULL,
    usage_count INTEGER DEFAULT 0       -- How many recordings use this tag
);

CREATE TABLE IF NOT EXISTS recording_tags (
    recording_id INTEGER NOT NULL,
    tag_id INTEGER NOT NULL,
    created_at DATETIME NOT NULL,
    PRIMARY KEY (recording_id, tag_id),
    FOREIGN KEY (recording_id) REFERENCES recordings(id) ON DELETE CASCADE,
    FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
);

-- Global entity registry for people, organizations, and other entities
CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL,              -- 'person', 'organization', 'project', etc.
    canonical_name TEXT NOT NULL,           -- "John Smith" - the primary/official name
    aliases TEXT,                           -- JSON: ["John", "Johnny", "J. Smith"]
    context TEXT,                           -- AI-maintained description/context
    metadata TEXT,                          -- JSON: {"role": "CTO", "company_id": 5}
    mention_count INTEGER DEFAULT 1,        -- How many times mentioned across recordings
    first_seen_at DATETIME,                 -- When first encountered
    last_seen_at DATETIME,                  -- When last mentioned
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL
);

-- Link mentions in recordings to entities
CREATE TABLE IF NOT EXISTS entity_mentions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id INTEGER,                      -- FK to entities (NULL if unlinked)
    recording_id INTEGER NOT NULL,          -- FK to recordings
    mention_text TEXT NOT NULL,             -- "John" as appeared in transcript
    context_snippet TEXT,                   -- ~100 chars around the mention
    confidence REAL DEFAULT 1.0,            -- AI linking confidence (0-1)
    linked_at DATETIME,                     -- When the link was established
    created_at DATETIME NOT NULL,
    FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE SET NULL,
    FOREIGN KEY (recording_id) REFERENCES recordings(id) ON DELETE CASCADE
);

-- Indexes for entity queries
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(canonical_name);
CREATE INDEX IF NOT EXISTS idx_entity_mentions_entity ON entity_mentions(entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_mentions_recording ON entity_mentions(recording_id);

-- Table for AI chat/query history (future feature)
CREATE TABLE IF NOT EXISTS ai_queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,                 -- User's question
    response TEXT NOT NULL,              -- AI's response
    context_recordings TEXT,             -- JSON array of recording IDs used for context
    created_at DATETIME NOT NULL,
    response_time_ms INTEGER,            -- How long the query took
    satisfaction_rating INTEGER          -- User feedback (1-5 stars)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_recordings_created_at ON recordings(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_recordings_directory_name ON recordings(directory_name);
CREATE INDEX IF NOT EXISTS idx_recordings_search ON recordings(display_name, tags, summary);
CREATE INDEX IF NOT EXISTS idx_transcripts_recording_id ON transcripts(recording_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_items_recording_id ON knowledge_items(recording_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_items_type ON knowledge_items(item_type);
CREATE INDEX IF NOT EXISTS idx_recording_tags_recording_id ON recording_tags(recording_id);

-- Full-text search virtual table (for transcript content)
-- Note: Virtual tables don't support IF NOT EXISTS, so we drop and recreate
DROP TABLE IF EXISTS transcripts_fts;
CREATE VIRTUAL TABLE transcripts_fts USING fts5(
    content,
    recording_id UNINDEXED,
    content='transcripts',
    content_rowid='id'
);

-- Triggers to maintain FTS index
-- Note: Triggers don't support IF NOT EXISTS, so we drop and recreate
DROP TRIGGER IF EXISTS transcripts_fts_insert;
CREATE TRIGGER transcripts_fts_insert AFTER INSERT ON transcripts BEGIN
    INSERT INTO transcripts_fts(rowid, content, recording_id)
    VALUES (new.id, new.content, new.recording_id);
END;

DROP TRIGGER IF EXISTS transcripts_fts_delete;
CREATE TRIGGER transcripts_fts_delete AFTER DELETE ON transcripts BEGIN
    INSERT INTO transcripts_fts(transcripts_fts, rowid, content, recording_id)
    VALUES('delete', old.id, old.content, old.recording_id);
END;

DROP TRIGGER IF EXISTS transcripts_fts_update;
CREATE TRIGGER transcripts_fts_update AFTER UPDATE ON transcripts BEGIN
    INSERT INTO transcripts_fts(transcripts_fts, rowid, content, recording_id)
    VALUES('delete', old.id, old.content, old.recording_id);
    INSERT INTO transcripts_fts(rowid, content, recording_id)
    VALUES (new.id, new.content, new.recording_id);
END;

-- Voice samples used to recognize known speakers (embeddings only, never audio).
CREATE TABLE IF NOT EXISTS speaker_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    speaker TEXT NOT NULL,                -- 'owner' or an entity canonical name
    is_owner INTEGER NOT NULL DEFAULT 0,
    embedding TEXT NOT NULL,              -- JSON array of floats, L2-normalized
    duration_secs REAL NOT NULL DEFAULT 0,
    source TEXT NOT NULL,                 -- 'enrollment' | 'mic-track' | 'confirmed'
    recording_id INTEGER,
    created_at DATETIME NOT NULL,
    FOREIGN KEY (recording_id) REFERENCES recordings(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_speaker_samples_speaker ON speaker_samples(speaker);

-- Initial data (only insert if not already present)
INSERT OR IGNORE INTO tags (name, color, created_at) VALUES
    ('meeting', '#3b82f6', datetime('now')),
    ('interview', '#10b981', datetime('now')),
    ('brainstorm', '#f59e0b', datetime('now')),
    ('call', '#8b5cf6', datetime('now')),
    ('personal', '#ef4444', datetime('now'));
