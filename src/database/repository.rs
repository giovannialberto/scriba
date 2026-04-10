//! Database repository with DRY row mapping.

use anyhow::{anyhow, Context, Result};
use chrono::Utc;
use dirs::home_dir;
use rusqlite::{params, Connection, Row};
use std::path::PathBuf;

use super::models::{Entity, EntityMentionRecord, Recording, RecordingStats, Transcript};

/// Maps a SQLite row to a Recording struct.
/// This eliminates the duplicate mapping code that was in 4+ places.
fn row_to_recording(row: &Row) -> rusqlite::Result<Recording> {
    Ok(Recording {
        id: Some(row.get(0)?),
        directory_name: row.get(1)?,
        display_name: row.get(2)?,
        created_at: row.get(3)?,
        updated_at: row.get(4)?,
        duration_seconds: row.get(5)?,
        file_size_bytes: row.get(6)?,
        audio_format: row.get(7)?,
        sample_rate: row.get(8)?,
        channels: row.get(9)?,
        has_transcript: row.get(10)?,
        transcript_status: row.get(11)?,
        language_code: row.get(12)?,
        model_used: row.get(13)?,
        tags: row.get(14)?,
        summary: row.get(15)?,
        key_points: row.get(16)?,
        action_items: row.get(17)?,
        speakers: row.get(18)?,
        sentiment_score: row.get(19)?,
        search_index: row.get(20)?,
        categories: row.get(21)?,
        confidence_score: row.get(22)?,
        audio_path: row.get(23)?,
        transcript_path: row.get(24)?,
    })
}

/// Maps a SQLite row to a Transcript struct.
fn row_to_transcript(row: &Row) -> rusqlite::Result<Transcript> {
    Ok(Transcript {
        id: Some(row.get(0)?),
        recording_id: row.get(1)?,
        content: row.get(2)?,
        created_at: row.get(3)?,
        updated_at: row.get(4)?,
        word_count: row.get(5)?,
        character_count: row.get(6)?,
        language_detected: row.get(7)?,
        confidence_scores: row.get(8)?,
        segments: row.get(9)?,
        entities: row.get(10)?,
        topics: row.get(11)?,
    })
}

/// Maps a combined row (recordings + transcripts join) to both structs.
fn row_to_recording_and_transcript(row: &Row) -> rusqlite::Result<(Recording, Transcript)> {
    let recording = Recording {
        id: Some(row.get(0)?),
        directory_name: row.get(1)?,
        display_name: row.get(2)?,
        created_at: row.get(3)?,
        updated_at: row.get(4)?,
        duration_seconds: row.get(5)?,
        file_size_bytes: row.get(6)?,
        audio_format: row.get(7)?,
        sample_rate: row.get(8)?,
        channels: row.get(9)?,
        has_transcript: row.get(10)?,
        transcript_status: row.get(11)?,
        language_code: row.get(12)?,
        model_used: row.get(13)?,
        tags: row.get(14)?,
        summary: row.get(15)?,
        key_points: row.get(16)?,
        action_items: row.get(17)?,
        speakers: row.get(18)?,
        sentiment_score: row.get(19)?,
        search_index: row.get(20)?,
        categories: row.get(21)?,
        confidence_score: row.get(22)?,
        audio_path: row.get(23)?,
        transcript_path: row.get(24)?,
    };

    let transcript = Transcript {
        id: Some(row.get(25)?),
        recording_id: row.get(26)?,
        content: row.get(27)?,
        created_at: row.get(28)?,
        updated_at: row.get(29)?,
        word_count: row.get(30)?,
        character_count: row.get(31)?,
        language_detected: row.get(32)?,
        confidence_scores: row.get(33)?,
        segments: row.get(34)?,
        entities: row.get(35)?,
        topics: row.get(36)?,
    };

    Ok((recording, transcript))
}

/// Database connection wrapper with all repository operations.
pub struct Database {
    conn: Connection,
}

impl Database {
    /// Create a new database connection.
    pub fn new() -> Result<Self> {
        let db_path = Self::get_database_path()?;

        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let conn = Connection::open(&db_path).context("Failed to open database connection")?;

        let mut db = Database { conn };

        db.initialize()
            .context("Failed to initialize database. Run `scriba db rebuild` to recover.")?;

        Ok(db)
    }

    fn get_database_path() -> Result<PathBuf> {
        let home = home_dir().context("Could not find home directory")?;
        Ok(home.join("scriba_recordings").join("scriba.db"))
    }

    fn initialize(&mut self) -> Result<()> {
        self.conn
            .execute("PRAGMA foreign_keys = ON", [])
            .context("Failed to enable foreign key constraints")?;

        if self.conn.execute("PRAGMA journal_mode = WAL", []).is_err() {
            // Fallback to default if not supported
        }

        let _ = self.conn.execute("PRAGMA busy_timeout = 5000", []);

        {
            let tx = self
                .conn
                .transaction()
                .context("Failed to start schema transaction")?;

            let user_version: i64 = tx
                .query_row("PRAGMA user_version", [], |row| row.get(0))
                .unwrap_or(0);

            if user_version == 0 {
                let schema = include_str!("../../schema.sql");
                tx.execute_batch(schema)
                    .context("Failed to initialize database schema")?;
                tx.execute("PRAGMA user_version = 3", [])
                    .context("Failed to set user_version")?;
                tx.commit()
                    .context("Failed to commit schema initialization")?;
            } else if user_version == 1 {
                // Migration v1 → v2: add entity tables for existing users
                tx.execute_batch(
                    "CREATE TABLE IF NOT EXISTS entities (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        entity_type TEXT NOT NULL,
                        canonical_name TEXT NOT NULL,
                        aliases TEXT,
                        context TEXT,
                        metadata TEXT,
                        mention_count INTEGER DEFAULT 1,
                        first_seen_at DATETIME,
                        last_seen_at DATETIME,
                        created_at DATETIME NOT NULL,
                        updated_at DATETIME NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS entity_mentions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        entity_id INTEGER,
                        recording_id INTEGER NOT NULL,
                        mention_text TEXT NOT NULL,
                        context_snippet TEXT,
                        confidence REAL DEFAULT 1.0,
                        linked_at DATETIME,
                        created_at DATETIME NOT NULL,
                        FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE SET NULL,
                        FOREIGN KEY (recording_id) REFERENCES recordings(id) ON DELETE CASCADE
                    );

                    CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
                    CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(canonical_name);
                    CREATE INDEX IF NOT EXISTS idx_entity_mentions_entity ON entity_mentions(entity_id);
                    CREATE INDEX IF NOT EXISTS idx_entity_mentions_recording ON entity_mentions(recording_id);",
                )
                .context("Failed to migrate database to v2 (entity tables)")?;
                tx.execute("PRAGMA user_version = 2", [])
                    .context("Failed to set user_version")?;
                tx.commit()
                    .context("Failed to commit v2 migration")?;
            } else {
                // No work for this transaction — drop cleanly
                tx.commit().ok();
            }
        }

        // Migration v2 → v3: add diarization columns (speakers, segments)
        {
            let user_version: i64 = self
                .conn
                .query_row("PRAGMA user_version", [], |row| row.get(0))
                .unwrap_or(0);

            if user_version == 2 {
                // Check which columns already exist (fresh v0.18 installs
                // have them from schema.sql despite user_version being 2)
                let has_speakers: bool = self
                    .conn
                    .prepare("SELECT speakers FROM recordings LIMIT 0")
                    .is_ok();
                let has_segments: bool = self
                    .conn
                    .prepare("SELECT segments FROM transcripts LIMIT 0")
                    .is_ok();

                let tx = self
                    .conn
                    .transaction()
                    .context("Failed to start v3 migration transaction")?;
                if !has_speakers {
                    tx.execute_batch("ALTER TABLE recordings ADD COLUMN speakers TEXT;")
                        .context("Failed to add speakers column")?;
                }
                if !has_segments {
                    tx.execute_batch("ALTER TABLE transcripts ADD COLUMN segments TEXT;")
                        .context("Failed to add segments column")?;
                }
                tx.execute("PRAGMA user_version = 3", [])
                    .context("Failed to set user_version")?;
                tx.commit()
                    .context("Failed to commit v3 migration")?;
            }
        }

        let fk_on: i64 = self
            .conn
            .query_row("PRAGMA foreign_keys", [], |row| row.get(0))
            .unwrap_or(0);
        if fk_on != 1 {
            return Err(anyhow!(
                "Foreign keys are not enabled on this SQLite connection"
            ));
        }

        let skip_integrity = std::env::var("SCRIBA_SKIP_INTEGRITY").is_ok()
            || std::env::var("SCRIBA_MCP_MODE").is_ok();
        if !skip_integrity {
            let ok: bool = self
                .conn
                .prepare("PRAGMA integrity_check")
                .and_then(|mut stmt| {
                    stmt.query_row([], |row| {
                        let r: String = row.get(0)?;
                        Ok(r == "ok")
                    })
                })
                .unwrap_or(true);
            if !ok {
                if self.rebuild_transcripts_fts().is_ok() {
                    let _ = self.conn.execute("PRAGMA optimize", []);
                }
            }
        }
        Ok(())
    }

    // =========================================================================
    // Recording Operations
    // =========================================================================

    pub fn insert_recording(&mut self, recording: &Recording) -> Result<i64> {
        let sql = r#"
            INSERT INTO recordings (
                directory_name, display_name, created_at, updated_at,
                duration_seconds, file_size_bytes, audio_format, sample_rate,
                channels, has_transcript, transcript_status, language_code,
                model_used, audio_path
            ) VALUES (
                ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14
            )
        "#;

        self.conn.execute(
            sql,
            params![
                recording.directory_name,
                recording.display_name,
                recording.created_at,
                recording.updated_at,
                recording.duration_seconds,
                recording.file_size_bytes,
                recording.audio_format,
                recording.sample_rate,
                recording.channels,
                recording.has_transcript,
                recording.transcript_status,
                recording.language_code,
                recording.model_used,
                recording.audio_path,
            ],
        )?;

        Ok(self.conn.last_insert_rowid())
    }

    pub fn get_recording(&self, id: i64) -> Result<Option<Recording>> {
        let mut stmt = self
            .conn
            .prepare("SELECT * FROM recordings WHERE id = ?1")?;

        let mut rows = stmt.query_map([id], row_to_recording)?;

        match rows.next() {
            Some(row) => Ok(Some(row?)),
            None => Ok(None),
        }
    }

    pub fn get_recording_by_directory(&self, directory_name: &str) -> Result<Option<Recording>> {
        let mut stmt = self
            .conn
            .prepare("SELECT * FROM recordings WHERE directory_name = ?1")?;

        let mut rows = stmt.query_map([directory_name], row_to_recording)?;

        match rows.next() {
            Some(row) => Ok(Some(row?)),
            None => Ok(None),
        }
    }

    pub fn list_recordings(
        &self,
        limit: Option<i64>,
        offset: Option<i64>,
    ) -> Result<Vec<Recording>> {
        self.list_recordings_filtered(limit, offset, None, None)
    }

    pub fn list_recordings_filtered(
        &self,
        limit: Option<i64>,
        offset: Option<i64>,
        from_date: Option<&str>,
        to_date: Option<&str>,
    ) -> Result<Vec<Recording>> {
        let mut sql = String::from("SELECT * FROM recordings");
        let mut conditions: Vec<String> = Vec::new();
        let mut params_vec: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

        if let Some(from) = from_date {
            conditions.push(format!("created_at >= ?{}", params_vec.len() + 1));
            params_vec.push(Box::new(format!("{} 00:00:00", from)));
        }
        if let Some(to) = to_date {
            conditions.push(format!("created_at < ?{}", params_vec.len() + 1));
            // Add one day to make to_date inclusive
            if let Ok(d) = chrono::NaiveDate::parse_from_str(to, "%Y-%m-%d") {
                let next_day = d + chrono::Duration::days(1);
                params_vec.push(Box::new(format!("{} 00:00:00", next_day)));
            } else {
                params_vec.push(Box::new(format!("{} 23:59:59", to)));
            }
        }

        if !conditions.is_empty() {
            sql.push_str(" WHERE ");
            sql.push_str(&conditions.join(" AND "));
        }
        sql.push_str(" ORDER BY created_at DESC");
        if let Some(l) = limit {
            sql.push_str(&format!(" LIMIT {}", l));
        }
        if let Some(o) = offset {
            sql.push_str(&format!(" OFFSET {}", o));
        }

        let mut stmt = self.conn.prepare(&sql)?;
        let param_refs: Vec<&dyn rusqlite::types::ToSql> = params_vec.iter().map(|p| p.as_ref()).collect();
        let rows = stmt.query_map(param_refs.as_slice(), row_to_recording)?;

        let mut recordings = Vec::new();
        for row in rows {
            recordings.push(row?);
        }
        Ok(recordings)
    }

    pub fn update_recording_transcript_status(
        &mut self,
        id: i64,
        status: &str,
        has_transcript: bool,
    ) -> Result<()> {
        let sql = "UPDATE recordings SET transcript_status = ?1, has_transcript = ?2, updated_at = ?3 WHERE id = ?4";
        self.conn
            .execute(sql, params![status, has_transcript, Utc::now(), id])?;
        Ok(())
    }

    pub fn update_recording_transcript_status_and_model(
        &mut self,
        id: i64,
        status: &str,
        has_transcript: bool,
        model_used: &str,
    ) -> Result<()> {
        let sql = "UPDATE recordings SET transcript_status = ?1, has_transcript = ?2, model_used = ?3, updated_at = ?4 WHERE id = ?5";
        self.conn.execute(
            sql,
            params![status, has_transcript, model_used, Utc::now(), id],
        )?;
        Ok(())
    }

    pub fn delete_recording(&mut self, id: i64) -> Result<()> {
        let check_sql = "SELECT COUNT(*) FROM recordings WHERE id = ?1";
        let count: i64 = self
            .conn
            .query_row(check_sql, params![id], |row| row.get(0))
            .context("Failed to check if recording exists")?;
        if count == 0 {
            return Err(anyhow!("No recording found with id {}", id));
        }

        match self.try_parent_delete_with_cascade(id) {
            Ok(()) => Ok(()),
            Err(e) => {
                if self.rebuild_transcripts_fts().is_err() {
                    return Err(e);
                }
                self.try_parent_delete_with_cascade(id)
            }
        }
    }

    fn try_parent_delete_with_cascade(&mut self, id: i64) -> Result<()> {
        let tx = self
            .conn
            .transaction()
            .context("Failed to start transaction")?;

        tx.execute("PRAGMA foreign_keys = ON", [])?;

        tx.execute("DELETE FROM recordings WHERE id = ?1", params![id])
            .context("Failed to delete recording")?;

        {
            let mut stmt = tx.prepare("PRAGMA foreign_key_check")?;
            let violations: Vec<String> = stmt
                .query_map([], |row| {
                    let table: String = row.get(0)?;
                    let rowid: i64 = row.get(1)?;
                    let parent: String = row.get(2)?;
                    let fkid: i64 = row.get(3)?;
                    Ok(format!(
                        "FK violation: table={}, rowid={}, parent={}, fkid={}",
                        table, rowid, parent, fkid
                    ))
                })?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            if !violations.is_empty() {
                return Err(anyhow!("Foreign key violations detected after delete"));
            }
        }

        tx.commit().context("Failed to commit transaction")?;
        Ok(())
    }

    fn rebuild_transcripts_fts(&mut self) -> Result<()> {
        let tx = self
            .conn
            .transaction()
            .context("Failed to start rebuild transaction")?;

        tx.execute_batch(
            r#"
            DROP TRIGGER IF EXISTS transcripts_fts_insert;
            DROP TRIGGER IF EXISTS transcripts_fts_delete;
            DROP TRIGGER IF EXISTS transcripts_fts_update;
            DROP TABLE IF EXISTS transcripts_fts;

            CREATE VIRTUAL TABLE transcripts_fts USING fts5(
                content,
                recording_id UNINDEXED,
                content='transcripts',
                content_rowid='id'
            );

            CREATE TRIGGER transcripts_fts_insert AFTER INSERT ON transcripts BEGIN
                INSERT INTO transcripts_fts(rowid, content, recording_id)
                VALUES (new.id, new.content, new.recording_id);
            END;

            CREATE TRIGGER transcripts_fts_delete AFTER DELETE ON transcripts BEGIN
                INSERT INTO transcripts_fts(transcripts_fts, rowid, content, recording_id)
                VALUES('delete', old.id, old.content, old.recording_id);
            END;

            CREATE TRIGGER transcripts_fts_update AFTER UPDATE ON transcripts BEGIN
                INSERT INTO transcripts_fts(transcripts_fts, rowid, content, recording_id)
                VALUES('delete', old.id, old.content, old.recording_id);
                INSERT INTO transcripts_fts(rowid, content, recording_id)
                VALUES (new.id, new.content, new.recording_id);
            END;
            "#,
        )
        .context("Failed to recreate transcripts_fts and triggers")?;

        tx.execute(
            "INSERT INTO transcripts_fts(rowid, content, recording_id)
             SELECT id, content, recording_id FROM transcripts",
            [],
        )
        .context("Failed to repopulate transcripts_fts from transcripts")?;

        tx.commit()
            .context("Failed to commit rebuild transaction")?;
        Ok(())
    }

    // =========================================================================
    // Transcript Operations
    // =========================================================================

    pub fn insert_transcript(&mut self, transcript: &Transcript) -> Result<i64> {
        let sql = r#"
            INSERT INTO transcripts (
                recording_id, content, created_at, updated_at,
                word_count, character_count, language_detected
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
        "#;

        let word_count = transcript.content.split_whitespace().count() as i64;
        let character_count = transcript.content.chars().count() as i64;

        self.conn.execute(
            sql,
            params![
                transcript.recording_id,
                transcript.content,
                transcript.created_at,
                transcript.updated_at,
                word_count,
                character_count,
                transcript.language_detected,
            ],
        )?;

        Ok(self.conn.last_insert_rowid())
    }

    pub fn update_transcript(&mut self, recording_id: i64, new_content: &str) -> Result<()> {
        let sql = r#"
            UPDATE transcripts
            SET content = ?1, updated_at = ?2, word_count = ?3, character_count = ?4
            WHERE recording_id = ?5
        "#;

        let word_count = new_content.split_whitespace().count() as i64;
        let character_count = new_content.chars().count() as i64;

        self.conn.execute(
            sql,
            params![
                new_content,
                Utc::now(),
                word_count,
                character_count,
                recording_id,
            ],
        )?;

        Ok(())
    }

    pub fn upsert_transcript(&mut self, recording_id: i64, content: &str) -> Result<()> {
        if self.get_transcript_by_recording_id(recording_id)?.is_some() {
            self.update_transcript(recording_id, content)
        } else {
            let transcript = Transcript {
                id: None,
                recording_id,
                content: content.to_string(),
                created_at: Utc::now(),
                updated_at: Utc::now(),
                word_count: None,
                character_count: None,
                language_detected: None,
                confidence_scores: None,
                segments: None,
                entities: None,
                topics: None,
            };
            self.insert_transcript(&transcript)?;
            Ok(())
        }
    }

    pub fn get_transcript_by_recording_id(&self, recording_id: i64) -> Result<Option<Transcript>> {
        let sql = "SELECT * FROM transcripts WHERE recording_id = ?1";

        let mut stmt = self.conn.prepare(sql)?;
        let mut rows = stmt.query_map([recording_id], row_to_transcript)?;

        match rows.next() {
            Some(row) => Ok(Some(row?)),
            None => Ok(None),
        }
    }

    // =========================================================================
    // Search Operations
    // =========================================================================

    pub fn search_transcripts(
        &self,
        query: &str,
        limit: Option<i64>,
    ) -> Result<Vec<(Recording, Transcript)>> {
        self.search_transcripts_filtered(query, limit, None, None)
    }

    pub fn search_transcripts_filtered(
        &self,
        query: &str,
        limit: Option<i64>,
        from_date: Option<&str>,
        to_date: Option<&str>,
    ) -> Result<Vec<(Recording, Transcript)>> {
        let mut sql = String::from(
            r#"SELECT r.*, t.* FROM recordings r
               JOIN transcripts t ON r.id = t.recording_id
               JOIN transcripts_fts ON transcripts_fts.rowid = t.id
               WHERE transcripts_fts MATCH ?1"#,
        );
        let mut params_vec: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
        params_vec.push(Box::new(query.to_string()));

        if let Some(from) = from_date {
            sql.push_str(&format!(" AND r.created_at >= ?{}", params_vec.len() + 1));
            params_vec.push(Box::new(format!("{} 00:00:00", from)));
        }
        if let Some(to) = to_date {
            sql.push_str(&format!(" AND r.created_at < ?{}", params_vec.len() + 1));
            if let Ok(d) = chrono::NaiveDate::parse_from_str(to, "%Y-%m-%d") {
                let next_day = d + chrono::Duration::days(1);
                params_vec.push(Box::new(format!("{} 00:00:00", next_day)));
            } else {
                params_vec.push(Box::new(format!("{} 23:59:59", to)));
            }
        }

        sql.push_str(" ORDER BY rank");
        if let Some(l) = limit {
            sql.push_str(&format!(" LIMIT {}", l));
        }

        let mut stmt = self.conn.prepare(&sql)?;
        let param_refs: Vec<&dyn rusqlite::types::ToSql> = params_vec.iter().map(|p| p.as_ref()).collect();
        let rows = stmt.query_map(param_refs.as_slice(), row_to_recording_and_transcript)?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    // =========================================================================
    // Statistics
    // =========================================================================

    pub fn get_stats(&self) -> Result<RecordingStats> {
        let sql = r#"
            SELECT
                COUNT(*) as total_recordings,
                SUM(duration_seconds) as total_duration,
                SUM(file_size_bytes) as total_size,
                COUNT(CASE WHEN has_transcript = 1 THEN 1 END) as transcribed_count,
                SUM(CASE WHEN t.word_count IS NOT NULL THEN t.word_count ELSE 0 END) as total_words
            FROM recordings r
            LEFT JOIN transcripts t ON r.id = t.recording_id
        "#;

        let mut stmt = self.conn.prepare(sql)?;
        let row = stmt.query_row([], |row| {
            Ok(RecordingStats {
                total_recordings: row.get(0)?,
                total_duration_seconds: row.get::<_, Option<i64>>(1)?.unwrap_or(0),
                total_size_bytes: row.get::<_, Option<i64>>(2)?.unwrap_or(0),
                transcribed_count: row.get(3)?,
                total_words: row.get::<_, Option<i64>>(4)?.unwrap_or(0),
            })
        })?;

        Ok(row)
    }

    // =========================================================================
    // Maintenance
    // =========================================================================

    pub fn check_integrity(&self) -> Result<bool> {
        let mut stmt = self.conn.prepare("PRAGMA integrity_check")?;
        let result = stmt.query_row([], |row| {
            let check_result: String = row.get(0)?;
            Ok(check_result == "ok")
        })?;
        Ok(result)
    }

    pub fn vacuum(&mut self) -> Result<()> {
        self.conn.execute("VACUUM", [])?;
        Ok(())
    }

    pub fn reset_database() -> Result<()> {
        let db_path = Self::get_database_path()?;

        if db_path.exists() {
            std::fs::remove_file(&db_path).context("Failed to remove corrupted database file")?;
        }

        let wal_path = db_path.with_extension("db-wal");
        let shm_path = db_path.with_extension("db-shm");

        if wal_path.exists() {
            std::fs::remove_file(&wal_path).ok();
        }

        if shm_path.exists() {
            std::fs::remove_file(&shm_path).ok();
        }

        Ok(())
    }

    // =========================================================================
    // Entity Operations
    // =========================================================================

    /// Insert a new entity into the database.
    pub fn insert_entity(&mut self, entity: &Entity) -> Result<i64> {
        let sql = r#"
            INSERT INTO entities (
                entity_type, canonical_name, aliases, context, metadata,
                mention_count, first_seen_at, last_seen_at, created_at, updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
        "#;

        self.conn.execute(
            sql,
            params![
                entity.entity_type,
                entity.canonical_name,
                entity.aliases,
                entity.context,
                entity.metadata,
                entity.mention_count,
                entity.first_seen_at,
                entity.last_seen_at,
                entity.created_at,
                entity.updated_at,
            ],
        )?;

        Ok(self.conn.last_insert_rowid())
    }

    /// Get an entity by ID.
    pub fn get_entity(&self, id: i64) -> Result<Option<Entity>> {
        let sql = "SELECT * FROM entities WHERE id = ?1";
        let mut stmt = self.conn.prepare(sql)?;

        let mut rows = stmt.query_map([id], |row| {
            Ok(Entity {
                id: Some(row.get(0)?),
                entity_type: row.get(1)?,
                canonical_name: row.get(2)?,
                aliases: row.get(3)?,
                context: row.get(4)?,
                metadata: row.get(5)?,
                mention_count: row.get(6)?,
                first_seen_at: row.get(7)?,
                last_seen_at: row.get(8)?,
                created_at: row.get(9)?,
                updated_at: row.get(10)?,
            })
        })?;

        match rows.next() {
            Some(row) => Ok(Some(row?)),
            None => Ok(None),
        }
    }

    /// Get an entity by canonical name (case-insensitive).
    pub fn get_entity_by_name(&self, name: &str) -> Result<Option<Entity>> {
        let sql = "SELECT * FROM entities WHERE LOWER(canonical_name) = LOWER(?1)";
        let mut stmt = self.conn.prepare(sql)?;

        let mut rows = stmt.query_map([name], |row| {
            Ok(Entity {
                id: Some(row.get(0)?),
                entity_type: row.get(1)?,
                canonical_name: row.get(2)?,
                aliases: row.get(3)?,
                context: row.get(4)?,
                metadata: row.get(5)?,
                mention_count: row.get(6)?,
                first_seen_at: row.get(7)?,
                last_seen_at: row.get(8)?,
                created_at: row.get(9)?,
                updated_at: row.get(10)?,
            })
        })?;

        match rows.next() {
            Some(row) => Ok(Some(row?)),
            None => Ok(None),
        }
    }

    /// Get an entity by canonical name OR alias (case-insensitive).
    ///
    /// First checks canonical_name, then scans aliases JSON arrays.
    /// This prevents recreating entities that were merged (where the old
    /// name became an alias of the target entity).
    pub fn get_entity_by_name_or_alias(&self, name: &str) -> Result<Option<Entity>> {
        // First try canonical name (fast path)
        if let Some(entity) = self.get_entity_by_name(name)? {
            return Ok(Some(entity));
        }

        // Scan all entities for alias match
        let name_lower = name.to_lowercase();
        let all = self.list_entities(None, None)?;
        for entity in all {
            for alias in entity.aliases_list() {
                if alias.to_lowercase() == name_lower {
                    return Ok(Some(entity));
                }
            }
        }

        Ok(None)
    }

    /// List all entities, optionally filtered by type.
    pub fn list_entities(&self, entity_type: Option<&str>, limit: Option<i64>) -> Result<Vec<Entity>> {
        let sql = match (entity_type, limit) {
            (Some(t), Some(l)) => format!(
                "SELECT * FROM entities WHERE entity_type = '{}' ORDER BY mention_count DESC, canonical_name LIMIT {}",
                t, l
            ),
            (Some(t), None) => format!(
                "SELECT * FROM entities WHERE entity_type = '{}' ORDER BY mention_count DESC, canonical_name",
                t
            ),
            (None, Some(l)) => format!(
                "SELECT * FROM entities ORDER BY mention_count DESC, canonical_name LIMIT {}",
                l
            ),
            (None, None) => "SELECT * FROM entities ORDER BY mention_count DESC, canonical_name".to_string(),
        };

        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map([], |row| {
            Ok(Entity {
                id: Some(row.get(0)?),
                entity_type: row.get(1)?,
                canonical_name: row.get(2)?,
                aliases: row.get(3)?,
                context: row.get(4)?,
                metadata: row.get(5)?,
                mention_count: row.get(6)?,
                first_seen_at: row.get(7)?,
                last_seen_at: row.get(8)?,
                created_at: row.get(9)?,
                updated_at: row.get(10)?,
            })
        })?;

        let mut entities = Vec::new();
        for row in rows {
            entities.push(row?);
        }
        Ok(entities)
    }

    /// Update an entity.
    pub fn update_entity(&mut self, entity: &Entity) -> Result<()> {
        let sql = r#"
            UPDATE entities SET
                entity_type = ?1, canonical_name = ?2, aliases = ?3, context = ?4,
                metadata = ?5, mention_count = ?6, first_seen_at = ?7, last_seen_at = ?8,
                updated_at = ?9
            WHERE id = ?10
        "#;

        self.conn.execute(
            sql,
            params![
                entity.entity_type,
                entity.canonical_name,
                entity.aliases,
                entity.context,
                entity.metadata,
                entity.mention_count,
                entity.first_seen_at,
                entity.last_seen_at,
                Utc::now(),
                entity.id,
            ],
        )?;

        Ok(())
    }

    /// Delete an entity.
    pub fn delete_entity(&mut self, id: i64) -> Result<()> {
        self.conn
            .execute("DELETE FROM entities WHERE id = ?1", params![id])?;
        Ok(())
    }

    /// Increment entity mention count and update last_seen_at.
    pub fn increment_entity_mention(&mut self, id: i64) -> Result<()> {
        let sql = r#"
            UPDATE entities SET
                mention_count = mention_count + 1,
                last_seen_at = ?1,
                updated_at = ?1
            WHERE id = ?2
        "#;

        self.conn.execute(sql, params![Utc::now(), id])?;
        Ok(())
    }

    // =========================================================================
    // Entity Mention Operations
    // =========================================================================

    /// Insert a new entity mention.
    pub fn insert_entity_mention(&mut self, mention: &EntityMentionRecord) -> Result<i64> {
        let sql = r#"
            INSERT INTO entity_mentions (
                entity_id, recording_id, mention_text, context_snippet,
                confidence, linked_at, created_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
        "#;

        self.conn.execute(
            sql,
            params![
                mention.entity_id,
                mention.recording_id,
                mention.mention_text,
                mention.context_snippet,
                mention.confidence,
                mention.linked_at,
                mention.created_at,
            ],
        )?;

        Ok(self.conn.last_insert_rowid())
    }

    /// Get all mentions for a recording.
    pub fn get_mentions_for_recording(&self, recording_id: i64) -> Result<Vec<EntityMentionRecord>> {
        let sql = "SELECT * FROM entity_mentions WHERE recording_id = ?1 ORDER BY id";
        let mut stmt = self.conn.prepare(sql)?;

        let rows = stmt.query_map([recording_id], |row| {
            Ok(EntityMentionRecord {
                id: Some(row.get(0)?),
                entity_id: row.get(1)?,
                recording_id: row.get(2)?,
                mention_text: row.get(3)?,
                context_snippet: row.get(4)?,
                confidence: row.get(5)?,
                linked_at: row.get(6)?,
                created_at: row.get(7)?,
            })
        })?;

        let mut mentions = Vec::new();
        for row in rows {
            mentions.push(row?);
        }
        Ok(mentions)
    }

    /// Get all mentions for an entity.
    pub fn get_mentions_for_entity(&self, entity_id: i64) -> Result<Vec<EntityMentionRecord>> {
        let sql = "SELECT * FROM entity_mentions WHERE entity_id = ?1 ORDER BY created_at DESC";
        let mut stmt = self.conn.prepare(sql)?;

        let rows = stmt.query_map([entity_id], |row| {
            Ok(EntityMentionRecord {
                id: Some(row.get(0)?),
                entity_id: row.get(1)?,
                recording_id: row.get(2)?,
                mention_text: row.get(3)?,
                context_snippet: row.get(4)?,
                confidence: row.get(5)?,
                linked_at: row.get(6)?,
                created_at: row.get(7)?,
            })
        })?;

        let mut mentions = Vec::new();
        for row in rows {
            mentions.push(row?);
        }
        Ok(mentions)
    }

    /// Get unlinked mentions (entity_id is NULL).
    pub fn get_unlinked_mentions(&self, limit: Option<i64>) -> Result<Vec<EntityMentionRecord>> {
        let sql = match limit {
            Some(l) => format!(
                "SELECT * FROM entity_mentions WHERE entity_id IS NULL ORDER BY created_at DESC LIMIT {}",
                l
            ),
            None => "SELECT * FROM entity_mentions WHERE entity_id IS NULL ORDER BY created_at DESC".to_string(),
        };

        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map([], |row| {
            Ok(EntityMentionRecord {
                id: Some(row.get(0)?),
                entity_id: row.get(1)?,
                recording_id: row.get(2)?,
                mention_text: row.get(3)?,
                context_snippet: row.get(4)?,
                confidence: row.get(5)?,
                linked_at: row.get(6)?,
                created_at: row.get(7)?,
            })
        })?;

        let mut mentions = Vec::new();
        for row in rows {
            mentions.push(row?);
        }
        Ok(mentions)
    }

    /// Link a mention to an entity.
    pub fn link_mention_to_entity(
        &mut self,
        mention_id: i64,
        entity_id: i64,
        confidence: f64,
    ) -> Result<()> {
        let sql = r#"
            UPDATE entity_mentions SET
                entity_id = ?1, confidence = ?2, linked_at = ?3
            WHERE id = ?4
        "#;

        self.conn
            .execute(sql, params![entity_id, confidence, Utc::now(), mention_id])?;
        Ok(())
    }

    /// Unlink a mention from its entity.
    pub fn unlink_mention(&mut self, mention_id: i64) -> Result<()> {
        let sql = "UPDATE entity_mentions SET entity_id = NULL, linked_at = NULL WHERE id = ?1";
        self.conn.execute(sql, params![mention_id])?;
        Ok(())
    }

    /// Reassign all mentions from one entity to another.
    /// Used when merging entities.
    pub fn reassign_mentions(&mut self, from_entity_id: i64, to_entity_id: i64) -> Result<usize> {
        let sql = r#"
            UPDATE entity_mentions SET
                entity_id = ?1,
                linked_at = ?2
            WHERE entity_id = ?3
        "#;

        let count = self
            .conn
            .execute(sql, params![to_entity_id, Utc::now(), from_entity_id])?;
        Ok(count)
    }

    /// Get recordings that mention a specific entity.
    pub fn get_recordings_for_entity(&self, entity_id: i64) -> Result<Vec<Recording>> {
        let sql = r#"
            SELECT DISTINCT r.* FROM recordings r
            JOIN entity_mentions em ON r.id = em.recording_id
            WHERE em.entity_id = ?1
            ORDER BY r.created_at DESC
        "#;

        let mut stmt = self.conn.prepare(sql)?;
        let rows = stmt.query_map([entity_id], row_to_recording)?;

        let mut recordings = Vec::new();
        for row in rows {
            recordings.push(row?);
        }
        Ok(recordings)
    }

    // =========================================================================
    // Enrichment Updates
    // =========================================================================

    /// Update recording with enrichment data (summary, key_points, action_items).
    pub fn update_recording_enrichment(
        &mut self,
        recording_id: i64,
        display_name: Option<&str>,
        summary: Option<&str>,
        key_points: Option<&str>,
        action_items: Option<&str>,
    ) -> Result<()> {
        let sql = r#"
            UPDATE recordings SET
                display_name = COALESCE(?1, display_name),
                summary = ?2,
                key_points = ?3,
                action_items = ?4,
                updated_at = ?5
            WHERE id = ?6
        "#;

        self.conn.execute(
            sql,
            params![display_name, summary, key_points, action_items, Utc::now(), recording_id],
        )?;

        Ok(())
    }

    /// Update transcript with enrichment data (entities, topics).
    pub fn update_transcript_enrichment(
        &mut self,
        recording_id: i64,
        entities: Option<&str>,
        topics: Option<&str>,
    ) -> Result<()> {
        let sql = r#"
            UPDATE transcripts SET
                entities = ?1,
                topics = ?2,
                updated_at = ?3
            WHERE recording_id = ?4
        "#;

        self.conn
            .execute(sql, params![entities, topics, Utc::now(), recording_id])?;

        Ok(())
    }

    /// Update transcript with diarization segments JSON.
    pub fn update_transcript_segments(
        &mut self,
        recording_id: i64,
        segments_json: &str,
    ) -> Result<()> {
        let sql = r#"
            UPDATE transcripts SET
                segments = ?1,
                updated_at = ?2
            WHERE recording_id = ?3
        "#;

        self.conn
            .execute(sql, params![segments_json, Utc::now(), recording_id])?;

        Ok(())
    }

    /// Update recording with speakers JSON.
    pub fn update_recording_speakers(
        &mut self,
        recording_id: i64,
        speakers_json: &str,
    ) -> Result<()> {
        let sql = r#"
            UPDATE recordings SET
                speakers = ?1,
                updated_at = ?2
            WHERE id = ?3
        "#;

        self.conn
            .execute(sql, params![speakers_json, Utc::now(), recording_id])?;

        Ok(())
    }
}
