use anyhow::{anyhow, Context, Result};
use chrono::{DateTime, Utc};
use dirs::home_dir;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recording {
    pub id: Option<i64>,
    pub directory_name: String,
    pub display_name: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub duration_seconds: Option<i64>,
    pub file_size_bytes: Option<i64>,
    pub audio_format: String,
    pub sample_rate: i64,
    pub channels: i64,
    pub has_transcript: bool,
    pub transcript_status: String,
    pub language_code: String,
    pub model_used: String,
    pub tags: Option<String>, // JSON
    pub summary: Option<String>,
    pub key_points: Option<String>,   // JSON
    pub action_items: Option<String>, // JSON
    pub speakers: Option<String>,     // JSON
    pub sentiment_score: Option<f64>,
    pub search_index: Option<String>,
    pub categories: Option<String>, // JSON
    pub confidence_score: Option<f64>,
    pub audio_path: String,
    pub transcript_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transcript {
    pub id: Option<i64>,
    pub recording_id: i64,
    pub content: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub word_count: Option<i64>,
    pub character_count: Option<i64>,
    pub language_detected: Option<String>,
    pub confidence_scores: Option<String>, // JSON
    pub segments: Option<String>,          // JSON
    pub entities: Option<String>,          // JSON
    pub topics: Option<String>,            // JSON
}

#[derive(Debug, Clone)]
pub struct Tag {
    pub id: Option<i64>,
    pub name: String,
    pub color: String,
    pub created_at: DateTime<Utc>,
    pub usage_count: i64,
}

pub struct Database {
    conn: Connection,
}

impl Database {
    pub fn new() -> Result<Self> {
        let db_path = Self::get_database_path()?;

        // Ensure the directory exists
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let conn = Connection::open(&db_path).context("Failed to open database connection")?;

        let mut db = Database { conn };

        if let Err(_e) = db.initialize() {
            drop(db);
            Self::reset_database()?;
            let conn = Connection::open(&db_path)
                .context("Failed to open database connection after reset")?;
            let mut db = Database { conn };
            db.initialize()
                .context("Failed to initialize fresh database")?;
            return Ok(db);
        }

        Ok(db)
    }

    fn get_database_path() -> Result<PathBuf> {
        let home = home_dir().context("Could not find home directory")?;
        Ok(home.join("scriba_recordings").join("scriba.db"))
    }

    fn initialize(&mut self) -> Result<()> {
        // Enable foreign key constraints (essential)
        self.conn
            .execute("PRAGMA foreign_keys = ON", [])
            .context("Failed to enable foreign key constraints")?;

        // Try to enable WAL mode for better concurrency, but don't fail if not supported
        if self.conn.execute("PRAGMA journal_mode = WAL", []).is_err() {
            // Fallback to default if not supported
        }

        // Busy timeout to reduce SQLITE_BUSY errors
        let _ = self.conn.execute("PRAGMA busy_timeout = 5000", []);

        // Apply the full schema only once to avoid dropping/recreating virtual tables on each open
        {
            let tx = self
                .conn
                .transaction()
                .context("Failed to start schema transaction")?;

            let user_version: i64 = tx
                .query_row("PRAGMA user_version", [], |row| row.get(0))
                .unwrap_or(0);

            if user_version == 0 {
                let schema = include_str!("../schema.sql");
                tx.execute_batch(schema)
                    .context("Failed to initialize database schema")?;
                // Mark as initialized
                tx.execute("PRAGMA user_version = 1", [])
                    .context("Failed to set user_version")?;
                tx.commit().context("Failed to commit schema initialization")?;
            } else {
                tx.commit().ok();
            }
        }

        // Verify that FK is active on this connection
        let fk_on: i64 = self
            .conn
            .query_row("PRAGMA foreign_keys", [], |row| row.get(0))
            .unwrap_or(0);
        if fk_on != 1 {
            return Err(anyhow!(
                "Foreign keys are not enabled on this SQLite connection"
            ));
        }

        // Attempt a quick integrity check; if it fails, try to rebuild FTS index once
        let ok: bool = self
            .conn
            .prepare("PRAGMA integrity_check")
            .and_then(|mut stmt| stmt.query_row([], |row| {
                let r: String = row.get(0)?;
                Ok(r == "ok")
            }))
            .unwrap_or(true); // don't block initialization if pragma not available
        if !ok {
            // Try to rebuild FTS as this is a common source of corruption-like errors
            if self.rebuild_transcripts_fts().is_ok() {
                let _ = self.conn.execute("PRAGMA optimize", []);
            }
        }
        Ok(())
    }

    // Recording operations
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

        let _result = self.conn.execute(
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

    pub fn get_recording_by_directory(&self, directory_name: &str) -> Result<Option<Recording>> {
        let sql = "SELECT * FROM recordings WHERE directory_name = ?1";

        let mut stmt = self.conn.prepare(sql)?;
        let mut rows = stmt.query_map([directory_name], |row| {
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
        })?;

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
        let sql = match (limit, offset) {
            (Some(l), Some(o)) => format!(
                "SELECT * FROM recordings ORDER BY created_at DESC LIMIT {} OFFSET {}",
                l, o
            ),
            (Some(l), None) => format!(
                "SELECT * FROM recordings ORDER BY created_at DESC LIMIT {}",
                l
            ),
            _ => "SELECT * FROM recordings ORDER BY created_at DESC".to_string(),
        };

        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map([], |row| {
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
        })?;

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
            Ok(()) => return Ok(()),
            Err(e) => {
                if let Err(_rebuild_err) = self.rebuild_transcripts_fts() {
                    return Err(e);
                }
                match self.try_parent_delete_with_cascade(id) {
                    Ok(()) => return Ok(()),
                    Err(e2) => {
                        return Err(e2);
                    }
                }
            }
        }
    }

    fn try_parent_delete_with_cascade(&mut self, id: i64) -> Result<()> {
        let tx = self
            .conn
            .transaction()
            .context("Failed to start transaction")?;

        tx.execute("PRAGMA foreign_keys = ON", [])?;

        let _rows_affected = tx
            .execute("DELETE FROM recordings WHERE id = ?1", params![id])
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

        let _inserted = tx
            .execute(
                "INSERT INTO transcripts_fts(rowid, content, recording_id)
                 SELECT id, content, recording_id FROM transcripts",
                [],
            )
            .context("Failed to repopulate transcripts_fts from transcripts")?;

        tx.commit()
            .context("Failed to commit rebuild transaction")?;
        Ok(())
    }

    // Transcript operations
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
        // Check if transcript already exists
        if self.get_transcript_by_recording_id(recording_id)?.is_some() {
            // Update existing transcript
            self.update_transcript(recording_id, content)
        } else {
            // Create new transcript
            let transcript = Transcript {
                id: None,
                recording_id,
                content: content.to_string(),
                created_at: Utc::now(),
                updated_at: Utc::now(),
                word_count: None,      // Will be calculated by the insert method
                character_count: None, // Will be calculated by the insert method
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
        let mut rows = stmt.query_map([recording_id], |row| {
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
        })?;

        match rows.next() {
            Some(row) => Ok(Some(row?)),
            None => Ok(None),
        }
    }

    // Search operations (for future AI features)
    pub fn search_transcripts(
        &self,
        query: &str,
        limit: Option<i64>,
    ) -> Result<Vec<(Recording, Transcript)>> {
        let sql = match limit {
            Some(l) => format!(
                r#"
                SELECT r.*, t.* FROM recordings r
                JOIN transcripts t ON r.id = t.recording_id
                JOIN transcripts_fts ON transcripts_fts.rowid = t.id
                WHERE transcripts_fts MATCH ?1
                ORDER BY rank
                LIMIT {}
            "#,
                l
            ),
            None => r#"
                SELECT r.*, t.* FROM recordings r
                JOIN transcripts t ON r.id = t.recording_id
                JOIN transcripts_fts ON transcripts_fts.rowid = t.id
                WHERE transcripts_fts MATCH ?1
                ORDER BY rank
            "#
            .to_string(),
        };

        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map([query], |row| {
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
        })?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    // Statistics for future dashboard
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

    // Database maintenance and diagnostics
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

    // Reset the database completely - useful for corruption recovery
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
}

#[derive(Debug)]
pub struct RecordingStats {
    pub total_recordings: i64,
    pub total_duration_seconds: i64,
    pub total_size_bytes: i64,
    pub transcribed_count: i64,
    pub total_words: i64,
}

impl RecordingStats {
    pub fn format_duration(&self) -> String {
        let hours = self.total_duration_seconds / 3600;
        let minutes = (self.total_duration_seconds % 3600) / 60;
        let seconds = self.total_duration_seconds % 60;

        if hours > 0 {
            format!("{}h {}m {}s", hours, minutes, seconds)
        } else if minutes > 0 {
            format!("{}m {}s", minutes, seconds)
        } else {
            format!("{}s", seconds)
        }
    }

    pub fn format_size(&self) -> String {
        const KB: i64 = 1024;
        const MB: i64 = KB * 1024;
        const GB: i64 = MB * 1024;

        if self.total_size_bytes >= GB {
            format!("{:.1} GB", self.total_size_bytes as f64 / GB as f64)
        } else if self.total_size_bytes >= MB {
            format!("{:.1} MB", self.total_size_bytes as f64 / MB as f64)
        } else if self.total_size_bytes >= KB {
            format!("{:.1} KB", self.total_size_bytes as f64 / KB as f64)
        } else {
            format!("{} bytes", self.total_size_bytes)
        }
    }
}
