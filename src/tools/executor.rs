//! Tool executor — dispatches tool calls to database/world queries.
//!
//! All 15 tools are implemented here. Consumers (agent, MCP) call
//! `execute_tool()` and wrap the result into their own wire format.

use crate::database::Database;
use crate::enrichment::WorldContext;
use crate::entities::EntityRegistry;
use serde_json::{json, Value};

use super::definitions::*;

/// The result of executing a tool.
pub struct ToolResult {
    /// JSON or plain-text output.
    pub output: String,
    /// True if the tool encountered an error.
    pub is_error: bool,
}

impl ToolResult {
    pub fn ok(output: String) -> Self {
        Self { output, is_error: false }
    }
    pub fn err(message: String) -> Self {
        Self { output: message, is_error: true }
    }
}

/// Execute a tool by name. The `db` must be `&mut` because write tools
/// need to create an `EntityRegistry`.
pub fn execute_tool(name: &str, input: &Value, db: &mut Database) -> ToolResult {
    match name {
        "list_recordings" => exec_list_recordings(input, db),
        "get_recording" => exec_get_recording(input, db),
        "get_transcript" => exec_get_transcript(input, db),
        "search_transcripts" => exec_search_transcripts(input, db),
        "list_entities" => exec_list_entities(input, db),
        "get_entity" => exec_get_entity(input, db),
        "get_recordings_for_entity" => exec_get_recordings_for_entity(input, db),
        "get_world_context" => exec_get_world_context(),
        "get_stats" => exec_get_stats(db),
        "create_entity" => exec_create_entity(input, db),
        "update_entity" => exec_update_entity(input, db),
        "add_entity_alias" => exec_add_entity_alias(input, db),
        "remove_entity_alias" => exec_remove_entity_alias(input, db),
        "merge_entities" => exec_merge_entities(input, db),
        "delete_entity" => exec_delete_entity(input, db),
        _ => ToolResult::err(format!("Unknown tool: {}", name)),
    }
}

// ─── Read Tools ──────────────────────────────────────────────────────────────

fn exec_list_recordings(input: &Value, db: &Database) -> ToolResult {
    let params: ListRecordingsParams = match serde_json::from_value(input.clone()) {
        Ok(p) => p,
        Err(e) => return ToolResult::err(format!("Invalid params: {}", e)),
    };
    match db.list_recordings_filtered(params.limit, params.offset, params.from_date.as_deref(), params.to_date.as_deref()) {
        Ok(recordings) => {
            let items: Vec<Value> = recordings
                .iter()
                .map(|r| {
                    json!({
                        "id": r.id,
                        "name": r.display_name.as_ref().unwrap_or(&r.directory_name),
                        "directory_name": r.directory_name,
                        "created_at": r.created_at.to_rfc3339(),
                        "duration_seconds": r.duration_seconds,
                        "has_transcript": r.has_transcript,
                        "summary": r.summary.as_deref().unwrap_or("").chars().take(200).collect::<String>(),
                    })
                })
                .collect();
            ToolResult::ok(serde_json::to_string_pretty(&items).unwrap_or_else(|_| "[]".to_string()))
        }
        Err(e) => ToolResult::err(format!("Error listing recordings: {}", e)),
    }
}

fn exec_get_recording(input: &Value, db: &Database) -> ToolResult {
    let params: GetRecordingParams = match serde_json::from_value(input.clone()) {
        Ok(p) => p,
        Err(e) => return ToolResult::err(format!("Invalid params: {}", e)),
    };

    let recording = if let Some(id) = params.id {
        match db.get_recording(id) {
            Ok(Some(r)) => r,
            Ok(None) => return ToolResult::err(format!("Recording {} not found", id)),
            Err(e) => return ToolResult::err(format!("Error: {}", e)),
        }
    } else if let Some(dir) = params.directory_name {
        match db.get_recording_by_directory(&dir) {
            Ok(Some(r)) => r,
            Ok(None) => return ToolResult::err(format!("Recording '{}' not found", dir)),
            Err(e) => return ToolResult::err(format!("Error: {}", e)),
        }
    } else {
        return ToolResult::err("Provide id or directory_name".to_string());
    };

    let result = json!({
        "id": recording.id,
        "name": recording.display_name.as_ref().unwrap_or(&recording.directory_name),
        "directory_name": recording.directory_name,
        "created_at": recording.created_at.to_rfc3339(),
        "duration_seconds": recording.duration_seconds,
        "has_transcript": recording.has_transcript,
        "language": recording.language_code,
        "summary": recording.summary,
        "key_points": recording.key_points,
        "action_items": recording.action_items,
        "speakers": recording.speakers,
        "topics": recording.categories,
        "tags": recording.tags,
    });
    ToolResult::ok(serde_json::to_string_pretty(&result).unwrap_or_default())
}

fn exec_get_transcript(input: &Value, db: &Database) -> ToolResult {
    let params: GetTranscriptParams = match serde_json::from_value(input.clone()) {
        Ok(p) => p,
        Err(e) => return ToolResult::err(format!("Invalid params: {}", e)),
    };

    let recording_id = if let Some(id) = params.recording_id {
        id
    } else if let Some(dir) = params.directory_name {
        match db.get_recording_by_directory(&dir) {
            Ok(Some(r)) => r.id.unwrap_or_default(),
            Ok(None) => return ToolResult::err("Recording not found".to_string()),
            Err(e) => return ToolResult::err(format!("Error: {}", e)),
        }
    } else {
        return ToolResult::err("Provide recording_id or directory_name".to_string());
    };

    match db.get_transcript_by_recording_id(recording_id) {
        Ok(Some(t)) => ToolResult::ok(t.content),
        Ok(None) => ToolResult::err(format!("No transcript found for recording {}", recording_id)),
        Err(e) => ToolResult::err(format!("Error: {}", e)),
    }
}

fn exec_search_transcripts(input: &Value, db: &Database) -> ToolResult {
    let params: SearchTranscriptsParams = match serde_json::from_value(input.clone()) {
        Ok(p) => p,
        Err(e) => return ToolResult::err(format!("Invalid params: {}", e)),
    };
    let limit = params.limit.or(Some(10));
    match db.search_transcripts_filtered(&params.query, limit, params.from_date.as_deref(), params.to_date.as_deref()) {
        Ok(results) => {
            let items: Vec<Value> = results
                .iter()
                .map(|(r, t)| {
                    let snippet = extract_snippet(&t.content, &params.query, 150);
                    json!({
                        "recording_id": r.id,
                        "recording_name": r.display_name.as_ref().unwrap_or(&r.directory_name),
                        "created_at": r.created_at.to_rfc3339(),
                        "word_count": t.word_count,
                        "snippet": snippet,
                    })
                })
                .collect();
            ToolResult::ok(serde_json::to_string_pretty(&items).unwrap_or_else(|_| "[]".to_string()))
        }
        Err(e) => ToolResult::err(format!("Error searching: {}", e)),
    }
}

fn exec_list_entities(input: &Value, db: &Database) -> ToolResult {
    let params: ListEntitiesParams = match serde_json::from_value(input.clone()) {
        Ok(p) => p,
        Err(e) => return ToolResult::err(format!("Invalid params: {}", e)),
    };
    match db.list_entities(params.entity_type.as_deref(), params.limit) {
        Ok(entities) => {
            let items: Vec<Value> = entities
                .iter()
                .map(|e| {
                    json!({
                        "id": e.id,
                        "type": e.entity_type,
                        "name": e.canonical_name,
                        "aliases": e.aliases_list(),
                        "context": e.context,
                        "mention_count": e.mention_count,
                    })
                })
                .collect();
            ToolResult::ok(serde_json::to_string_pretty(&items).unwrap_or_else(|_| "[]".to_string()))
        }
        Err(e) => ToolResult::err(format!("Error: {}", e)),
    }
}

fn exec_get_entity(input: &Value, db: &Database) -> ToolResult {
    let params: GetEntityParams = match serde_json::from_value(input.clone()) {
        Ok(p) => p,
        Err(e) => return ToolResult::err(format!("Invalid params: {}", e)),
    };

    let entity = if let Some(id) = params.id {
        match db.get_entity(id) {
            Ok(Some(e)) => e,
            Ok(None) => return ToolResult::err(format!("Entity {} not found", id)),
            Err(e) => return ToolResult::err(format!("Error: {}", e)),
        }
    } else if let Some(name) = params.name {
        match db.get_entity_by_name(&name) {
            Ok(Some(e)) => e,
            Ok(None) => return ToolResult::err(format!("Entity '{}' not found", name)),
            Err(e) => return ToolResult::err(format!("Error: {}", e)),
        }
    } else {
        return ToolResult::err("Provide id or name".to_string());
    };

    let entity_id = entity.id.unwrap_or_default();
    let mentions = db.get_mentions_for_entity(entity_id).unwrap_or_default();
    let recordings = db.get_recordings_for_entity(entity_id).unwrap_or_default();

    let result = json!({
        "id": entity.id,
        "type": entity.entity_type,
        "name": entity.canonical_name,
        "aliases": entity.aliases_list(),
        "context": entity.context,
        "mention_count": entity.mention_count,
        "mentions": mentions.iter().take(20).map(|m| {
            json!({
                "recording_id": m.recording_id,
                "mention_text": m.mention_text,
                "context_snippet": m.context_snippet,
            })
        }).collect::<Vec<_>>(),
        "recordings": recordings.iter().map(|r| {
            json!({
                "id": r.id,
                "name": r.display_name.as_ref().unwrap_or(&r.directory_name),
                "summary": r.summary.as_deref().unwrap_or("").chars().take(150).collect::<String>(),
            })
        }).collect::<Vec<_>>(),
    });
    ToolResult::ok(serde_json::to_string_pretty(&result).unwrap_or_default())
}

fn exec_get_recordings_for_entity(input: &Value, db: &Database) -> ToolResult {
    let params: GetRecordingsForEntityParams = match serde_json::from_value(input.clone()) {
        Ok(p) => p,
        Err(e) => return ToolResult::err(format!("Invalid params: {}", e)),
    };
    match db.get_recordings_for_entity(params.entity_id) {
        Ok(recordings) => {
            let items: Vec<Value> = recordings
                .iter()
                .map(|r| {
                    json!({
                        "id": r.id,
                        "name": r.display_name.as_ref().unwrap_or(&r.directory_name),
                        "created_at": r.created_at.to_rfc3339(),
                        "summary": r.summary.as_deref().unwrap_or("").chars().take(200).collect::<String>(),
                    })
                })
                .collect();
            ToolResult::ok(serde_json::to_string_pretty(&items).unwrap_or_else(|_| "[]".to_string()))
        }
        Err(e) => ToolResult::err(format!("Error: {}", e)),
    }
}

fn exec_get_world_context() -> ToolResult {
    match WorldContext::load() {
        Ok(wc) => {
            if wc.content.is_empty() {
                ToolResult::ok("(no world context)".to_string())
            } else {
                ToolResult::ok(wc.content)
            }
        }
        Err(e) => ToolResult::err(format!("Error loading world context: {}", e)),
    }
}

fn exec_get_stats(db: &Database) -> ToolResult {
    match db.get_stats() {
        Ok(stats) => {
            let entity_count = db.list_entities(None, None).map(|e| e.len()).unwrap_or(0);
            let result = json!({
                "total_recordings": stats.total_recordings,
                "total_duration": stats.format_duration(),
                "total_size": stats.format_size(),
                "transcribed_count": stats.transcribed_count,
                "total_words": stats.total_words,
                "total_entities": entity_count,
            });
            ToolResult::ok(serde_json::to_string_pretty(&result).unwrap_or_default())
        }
        Err(e) => ToolResult::err(format!("Error: {}", e)),
    }
}

// ─── Write Tools ─────────────────────────────────────────────────────────────

fn exec_create_entity(input: &Value, db: &mut Database) -> ToolResult {
    let params: CreateEntityParams = match serde_json::from_value(input.clone()) {
        Ok(p) => p,
        Err(e) => return ToolResult::err(format!("Invalid params: {}", e)),
    };

    // Validate entity_type
    if params.entity_type != "person" && params.entity_type != "organization" {
        return ToolResult::err(format!(
            "Invalid entity_type '{}'. Must be 'person' or 'organization'.",
            params.entity_type
        ));
    }

    let mut registry = EntityRegistry::new(db);

    // Check for duplicates by name or alias
    if let Ok(Some(existing)) = registry.get_entity_by_name_or_alias(&params.name) {
        return ToolResult::err(format!(
            "Entity '{}' already exists (id: {}, canonical name: '{}')",
            params.name,
            existing.id.unwrap_or_default(),
            existing.canonical_name
        ));
    }

    let entity = match registry.create_entity(
        &params.entity_type,
        &params.name,
        params.context.as_deref(),
    ) {
        Ok(e) => e,
        Err(e) => return ToolResult::err(format!("Error creating entity: {}", e)),
    };

    let entity_id = entity.id.unwrap_or_default();

    // Add aliases if provided
    if let Some(aliases) = &params.aliases {
        for alias in aliases {
            if let Err(e) = registry.add_entity_alias(entity_id, alias) {
                return ToolResult::err(format!("Entity created but failed to add alias '{}': {}", alias, e));
            }
        }
    }

    // Re-fetch to include aliases
    let final_entity = registry.get_entity(entity_id).ok().flatten().unwrap_or(entity);

    let result = json!({
        "success": true,
        "message": format!("Created {} '{}'", params.entity_type, params.name),
        "entity": {
            "id": final_entity.id,
            "type": final_entity.entity_type,
            "name": final_entity.canonical_name,
            "aliases": final_entity.aliases_list(),
            "context": final_entity.context,
            "mention_count": final_entity.mention_count,
        }
    });
    ToolResult::ok(serde_json::to_string_pretty(&result).unwrap_or_default())
}

fn exec_update_entity(input: &Value, db: &mut Database) -> ToolResult {
    let params: UpdateEntityParams = match serde_json::from_value(input.clone()) {
        Ok(p) => p,
        Err(e) => return ToolResult::err(format!("Invalid params: {}", e)),
    };

    let mut registry = EntityRegistry::new(db);

    let entity = match registry.get_entity(params.entity_id) {
        Ok(Some(e)) => e,
        Ok(None) => return ToolResult::err("Entity not found".to_string()),
        Err(e) => return ToolResult::err(format!("Error: {}", e)),
    };

    let old_name = entity.canonical_name.clone();
    let mut changes = Vec::new();

    if let Some(new_name) = &params.new_name {
        if let Err(e) = registry.rename_entity(params.entity_id, new_name) {
            return ToolResult::err(format!("Rename error: {}", e));
        }
        changes.push(format!("Renamed from '{}' to '{}' (old name added as alias)", old_name, new_name));
    }

    if let Some(new_context) = &params.new_context {
        if let Err(e) = registry.update_entity_context(params.entity_id, new_context) {
            return ToolResult::err(format!("Context update error: {}", e));
        }
        changes.push("Context updated".to_string());
    }

    let updated = match registry.get_entity(params.entity_id) {
        Ok(Some(e)) => e,
        Ok(None) => return ToolResult::err("Entity not found after update".to_string()),
        Err(e) => return ToolResult::err(format!("Error: {}", e)),
    };

    let result = json!({
        "success": true,
        "changes": changes,
        "entity": {
            "id": updated.id,
            "type": updated.entity_type,
            "name": updated.canonical_name,
            "aliases": updated.aliases_list(),
            "context": updated.context,
            "mention_count": updated.mention_count,
        }
    });
    ToolResult::ok(serde_json::to_string_pretty(&result).unwrap_or_default())
}

fn exec_add_entity_alias(input: &Value, db: &mut Database) -> ToolResult {
    let params: AddEntityAliasParams = match serde_json::from_value(input.clone()) {
        Ok(p) => p,
        Err(e) => return ToolResult::err(format!("Invalid params: {}", e)),
    };

    let mut registry = EntityRegistry::new(db);

    match registry.get_entity(params.entity_id) {
        Ok(Some(_)) => {}
        Ok(None) => return ToolResult::err("Entity not found".to_string()),
        Err(e) => return ToolResult::err(format!("Error: {}", e)),
    }

    if let Err(e) = registry.add_entity_alias(params.entity_id, &params.alias) {
        return ToolResult::err(format!("Error adding alias: {}", e));
    }

    let updated = match registry.get_entity(params.entity_id) {
        Ok(Some(e)) => e,
        Ok(None) => return ToolResult::err("Entity not found".to_string()),
        Err(e) => return ToolResult::err(format!("Error: {}", e)),
    };

    let result = json!({
        "success": true,
        "message": format!("Added alias '{}' to entity '{}'", params.alias, updated.canonical_name),
        "entity": {
            "id": updated.id,
            "type": updated.entity_type,
            "name": updated.canonical_name,
            "aliases": updated.aliases_list(),
            "context": updated.context,
            "mention_count": updated.mention_count,
        }
    });
    ToolResult::ok(serde_json::to_string_pretty(&result).unwrap_or_default())
}

fn exec_remove_entity_alias(input: &Value, db: &mut Database) -> ToolResult {
    let params: RemoveEntityAliasParams = match serde_json::from_value(input.clone()) {
        Ok(p) => p,
        Err(e) => return ToolResult::err(format!("Invalid params: {}", e)),
    };

    let mut registry = EntityRegistry::new(db);

    match registry.get_entity(params.entity_id) {
        Ok(Some(_)) => {}
        Ok(None) => return ToolResult::err("Entity not found".to_string()),
        Err(e) => return ToolResult::err(format!("Error: {}", e)),
    }

    if let Err(e) = registry.remove_entity_alias(params.entity_id, &params.alias) {
        return ToolResult::err(format!("Error removing alias: {}", e));
    }

    let updated = match registry.get_entity(params.entity_id) {
        Ok(Some(e)) => e,
        Ok(None) => return ToolResult::err("Entity not found".to_string()),
        Err(e) => return ToolResult::err(format!("Error: {}", e)),
    };

    let result = json!({
        "success": true,
        "message": format!("Removed alias '{}' from entity '{}'", params.alias, updated.canonical_name),
        "entity": {
            "id": updated.id,
            "type": updated.entity_type,
            "name": updated.canonical_name,
            "aliases": updated.aliases_list(),
            "context": updated.context,
            "mention_count": updated.mention_count,
        }
    });
    ToolResult::ok(serde_json::to_string_pretty(&result).unwrap_or_default())
}

fn exec_merge_entities(input: &Value, db: &mut Database) -> ToolResult {
    let params: MergeEntitiesParams = match serde_json::from_value(input.clone()) {
        Ok(p) => p,
        Err(e) => return ToolResult::err(format!("Invalid params: {}", e)),
    };

    let mut registry = EntityRegistry::new(db);

    let source = match registry.get_entity(params.source_id) {
        Ok(Some(e)) => e,
        Ok(None) => return ToolResult::err("Source entity not found".to_string()),
        Err(e) => return ToolResult::err(format!("Error: {}", e)),
    };
    let target = match registry.get_entity(params.target_id) {
        Ok(Some(e)) => e,
        Ok(None) => return ToolResult::err("Target entity not found".to_string()),
        Err(e) => return ToolResult::err(format!("Error: {}", e)),
    };

    let source_name = source.canonical_name.clone();
    let target_name = target.canonical_name.clone();

    if let Err(e) = registry.merge_entities(params.source_id, params.target_id) {
        return ToolResult::err(format!("Merge error: {}", e));
    }

    let updated = match registry.get_entity(params.target_id) {
        Ok(Some(e)) => e,
        Ok(None) => return ToolResult::err("Target entity not found after merge".to_string()),
        Err(e) => return ToolResult::err(format!("Error: {}", e)),
    };

    let result = json!({
        "success": true,
        "message": format!("Merged '{}' into '{}'. Source entity deleted, name added as alias.", source_name, target_name),
        "entity": {
            "id": updated.id,
            "type": updated.entity_type,
            "name": updated.canonical_name,
            "aliases": updated.aliases_list(),
            "context": updated.context,
            "mention_count": updated.mention_count,
        }
    });
    ToolResult::ok(serde_json::to_string_pretty(&result).unwrap_or_default())
}

fn exec_delete_entity(input: &Value, db: &mut Database) -> ToolResult {
    let params: DeleteEntityParams = match serde_json::from_value(input.clone()) {
        Ok(p) => p,
        Err(e) => return ToolResult::err(format!("Invalid params: {}", e)),
    };

    let mut registry = EntityRegistry::new(db);

    let entity = match registry.get_entity(params.entity_id) {
        Ok(Some(e)) => e,
        Ok(None) => return ToolResult::err("Entity not found".to_string()),
        Err(e) => return ToolResult::err(format!("Error: {}", e)),
    };

    let entity_name = entity.canonical_name.clone();

    if let Err(e) = registry.delete_entity(params.entity_id) {
        return ToolResult::err(format!("Delete error: {}", e));
    }

    let result = json!({
        "success": true,
        "message": format!("Deleted entity '{}' (ID: {})", entity_name, params.entity_id)
    });
    ToolResult::ok(serde_json::to_string_pretty(&result).unwrap_or_default())
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Extract a snippet of text around a search query match.
/// Uses char-level indexing to avoid panics on multi-byte UTF-8.
fn extract_snippet(content: &str, query: &str, max_chars: usize) -> String {
    let content_lower = content.to_lowercase();
    let first_word = query.split_whitespace().next().unwrap_or(query).to_lowercase();

    let match_char_offset = if let Some(byte_pos) = content_lower.find(&first_word) {
        content_lower[..byte_pos].chars().count()
    } else {
        return content.chars().take(max_chars).collect::<String>();
    };

    let total_chars = content.chars().count();
    let half = max_chars / 2;
    let start_char = match_char_offset.saturating_sub(half);
    let end_char = (match_char_offset + half).min(total_chars);

    let snippet: String = content.chars().skip(start_char).take(end_char - start_char).collect();

    let snippet = if start_char > 0 {
        if let Some(pos) = snippet.find(' ') {
            snippet[pos + 1..].to_string()
        } else {
            snippet
        }
    } else {
        snippet
    };
    let snippet = if end_char < total_chars {
        if let Some(pos) = snippet.rfind(' ') {
            snippet[..pos].to_string()
        } else {
            snippet
        }
    } else {
        snippet
    };

    if start_char > 0 {
        format!("...{}...", snippet)
    } else {
        format!("{}...", snippet)
    }
}
