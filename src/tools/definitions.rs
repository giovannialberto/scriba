//! Tool schema definitions for all 15 Scriba tools.
//!
//! Each tool has a param struct (used for deserialization in the executor)
//! and a schema generated via `schemars`. The `ToolSchema` type is
//! consumer-agnostic — both the agent and MCP wrap it into their formats.

use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

/// A tool schema with name, description, and JSON Schema for the input.
pub struct ToolSchema {
    pub name: &'static str,
    pub description: &'static str,
    pub input_schema: Value,
}

// ─── Read-tool param structs ─────────────────────────────────────────────────

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListRecordingsParams {
    /// Maximum number of recordings to return. Omit for all.
    pub limit: Option<i64>,
    /// Number of items to skip.
    pub offset: Option<i64>,
    /// Only include recordings from this date onward (ISO 8601, e.g. "2026-04-10").
    pub from_date: Option<String>,
    /// Only include recordings up to this date (ISO 8601, e.g. "2026-04-10").
    pub to_date: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetRecordingParams {
    /// Recording ID.
    pub id: Option<i64>,
    /// Directory name to look up.
    pub directory_name: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetTranscriptParams {
    /// Recording ID to fetch transcript for.
    pub recording_id: Option<i64>,
    /// Directory name to fetch transcript for.
    pub directory_name: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchTranscriptsParams {
    /// Search query (FTS5 syntax supported).
    pub query: String,
    /// Maximum results to return. Default: 10.
    pub limit: Option<i64>,
    /// Only include recordings from this date onward (ISO 8601, e.g. "2026-04-10").
    pub from_date: Option<String>,
    /// Only include recordings up to this date (ISO 8601, e.g. "2026-04-10").
    pub to_date: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListEntitiesParams {
    /// Filter by type: "person" or "organization". Omit for all.
    pub entity_type: Option<String>,
    /// Maximum entities to return. Omit for all.
    pub limit: Option<i64>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetEntityParams {
    /// Entity ID.
    pub id: Option<i64>,
    /// Entity name (case-insensitive).
    pub name: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetRecordingsForEntityParams {
    /// Entity ID to find recordings for.
    pub entity_id: i64,
}

// ─── Write-tool param structs ────────────────────────────────────────────────

#[derive(Debug, Deserialize, JsonSchema)]
pub struct CreateEntityParams {
    /// Entity type: "person" or "organization".
    pub entity_type: String,
    /// Canonical name for the entity.
    pub name: String,
    /// Optional aliases (e.g. nicknames, misspellings).
    pub aliases: Option<Vec<String>>,
    /// Optional context description.
    pub context: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct UpdateEntityParams {
    /// Entity ID to update.
    pub entity_id: i64,
    /// New name for the entity (old name becomes an alias).
    pub new_name: Option<String>,
    /// New context description.
    pub new_context: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct AddEntityAliasParams {
    /// Entity ID to add alias to.
    pub entity_id: i64,
    /// Alias to add.
    pub alias: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RemoveEntityAliasParams {
    /// Entity ID to remove alias from.
    pub entity_id: i64,
    /// Alias to remove.
    pub alias: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct MergeEntitiesParams {
    /// Source entity ID (will be deleted after merge).
    pub source_id: i64,
    /// Target entity ID (will receive merged data).
    pub target_id: i64,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct DeleteEntityParams {
    /// Entity ID to delete.
    pub entity_id: i64,
}

// ─── Schema generation ───────────────────────────────────────────────────────

/// Return schemas for all 14 tools.
pub fn all_tool_schemas() -> Vec<ToolSchema> {
    vec![
        ToolSchema {
            name: "list_recordings",
            description: "List all recordings with metadata (id, name, date, duration, summary snippet). Supports date filtering with from_date/to_date (ISO 8601). Use this to see what recordings are available.",
            input_schema: serde_json::to_value(schemars::schema_for!(ListRecordingsParams)).unwrap(),
        },
        ToolSchema {
            name: "get_recording",
            description: "Get full recording metadata including summary, topics, key_points, action_items, and speakers. Supports lookup by ID or directory name.",
            input_schema: serde_json::to_value(schemars::schema_for!(GetRecordingParams)).unwrap(),
        },
        ToolSchema {
            name: "get_transcript",
            description: "Fetch the full transcript text for a recording. Returns the entire content with no word limit.",
            input_schema: serde_json::to_value(schemars::schema_for!(GetTranscriptParams)).unwrap(),
        },
        ToolSchema {
            name: "search_transcripts",
            description: "Full-text search across all transcripts. Returns matching recording IDs with snippets. Supports date filtering with from_date/to_date (ISO 8601). Use this to find recordings about a topic.",
            input_schema: serde_json::to_value(schemars::schema_for!(SearchTranscriptsParams)).unwrap(),
        },
        ToolSchema {
            name: "list_entities",
            description: "List known entities (people, organizations) with context and mention counts.",
            input_schema: serde_json::to_value(schemars::schema_for!(ListEntitiesParams)).unwrap(),
        },
        ToolSchema {
            name: "get_entity",
            description: "Get entity details including all mentions and which recordings reference this entity. Supports lookup by ID or name.",
            input_schema: serde_json::to_value(schemars::schema_for!(GetEntityParams)).unwrap(),
        },
        ToolSchema {
            name: "get_recordings_for_entity",
            description: "Find all recordings that mention a specific entity. Great for cross-correlating information about a person or organization.",
            input_schema: serde_json::to_value(schemars::schema_for!(GetRecordingsForEntityParams)).unwrap(),
        },
        ToolSchema {
            name: "get_world_context",
            description: "Read the full world.md knowledge base containing info about the owner, their relationships, organizations, projects, and beliefs.",
            input_schema: serde_json::json!({"type": "object", "properties": {}, "required": []}),
        },
        ToolSchema {
            name: "get_stats",
            description: "Get recording statistics: total recordings, total hours, total words, entity counts.",
            input_schema: serde_json::json!({"type": "object", "properties": {}, "required": []}),
        },
        // ─── Write tools ─────────────────────────────────────────────
        ToolSchema {
            name: "create_entity",
            description: "Create a new entity (person or organization) in the knowledge base. Returns an error if an entity with the same name already exists.",
            input_schema: serde_json::to_value(schemars::schema_for!(CreateEntityParams)).unwrap(),
        },
        ToolSchema {
            name: "update_entity",
            description: "Update an entity's name or context. When renaming, the old name is automatically added as an alias for smart matching in future transcripts.",
            input_schema: serde_json::to_value(schemars::schema_for!(UpdateEntityParams)).unwrap(),
        },
        ToolSchema {
            name: "add_entity_alias",
            description: "Add an alias to an entity. Aliases are used for smart matching - if a transcript mentions the alias, it will be linked to this entity.",
            input_schema: serde_json::to_value(schemars::schema_for!(AddEntityAliasParams)).unwrap(),
        },
        ToolSchema {
            name: "remove_entity_alias",
            description: "Remove an alias from an entity.",
            input_schema: serde_json::to_value(schemars::schema_for!(RemoveEntityAliasParams)).unwrap(),
        },
        ToolSchema {
            name: "merge_entities",
            description: "Merge two entities into one. The source entity's name becomes an alias of the target, all mentions are transferred, and contexts are combined. Source entity is deleted.",
            input_schema: serde_json::to_value(schemars::schema_for!(MergeEntitiesParams)).unwrap(),
        },
        ToolSchema {
            name: "delete_entity",
            description: "Delete an entity from the knowledge base. Mentions will be unlinked but not deleted.",
            input_schema: serde_json::to_value(schemars::schema_for!(DeleteEntityParams)).unwrap(),
        },
    ]
}
