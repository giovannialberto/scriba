use anyhow::Result;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::io::{self, AsyncBufReadExt, AsyncWriteExt, BufReader};

use crate::database::{Database, Recording, Transcript};

#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    #[serde(default)]
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Option<Value>,
}

#[derive(Debug, Serialize)]
struct JsonRpcResponse<'a> {
    jsonrpc: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

#[derive(Debug, Serialize)]
struct JsonRpcError {
    code: i64,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<Value>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ListTranscriptsParams {
    /// Maximum number of items to return
    limit: Option<i64>,
    /// Number of items to skip
    offset: Option<i64>,
    /// Include recordings without transcripts
    include_without_transcripts: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct GetTranscriptParams {
    /// Recording ID to fetch transcript for
    recording_id: Option<i64>,
    /// Directory name to fetch transcript for
    directory_name: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct SearchTranscriptsParams {
    /// Search query for full-text search
    query: String,
    /// Maximum number of results to return
    limit: Option<i64>,
}

#[derive(Debug, Serialize, JsonSchema)]
struct RecordingInfo {
    id: Option<i64>,
    directory_name: String,
    display_name: Option<String>,
    created_at: String,
    updated_at: String,
    has_transcript: bool,
    transcript_status: Option<String>,
    language_code: Option<String>,
    model_used: Option<String>,
}

#[derive(Debug, Serialize, JsonSchema)]
struct TranscriptInfo {
    id: Option<i64>,
    recording_id: i64,
    created_at: String,
    updated_at: String,
    word_count: Option<i64>,
    character_count: Option<i64>,
    language_detected: Option<String>,
}

#[derive(Debug, Serialize, JsonSchema)]
struct SearchResult {
    recording: RecordingInfo,
    transcript: TranscriptInfo,
}

fn make_error(code: i64, message: impl Into<String>, data: Option<Value>) -> JsonRpcError {
    JsonRpcError {
        code,
        message: message.into(),
        data,
    }
}

// MCP STDIO transport: newline-delimited JSON messages
async fn read_json_message(reader: &mut BufReader<tokio::io::Stdin>) -> Result<Option<String>> {
    let mut line = String::new();
    let bytes_read = reader.read_line(&mut line).await?;
    
    if bytes_read == 0 {
        return Ok(None); // EOF
    }
    
    // Remove trailing newline
    let trimmed = line.trim_end().to_string();
    if trimmed.is_empty() {
        // Skip empty lines
        return Ok(Some(String::new()));
    }
    
    Ok(Some(trimmed))
}

async fn write_json_message(writer: &mut tokio::io::Stdout, payload: &Value) -> Result<()> {
    let json_str = serde_json::to_string(payload)?;
    writer.write_all(json_str.as_bytes()).await?;
    writer.write_all(b"\n").await?;
    writer.flush().await?;
    Ok(())
}

fn response_ok(id: Option<Value>, result: Value) -> Value {
    serde_json::to_value(JsonRpcResponse {
        jsonrpc: "2.0",
        id,
        result: Some(result),
        error: None,
    })
    .unwrap()
}

fn response_err(id: Option<Value>, err: JsonRpcError) -> Value {
    serde_json::to_value(JsonRpcResponse {
        jsonrpc: "2.0",
        id,
        result: None,
        error: Some(err),
    })
    .unwrap()
}

fn recording_to_info(recording: &Recording) -> RecordingInfo {
    RecordingInfo {
        id: recording.id,
        directory_name: recording.directory_name.clone(),
        display_name: recording.display_name.clone(),
        created_at: recording.created_at.to_rfc3339(),
        updated_at: recording.updated_at.to_rfc3339(),
        has_transcript: recording.has_transcript,
        transcript_status: Some(recording.transcript_status.clone()),
        language_code: Some(recording.language_code.clone()),
        model_used: Some(recording.model_used.clone()),
    }
}

fn transcript_to_info(transcript: &Transcript) -> TranscriptInfo {
    TranscriptInfo {
        id: transcript.id,
        recording_id: transcript.recording_id,
        created_at: transcript.created_at.to_rfc3339(),
        updated_at: transcript.updated_at.to_rfc3339(),
        word_count: transcript.word_count,
        character_count: transcript.character_count,
        language_detected: transcript.language_detected.clone(),
    }
}

fn tool_schema_list_transcripts() -> Value {
    let schema = schemars::schema_for!(ListTranscriptsParams);
    json!({
        "name": "list_transcripts",
        "description": "List recordings with transcripts, newest first. Returns recording metadata including creation dates, transcript status, and audio format information.",
        "inputSchema": schema
    })
}

fn tool_schema_get_transcript() -> Value {
    let schema = schemars::schema_for!(GetTranscriptParams);
    json!({
        "name": "get_transcript",
        "description": "Fetch the full transcript content for a specific recording by ID or directory name. Returns the complete transcribed text.",
        "inputSchema": schema
    })
}

fn tool_schema_search_transcripts() -> Value {
    let schema = schemars::schema_for!(SearchTranscriptsParams);
    json!({
        "name": "search_transcripts",
        "description": "Full-text search across all transcripts using SQLite FTS. Supports complex queries and phrase matching.",
        "inputSchema": schema
    })
}

fn tool_schema_get_recording_info() -> Value {
    let schema = schemars::schema_for!(GetTranscriptParams);
    json!({
        "name": "get_recording_info",
        "description": "Get detailed metadata about a specific recording including audio format, duration, file size, and transcript status.",
        "inputSchema": schema
    })
}

async fn handle_initialize(id: Option<Value>, params: Option<Value>) -> Value {
    let mut protocol_version = "2024-11-05".to_string();
    
    if let Some(Value::Object(p)) = params {
        if let Some(Value::String(v)) = p.get("protocolVersion") {
            protocol_version = v.clone();
        }
    }
    
    // Accept any protocol version for compatibility
    let result = json!({
        "protocolVersion": protocol_version,
        "serverInfo": {
            "name": "scriba-mcp",
            "version": env!("CARGO_PKG_VERSION")
        },
        "capabilities": {
            "tools": {
                "listChanged": false
            },
            "resources": {
                "subscribe": false,
                "listChanged": false
            },
            "prompts": {
                "listChanged": false
            },
            "logging": {}
        }
    });
    
    response_ok(id, result)
}

async fn handle_tools_list(id: Option<Value>) -> Value {
    let tools = vec![
        tool_schema_list_transcripts(),
        tool_schema_get_transcript(),
        tool_schema_search_transcripts(),
        tool_schema_get_recording_info(),
    ];
    response_ok(id, json!({"tools": tools}))
}

async fn handle_tools_call(id: Option<Value>, params: Option<Value>) -> Value {
    let Some(Value::Object(map)) = params else {
        return response_err(id, make_error(-32602, "Missing params", None));
    };
    let name = match map.get("name").and_then(|v| v.as_str()) {
        Some(s) => s,
        None => return response_err(id, make_error(-32602, "Missing tool name", None)),
    };
    let args = map.get("arguments").cloned().unwrap_or_else(|| json!({}));

    // Create fresh database connection to ensure latest data from all Scriba instances
    let db = match Database::new() {
        Ok(database) => database,
        Err(e) => return response_err(id, make_error(-32000, format!("Database error: {}", e), None)),
    };

    match name {
        "list_transcripts" => {
            let params: ListTranscriptsParams = match serde_json::from_value(args) {
                Ok(p) => p,
                Err(e) => return response_err(id, make_error(-32602, format!("Invalid params: {}", e), None)),
            };

            let recordings = match db.list_recordings(params.limit, params.offset) {
                Ok(list) => list,
                Err(e) => return response_err(id, make_error(-32000, format!("DB error: {}", e), None)),
            };

            let include_without = params.include_without_transcripts.unwrap_or(false);
            let filtered: Vec<_> = recordings
                .into_iter()
                .filter(|r| include_without || r.has_transcript)
                .map(|r| recording_to_info(&r))
                .collect();

            response_ok(id, json!({
                "content": [{
                    "type": "text",
                    "text": serde_json::to_string_pretty(&filtered).unwrap()
                }],
                "isError": false
            }))
        }
        "get_transcript" => {
            let params: GetTranscriptParams = match serde_json::from_value(args) {
                Ok(p) => p,
                Err(e) => return response_err(id, make_error(-32602, format!("Invalid params: {}", e), None)),
            };

            let recording_id = if let Some(idv) = params.recording_id {
                idv
            } else if let Some(dn) = params.directory_name {
                match db.get_recording_by_directory(&dn) {
                    Ok(Some(r)) => r.id.unwrap_or_default(),
                    Ok(None) => return response_err(id, make_error(404, "Recording not found", None)),
                    Err(e) => return response_err(id, make_error(-32000, format!("DB error: {}", e), None)),
                }
            } else {
                return response_err(id, make_error(-32602, "Provide recording_id or directory_name", None));
            };

            let transcript = match db.get_transcript_by_recording_id(recording_id) {
                Ok(Some(t)) => t,
                Ok(None) => return response_err(id, make_error(404, "Transcript not found", None)),
                Err(e) => return response_err(id, make_error(-32000, format!("DB error: {}", e), None)),
            };

            response_ok(id, json!({
                "content": [{
                    "type": "text",
                    "text": transcript.content
                }],
                "isError": false
            }))
        }
        "search_transcripts" => {
            let params: SearchTranscriptsParams = match serde_json::from_value(args) {
                Ok(p) => p,
                Err(e) => return response_err(id, make_error(-32602, format!("Invalid params: {}", e), None)),
            };

            let results = match db.search_transcripts(&params.query, params.limit) {
                Ok(v) => v,
                Err(e) => return response_err(id, make_error(-32000, format!("DB error: {}", e), None)),
            };

            let search_results: Vec<_> = results
                .iter()
                .map(|(r, t)| SearchResult {
                    recording: recording_to_info(r),
                    transcript: transcript_to_info(t),
                })
                .collect();

            response_ok(id, json!({
                "content": [{
                    "type": "text",
                    "text": serde_json::to_string_pretty(&search_results).unwrap()
                }],
                "isError": false
            }))
        }
        "get_recording_info" => {
            let params: GetTranscriptParams = match serde_json::from_value(args) {
                Ok(p) => p,
                Err(e) => return response_err(id, make_error(-32602, format!("Invalid params: {}", e), None)),
            };

            let recording = if let Some(id_val) = params.recording_id {
                match db.get_recording(id_val) {
                    Ok(Some(r)) => r,
                    Ok(None) => return response_err(id, make_error(404, "Recording not found", None)),
                    Err(e) => return response_err(id, make_error(-32000, format!("DB error: {}", e), None)),
                }
            } else if let Some(dir_name) = params.directory_name {
                match db.get_recording_by_directory(&dir_name) {
                    Ok(Some(r)) => r,
                    Ok(None) => return response_err(id, make_error(404, "Recording not found", None)),
                    Err(e) => return response_err(id, make_error(-32000, format!("DB error: {}", e), None)),
                }
            } else {
                return response_err(id, make_error(-32602, "Provide recording_id or directory_name", None));
            };

            let recording_info = recording_to_info(&recording);
            response_ok(id, json!({
                "content": [{
                    "type": "text",
                    "text": serde_json::to_string_pretty(&recording_info).unwrap()
                }],
                "isError": false
            }))
        }
        other => response_err(
            id,
            make_error(-32601, format!("Unknown tool: {}", other), None),
        ),
    }
}

async fn handle_resources_list(id: Option<Value>) -> Value {
    // Resources are not implemented in this version - focus on tools
    response_ok(id, json!({"resources": []}))
}

/// Run the MCP server over stdio
pub async fn run_mcp_server() -> Result<()> {
    // Hint to DB layer to skip heavy integrity checks in MCP mode
    std::env::set_var("SCRIBA_MCP_MODE", "1");

    let mut reader = BufReader::new(io::stdin());
    let mut writer = io::stdout();

    while let Some(line) = read_json_message(&mut reader).await? {
        if line.is_empty() {
            continue; // Skip empty lines
        }
        
        let req: JsonRpcRequest = match serde_json::from_str::<JsonRpcRequest>(&line) {
            Ok(v) => {
                // Validate JSON-RPC version
                if v.jsonrpc != "2.0" {
                    eprintln!("[scriba-mcp] Invalid JSON-RPC version: {}", v.jsonrpc);
                }
                v
            },
            Err(e) => {
                eprintln!("[scriba-mcp] JSON parse error: {}", e);
                let resp = response_err(None, make_error(-32700, format!("Parse error: {}", e), None));
                if let Err(write_err) = write_json_message(&mut writer, &resp).await {
                    eprintln!("[scriba-mcp] Error writing response: {}", write_err);
                }
                continue;
            }
        };

        let id = req.id.clone();
        let method = req.method.as_str();

        let resp = match method {
            "initialize" => handle_initialize(id, req.params).await,
            "ping" => response_ok(id, json!({})),
            "tools/list" => handle_tools_list(id).await,
            "tools/call" => handle_tools_call(id.clone(), req.params).await,
            "resources/list" => handle_resources_list(id.clone()).await,
            "resources/templates/list" => response_ok(id, json!({"resourceTemplates": []})),
            "prompts/list" => response_ok(id, json!({"prompts": []})),
            "prompts/get" => response_err(id, make_error(404, "Prompt not found", None)),
            "logging/setLevel" => response_ok(id, json!({})),
            "notifications/initialized" => continue, // No response for notifications
            other => response_err(
                id,
                make_error(-32601, format!("Method not found: {}", other), None),
            ),
        };

        if req.id.is_some() {
            if let Err(write_err) = write_json_message(&mut writer, &resp).await {
                eprintln!("[scriba-mcp] Error writing response: {}", write_err);
                return Err(write_err);
            }
        }
    }
    Ok(())
}