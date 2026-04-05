use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::io::{self, AsyncBufReadExt, AsyncWriteExt, BufReader};

use crate::database::Database;
use crate::tools;

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

    let trimmed = line.trim_end().to_string();
    if trimmed.is_empty() {
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

async fn handle_initialize(id: Option<Value>, params: Option<Value>) -> Value {
    let mut protocol_version = "2024-11-05".to_string();

    if let Some(Value::Object(p)) = params {
        if let Some(Value::String(v)) = p.get("protocolVersion") {
            protocol_version = v.clone();
        }
    }

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
    let tool_list: Vec<Value> = tools::all_tool_schemas()
        .into_iter()
        .map(|s| {
            json!({
                "name": s.name,
                "description": s.description,
                "inputSchema": s.input_schema,
            })
        })
        .collect();
    response_ok(id, json!({ "tools": tool_list }))
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

    // Legacy name aliases for backward compatibility
    let name = match name {
        "list_transcripts" => "list_recordings",
        "get_recording_info" => "get_recording",
        "search_by_entity" => "get_recordings_for_entity",
        other => other,
    };

    let mut db = match Database::new() {
        Ok(database) => database,
        Err(e) => {
            return response_err(
                id,
                make_error(-32000, format!("Database error: {}", e), None),
            )
        }
    };

    let result = tools::execute_tool(name, &args, &mut db);

    if result.is_error {
        response_err(id, make_error(-32000, result.output, None))
    } else {
        response_ok(
            id,
            json!({
                "content": [{
                    "type": "text",
                    "text": result.output,
                }],
                "isError": false,
            }),
        )
    }
}

async fn handle_resources_list(id: Option<Value>) -> Value {
    response_ok(id, json!({"resources": []}))
}

/// Run the MCP server over stdio
pub async fn run_mcp_server() -> Result<()> {
    // Hint to DB layer to skip heavy integrity checks in MCP mode
    // SAFETY: called once at MCP server startup before any threads are spawned.
    unsafe { std::env::set_var("SCRIBA_MCP_MODE", "1") };

    let mut reader = BufReader::new(io::stdin());
    let mut writer = io::stdout();

    while let Some(line) = read_json_message(&mut reader).await? {
        if line.is_empty() {
            continue;
        }

        let req: JsonRpcRequest = match serde_json::from_str::<JsonRpcRequest>(&line) {
            Ok(v) => {
                if v.jsonrpc != "2.0" {
                    eprintln!("[scriba-mcp] Invalid JSON-RPC version: {}", v.jsonrpc);
                }
                v
            }
            Err(e) => {
                eprintln!("[scriba-mcp] JSON parse error: {}", e);
                let resp =
                    response_err(None, make_error(-32700, format!("Parse error: {}", e), None));
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
            "notifications/initialized" => continue,
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
