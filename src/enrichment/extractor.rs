//! Extraction service for knowledge extraction from transcripts.

use std::collections::HashMap;

use anyhow::{Context, Result};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::provider::LlmProvider;
use super::prompts;
use super::search::EntitySearchResults;
use super::world::WorldData;
use crate::core::config::EnrichmentConfig;

/// Result of extracting metadata from a transcript.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionResult {
    /// AI-generated title for the recording.
    pub title: String,
    /// Brief summary of the content.
    pub summary: String,
    /// Main topics discussed.
    pub topics: Vec<String>,
    /// People mentioned in the transcript.
    pub people: Vec<ExtractedEntity>,
    /// Organizations mentioned in the transcript.
    pub organizations: Vec<ExtractedEntity>,
    /// Key points or insights.
    pub key_points: Vec<String>,
    /// Action items or tasks mentioned.
    pub action_items: Vec<String>,
}

/// An entity (person or organization) extracted from a transcript.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEntity {
    /// Name as mentioned in the transcript.
    pub name: String,
    /// Context about this entity from the transcript.
    pub context: String,
    /// If this entity matches a known entity from the world, the canonical name.
    /// None means this is a genuinely new entity.
    #[serde(default)]
    pub resolved_to: Option<String>,
}

/// Result of context update.
#[derive(Debug, Clone, Deserialize)]
pub struct ContextUpdateResult {
    pub updated_context: String,
    pub new_facts: Vec<String>,
}

/// Entity extracted from world description.
#[derive(Debug, Clone, Deserialize)]
pub struct WorldEntityPerson {
    pub name: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default)]
    pub context: String,
    #[serde(default)]
    pub is_owner: bool,
}

/// Organization extracted from world description.
#[derive(Debug, Clone, Deserialize)]
pub struct WorldEntityOrganization {
    pub name: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default)]
    pub context: String,
}

/// Result of extracting entities from world description.
#[derive(Debug, Clone, Deserialize)]
pub struct WorldEntityExtractionResult {
    #[serde(default)]
    pub people: Vec<WorldEntityPerson>,
    #[serde(default)]
    pub organizations: Vec<WorldEntityOrganization>,
}

/// Rough tokens-per-word ratio for English/Italian transcripts.
const TOKENS_PER_WORD: f32 = 1.4;
/// Prompt scaffolding plus JSON output, in tokens, on top of transcript and world.
const EXTRACTION_OVERHEAD_TOKENS: u32 = 6_000;
/// Chunk size used when the model's context window is unknown.
const DEFAULT_CHUNK_WORDS: usize = 6_000;
/// Never chunk finer than this; the extraction needs surrounding context.
const MIN_CHUNK_WORDS: usize = 1_500;
/// Never send more than this in one extraction call, even to 1M-token models:
/// extraction quality drops before the window runs out.
const MAX_CHUNK_WORDS: usize = 20_000;
/// Words repeated between consecutive chunks so nothing is cut mid-thought.
const CHUNK_OVERLAP_WORDS: usize = 150;
/// Share of the context window we allow one extraction request to use.
const WINDOW_BUDGET: f32 = 0.6;

/// JSON schema for [`ExtractionResult`], handed to providers that enforce
/// structured output. Every property is required so strict modes accept it;
/// `resolved_to` is nullable.
pub fn extraction_schema() -> Value {
    let entity = json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "context": {"type": "string"},
            "resolved_to": {"type": ["string", "null"]}
        },
        "required": ["name", "context", "resolved_to"]
    });
    json!({
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "summary": {"type": "string"},
            "topics": {"type": "array", "items": {"type": "string"}},
            "people": {"type": "array", "items": entity},
            "organizations": {"type": "array", "items": entity},
            "key_points": {"type": "array", "items": {"type": "string"}},
            "action_items": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["title", "summary", "topics", "people", "organizations", "key_points", "action_items"]
    })
}

/// Split text into word-bounded chunks of at most `chunk_words`, each
/// repeating the last `overlap` words of the previous chunk.
pub fn split_into_chunks(text: &str, chunk_words: usize, overlap: usize) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let chunk_words = chunk_words.max(1);
    let overlap = overlap.min(chunk_words / 2);
    if words.len() <= chunk_words {
        return vec![words.join(" ")];
    }
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < words.len() {
        let end = (start + chunk_words).min(words.len());
        chunks.push(words[start..end].join(" "));
        if end == words.len() {
            break;
        }
        start = end - overlap;
    }
    chunks
}

/// How many transcript words fit in one extraction call for a model with the
/// given context window, leaving room for the world context and the prompt.
pub fn chunk_words_for_window(window: Option<u32>, world_chars: usize) -> usize {
    let Some(window) = window else {
        return DEFAULT_CHUNK_WORDS;
    };
    let world_tokens = (world_chars / 4) as f32;
    let budget = window as f32 * WINDOW_BUDGET - EXTRACTION_OVERHEAD_TOKENS as f32 - world_tokens;
    let words = (budget / TOKENS_PER_WORD).max(0.0) as usize;
    words.clamp(MIN_CHUNK_WORDS, MAX_CHUNK_WORDS)
}

/// Merge entity lists from several chunks: one entry per canonical identity
/// (resolved name, else transcript name, case-insensitive), contexts combined.
pub fn merge_entities(lists: &[&[ExtractedEntity]]) -> Vec<ExtractedEntity> {
    const MAX_CONTEXT_CHARS: usize = 400;
    let mut merged: Vec<ExtractedEntity> = Vec::new();
    let mut index: HashMap<String, usize> = HashMap::new();
    for list in lists {
        for entity in list.iter() {
            let key = entity
                .resolved_to
                .as_deref()
                .unwrap_or(&entity.name)
                .trim()
                .to_lowercase();
            if key.is_empty() {
                continue;
            }
            match index.get(&key) {
                Some(&i) => {
                    let existing = &mut merged[i];
                    if existing.resolved_to.is_none() {
                        existing.resolved_to = entity.resolved_to.clone();
                    }
                    let addition = entity.context.trim();
                    if !addition.is_empty()
                        && !existing.context.contains(addition)
                        && existing.context.len() + addition.len() + 2 <= MAX_CONTEXT_CHARS
                    {
                        if !existing.context.is_empty() {
                            existing.context.push_str("; ");
                        }
                        existing.context.push_str(addition);
                    }
                }
                None => {
                    index.insert(key, merged.len());
                    merged.push(entity.clone());
                }
            }
        }
    }
    merged
}

/// Service for extracting knowledge from transcripts using LLM.
pub struct EnrichmentService {
    provider: Box<dyn LlmProvider>,
}

impl EnrichmentService {
    /// Create from an LLM provider.
    pub fn new(provider: Box<dyn LlmProvider>) -> Self {
        Self { provider }
    }

    /// Create from enrichment configuration.
    pub fn from_config(config: &EnrichmentConfig) -> Self {
        Self {
            provider: super::create_provider(config),
        }
    }

    /// Check if the enrichment service is available.
    pub async fn health_check(&self) -> Result<()> {
        self.provider
            .health_check()
            .await
            .map_err(|e| anyhow::anyhow!("{}", e))
            .context("Enrichment service health check failed")
    }

    /// Get the provider display name.
    pub fn provider_display_name(&self) -> &str {
        self.provider.display_name()
    }

    /// Get the model name in use.
    pub fn model(&self) -> &str {
        self.provider.model()
    }

    /// Extract metadata from a transcript (no world context).
    pub async fn extract(&self, transcript: &str) -> Result<ExtractionResult> {
        self.extract_with_full_context(transcript, None).await
    }

    /// Ask for schema-constrained JSON and parse it, retrying once on a parse
    /// failure (models occasionally wrap or truncate output).
    async fn generate_and_parse<T: DeserializeOwned>(
        &self,
        prompt: &str,
        schema: &Value,
        what: &str,
    ) -> Result<T> {
        let mut last_err = None;
        for _ in 0..2 {
            let response = self
                .provider
                .generate_structured(prompt, schema)
                .await
                .map_err(|e| anyhow::anyhow!("{}", e))
                .with_context(|| format!("Failed to generate {what}"))?;
            match serde_json::from_str::<T>(strip_markdown_fences(&response)) {
                Ok(parsed) => return Ok(parsed),
                Err(e) => last_err = Some(e),
            }
        }
        Err(anyhow::anyhow!(last_err.expect("at least one attempt")))
            .with_context(|| format!("Failed to parse {what}"))
    }

    /// Update entity context with new mentions.
    pub async fn update_entity_context(
        &self,
        entity_name: &str,
        entity_type: &str,
        existing_context: &str,
        new_mentions: &[(&str, &str)],
    ) -> Result<ContextUpdateResult> {
        let prompt = prompts::build_context_update_prompt(
            entity_name,
            entity_type,
            existing_context,
            new_mentions,
        );

        let response = self
            .provider
            .generate(&prompt)
            .await
            .map_err(|e| anyhow::anyhow!("{}", e))
            .context("Failed to update entity context")?;

        let cleaned = strip_markdown_fences(&response);

        let result: ContextUpdateResult =
            serde_json::from_str(cleaned).context("Failed to parse context update result")?;

        Ok(result)
    }

    /// Extract metadata with world context.
    ///
    /// The world context contains everything the LLM needs: owner profile,
    /// known people, organizations, etc. The LLM resolves entity mentions
    /// inline against the world.
    pub async fn extract_with_full_context(
        &self,
        transcript: &str,
        world_context: Option<&str>,
    ) -> Result<ExtractionResult> {
        let schema = extraction_schema();
        let world_chars = world_context.map(str::len).unwrap_or(0);
        let chunk_words = chunk_words_for_window(self.provider.context_window().await, world_chars);
        let total_words = transcript.split_whitespace().count();

        if total_words <= chunk_words {
            let prompt = prompts::build_full_context_extraction_prompt(transcript, world_context);
            return self.generate_and_parse(&prompt, &schema, "extraction result").await;
        }

        // Map: extract each chunk with the same world context.
        let chunks = split_into_chunks(transcript, chunk_words, CHUNK_OVERLAP_WORDS);
        let mut partials: Vec<ExtractionResult> = Vec::with_capacity(chunks.len());
        for (i, chunk) in chunks.iter().enumerate() {
            let prompt = prompts::build_full_context_extraction_prompt(chunk, world_context);
            let partial: ExtractionResult = self
                .generate_and_parse(&prompt, &schema, "extraction result")
                .await
                .with_context(|| format!("Chunk {} of {}", i + 1, chunks.len()))?;
            partials.push(partial);
        }

        // Reduce: entities are merged deterministically, recording-level fields
        // by the model (falling back to a plain union if that call fails).
        let people_lists: Vec<&[ExtractedEntity]> = partials.iter().map(|p| p.people.as_slice()).collect();
        let org_lists: Vec<&[ExtractedEntity]> = partials.iter().map(|p| p.organizations.as_slice()).collect();
        let people = merge_entities(&people_lists);
        let organizations = merge_entities(&org_lists);

        let partial_json: Vec<String> = partials
            .iter()
            .map(|p| {
                serde_json::to_string(&json!({
                    "title": p.title,
                    "summary": p.summary,
                    "topics": p.topics,
                    "key_points": p.key_points,
                    "action_items": p.action_items,
                }))
                .unwrap_or_default()
            })
            .collect();
        let merge_prompt = prompts::build_extraction_merge_prompt(&partial_json);
        let mut merged: ExtractionResult = match self
            .generate_and_parse(&merge_prompt, &schema, "merged extraction")
            .await
        {
            Ok(m) => m,
            Err(_) => union_extractions(&partials),
        };
        merged.people = people;
        merged.organizations = organizations;
        Ok(merged)
    }

    /// Evolve the world by extracting a conservative JSON delta from a new recording.
    ///
    /// Returns the parsed delta as `WorldData`. The caller is responsible for
    /// merging it into the existing world via `WorldData::merge()`.
    /// Returns `None` if the LLM response couldn't be parsed (non-fatal).
    pub async fn evolve_world(
        &self,
        current_world: &str,
        transcript: &str,
        extraction: &ExtractionResult,
    ) -> Result<Option<WorldData>> {
        let extraction_summary = format!(
            "Title: {}\nSummary: {}\nTopics: {}\nPeople: {}\nOrganizations: {}",
            extraction.title,
            extraction.summary,
            extraction.topics.join(", "),
            extraction
                .people
                .iter()
                .map(|p| format!("{} ({})", p.name, p.context))
                .collect::<Vec<_>>()
                .join(", "),
            extraction
                .organizations
                .iter()
                .map(|o| format!("{} ({})", o.name, o.context))
                .collect::<Vec<_>>()
                .join(", ")
        );

        let prompt =
            prompts::build_world_evolution_prompt(current_world, transcript, &extraction_summary);

        let response = self
            .provider
            .generate(&prompt)
            .await
            .map_err(|e| anyhow::anyhow!("{}", e))
            .context("Failed to evolve world description")?;

        let cleaned = strip_markdown_fences(&response);

        match WorldData::from_json(cleaned) {
            Ok(delta) => Ok(Some(delta)),
            Err(e) => {
                eprintln!("  Warning: could not parse world evolution response as JSON: {}", e);
                Ok(None)
            }
        }
    }

    /// Extract entities from a world description.
    pub async fn extract_world_entities(
        &self,
        world_content: &str,
    ) -> Result<WorldEntityExtractionResult> {
        let prompt = prompts::build_world_entity_extraction_prompt(world_content);

        let response = self
            .provider
            .generate(&prompt)
            .await
            .map_err(|e| anyhow::anyhow!("{}", e))
            .context("Failed to extract entities from world description")?;

        let cleaned = strip_markdown_fences(&response);

        let result: WorldEntityExtractionResult = serde_json::from_str(cleaned)
            .context("Failed to parse world entity extraction result")?;

        Ok(result)
    }

    /// Extract a structured world profile from a free-form seed description.
    pub async fn extract_world_seed(&self, seed_content: &str) -> Result<WorldData> {
        let prompt = prompts::build_world_seed_extraction_prompt(seed_content);

        let response = self
            .provider
            .generate(&prompt)
            .await
            .map_err(|e| anyhow::anyhow!("{}", e))
            .context("Failed to extract world seed")?;

        let cleaned = strip_markdown_fences(&response);

        WorldData::from_json(cleaned).context("Failed to parse world seed extraction result")
    }

    /// Compact an entity's context by merging existing + new info into a clean description.
    pub async fn compact_entity_context(
        &self,
        entity_name: &str,
        entity_type: &str,
        existing_context: &str,
        new_info: &str,
    ) -> Result<String> {
        let prompt = prompts::build_context_compaction_prompt(
            entity_name,
            entity_type,
            existing_context,
            new_info,
        );

        let response = self
            .provider
            .generate(&prompt)
            .await
            .map_err(|e| anyhow::anyhow!("{}", e))
            .context("Failed to compact entity context")?;

        let compacted = response
            .trim()
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim_matches('"')
            .trim()
            .to_string();

        if compacted.is_empty() || compacted.starts_with('{') || compacted.starts_with('[') {
            Ok(format!("{}. {}", existing_context.trim_end_matches('.'), new_info))
        } else {
            Ok(compacted)
        }
    }

    /// Identify speakers in a diarized transcript using world context.
    pub async fn identify_speakers(
        &self,
        diarized_text: &str,
        world_context: Option<&str>,
        num_speakers: usize,
    ) -> Result<std::collections::HashMap<String, Option<String>>> {
        let prompt = prompts::build_speaker_identification_prompt(
            diarized_text,
            world_context,
            num_speakers,
        );

        let response = self
            .provider
            .generate(&prompt)
            .await
            .map_err(|e| anyhow::anyhow!("{}", e))
            .context("Failed to identify speakers")?;

        let cleaned = strip_markdown_fences(&response);

        let parsed: serde_json::Value =
            serde_json::from_str(cleaned).context("Failed to parse speaker identification JSON")?;

        let mut result = std::collections::HashMap::new();

        if let Some(speakers) = parsed.get("speakers").and_then(|s| s.as_object()) {
            for (key, value) in speakers {
                let resolved = if value.is_null() {
                    None
                } else {
                    value.as_str().map(|s| s.to_string())
                };
                result.insert(key.clone(), resolved);
            }
        }

        Ok(result)
    }

    /// Resolve unresolved entities using web search results.
    ///
    /// Takes search results for unresolved entities plus a compact world summary,
    /// builds a resolution prompt, and asks the LLM to cross-reference.
    /// Returns a map from transcript name -> canonical name (or None for genuinely new).
    /// On any failure, returns an empty map (non-fatal).
    pub async fn resolve_with_search(
        &self,
        search_results: &[EntitySearchResults],
        world_summary: &str,
    ) -> HashMap<String, Option<String>> {
        // Build the input tuples for the prompt: (name, type, context, formatted_search_results)
        let unresolved: Vec<(String, String, String, String)> = search_results
            .iter()
            .map(|sr| {
                let formatted = sr
                    .results
                    .iter()
                    .map(|r| {
                        format!("    - \"{}\" ({}) — {}", r.title, r.url, r.snippet)
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                (
                    sr.entity_name.clone(),
                    sr.entity_type.clone(),
                    sr.entity_context.clone(),
                    formatted,
                )
            })
            .collect();

        let prompt = prompts::build_search_resolution_prompt(&unresolved, world_summary);

        let response = match self
            .provider
            .generate(&prompt)
            .await
        {
            Ok(r) => r,
            Err(_) => return HashMap::new(),
        };

        let cleaned = strip_markdown_fences(&response);

        // Parse: { "resolutions": { "Name": "Canonical" | null } }
        let parsed: serde_json::Value = match serde_json::from_str(cleaned) {
            Ok(v) => v,
            Err(_) => return HashMap::new(),
        };

        let mut result = HashMap::new();

        if let Some(resolutions) = parsed.get("resolutions").and_then(|r| r.as_object()) {
            for (key, value) in resolutions {
                let resolved = if value.is_null() {
                    None
                } else {
                    value.as_str().map(|s| s.to_string())
                };
                result.insert(key.clone(), resolved);
            }
        }

        result
    }
}

/// Deterministic fallback for the reduce step: first title, joined summaries,
/// deduplicated lists.
fn union_extractions(partials: &[ExtractionResult]) -> ExtractionResult {
    fn dedup(items: impl Iterator<Item = String>) -> Vec<String> {
        let mut seen = std::collections::HashSet::new();
        items
            .filter(|s| !s.trim().is_empty())
            .filter(|s| seen.insert(s.trim().to_lowercase()))
            .collect()
    }
    ExtractionResult {
        title: partials.first().map(|p| p.title.clone()).unwrap_or_default(),
        summary: partials
            .iter()
            .map(|p| p.summary.trim())
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(" "),
        topics: dedup(partials.iter().flat_map(|p| p.topics.iter().cloned())),
        people: Vec::new(),
        organizations: Vec::new(),
        key_points: dedup(partials.iter().flat_map(|p| p.key_points.iter().cloned())),
        action_items: dedup(partials.iter().flat_map(|p| p.action_items.iter().cloned())),
    }
}

/// Strip markdown fences from LLM responses.
fn strip_markdown_fences(response: &str) -> &str {
    response
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim()
}

impl ExtractionResult {
    /// Get all extracted entities (people + organizations) with their type.
    pub fn all_entities(&self) -> Vec<(&ExtractedEntity, &str)> {
        let mut entities: Vec<(&ExtractedEntity, &str)> = self.people
            .iter()
            .map(|p| (p, "person"))
            .collect();
        entities.extend(self.organizations.iter().map(|o| (o, "organization")));
        entities
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::provider::ProviderError;
    use async_trait::async_trait;
    use std::sync::Mutex;

    /// Fake provider: returns a partial per transcript chunk and a merged
    /// result for the reduce prompt, recording every prompt it saw.
    struct ChunkingFake {
        prompts: Mutex<Vec<String>>,
        window: Option<u32>,
    }

    #[async_trait]
    impl LlmProvider for ChunkingFake {
        async fn generate(&self, prompt: &str) -> Result<String, ProviderError> {
            self.prompts.lock().unwrap().push(prompt.to_string());
            let n = self.prompts.lock().unwrap().len();
            if prompt.contains("PARTIAL RESULTS") {
                return Ok(r#"{"title":"Merged title","summary":"Whole recording.","topics":["a","b"],
                    "people":[],"organizations":[],"key_points":["k"],"action_items":[]}"#.to_string());
            }
            Ok(format!(
                r#"{{"title":"Part {n}","summary":"S{n}","topics":["t{n}"],
                    "people":[{{"name":"Gio","context":"c{n}","resolved_to":"Giovanni"}},
                              {{"name":"Only{n}","context":"x","resolved_to":null}}],
                    "organizations":[{{"name":"Exane","context":"o{n}","resolved_to":"Exein"}}],
                    "key_points":["p{n}"],"action_items":[]}}"#
            ))
        }
        async fn generate_text(&self, _prompt: &str) -> Result<String, ProviderError> {
            Ok(String::new())
        }
        async fn health_check(&self) -> Result<(), ProviderError> {
            Ok(())
        }
        async fn context_window(&self) -> Option<u32> {
            self.window
        }
        fn display_name(&self) -> &str {
            "fake"
        }
        fn model(&self) -> &str {
            "fake"
        }
    }

    #[test]
    fn chunks_respect_size_and_overlap() {
        let text = (1..=100).map(|i| format!("w{i}")).collect::<Vec<_>>().join(" ");
        let chunks = split_into_chunks(&text, 40, 10);
        assert_eq!(chunks.len(), 3);
        let first: Vec<&str> = chunks[0].split_whitespace().collect();
        let second: Vec<&str> = chunks[1].split_whitespace().collect();
        assert_eq!(first.len(), 40);
        assert_eq!(second[0], "w31", "second chunk starts inside the overlap");
        assert!(chunks[2].ends_with("w100"));
        assert_eq!(split_into_chunks("a b c", 10, 2), vec!["a b c".to_string()]);
    }

    #[test]
    fn chunk_size_follows_context_window() {
        assert_eq!(chunk_words_for_window(None, 0), DEFAULT_CHUNK_WORDS);
        // 32k local model with a 2k-char world: (32768*0.6 - 6000 - 500) / 1.4 ≈ 9400
        let local = chunk_words_for_window(Some(32_768), 2_000);
        assert!((9_000..10_000).contains(&local), "{local}");
        // Tiny window clamps to the minimum, huge window to the maximum.
        assert_eq!(chunk_words_for_window(Some(4_096), 0), MIN_CHUNK_WORDS);
        assert_eq!(chunk_words_for_window(Some(1_000_000), 0), MAX_CHUNK_WORDS);
    }

    #[test]
    fn merging_entities_dedupes_by_canonical_name() {
        let a = vec![
            ExtractedEntity { name: "Gio".into(), context: "engineer".into(), resolved_to: Some("Giovanni".into()) },
            ExtractedEntity { name: "Acme".into(), context: "vendor".into(), resolved_to: None },
        ];
        let b = vec![
            ExtractedEntity { name: "giovanni".into(), context: "spoke about budget".into(), resolved_to: None },
            ExtractedEntity { name: "ACME".into(), context: "vendor".into(), resolved_to: None },
        ];
        let merged = merge_entities(&[&a, &b]);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].resolved_to.as_deref(), Some("Giovanni"));
        assert_eq!(merged[0].context, "engineer; spoke about budget");
        assert_eq!(merged[1].context, "vendor", "identical context is not repeated");
    }

    #[test]
    fn schema_matches_result_fields() {
        let schema = extraction_schema();
        let sample = ExtractionResult {
            title: "t".into(), summary: "s".into(), topics: vec![], people: vec![],
            organizations: vec![], key_points: vec![], action_items: vec![],
        };
        let value = serde_json::to_value(&sample).unwrap();
        let mut keys: Vec<&str> = value.as_object().unwrap().keys().map(String::as_str).collect();
        keys.sort();
        let mut required: Vec<&str> = schema["required"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
        required.sort();
        assert_eq!(keys, required);
    }

    #[tokio::test]
    async fn long_transcripts_are_chunked_and_merged() {
        let fake = ChunkingFake { prompts: Mutex::new(Vec::new()), window: Some(4_096) };
        let service = EnrichmentService::new(Box::new(fake));
        // 4k window => MIN_CHUNK_WORDS per chunk; 3,200 words => 3 chunks (with overlap)
        let transcript = (0..3_200).map(|i| format!("word{i}")).collect::<Vec<_>>().join(" ");
        let result = service.extract_with_full_context(&transcript, Some("owner: Giovanni")).await.unwrap();

        assert_eq!(result.title, "Merged title");
        assert_eq!(result.topics, vec!["a", "b"]);
        // Entities merged deterministically: Giovanni + Only1..3, Exein once.
        assert_eq!(result.people.iter().filter(|p| p.resolved_to.as_deref() == Some("Giovanni")).count(), 1);
        assert_eq!(result.people.len(), 4);
        assert_eq!(result.organizations.len(), 1);
        assert_eq!(result.organizations[0].resolved_to.as_deref(), Some("Exein"));
    }

    #[tokio::test]
    async fn short_transcripts_take_a_single_call() {
        let fake = ChunkingFake { prompts: Mutex::new(Vec::new()), window: None };
        let service = EnrichmentService::new(Box::new(fake));
        let result = service.extract_with_full_context("short meeting about budgets", None).await.unwrap();
        assert_eq!(result.title, "Part 1");
    }

    #[test]
    fn test_extraction_result_parsing() {
        let json = r#"{
            "title": "Test Meeting",
            "summary": "A test meeting about things.",
            "topics": ["testing", "meetings"],
            "people": [{"name": "John", "context": "The host"}],
            "organizations": [{"name": "Acme", "context": "The company"}],
            "key_points": ["Point 1"],
            "action_items": ["Do something"]
        }"#;

        let result: ExtractionResult = serde_json::from_str(json).unwrap();
        assert_eq!(result.title, "Test Meeting");
        assert_eq!(result.topics.len(), 2);
        assert_eq!(result.people.len(), 1);
        assert_eq!(result.people[0].name, "John");
        assert!(result.people[0].resolved_to.is_none());
    }

    #[test]
    fn test_extraction_result_with_resolved_to() {
        let json = r#"{
            "title": "Test",
            "summary": "Test",
            "topics": [],
            "people": [{"name": "Gerardo", "context": "discussing budget", "resolved_to": "Gerardo Gagliardo"}],
            "organizations": [{"name": "Exane", "context": "their product", "resolved_to": "Exein"}],
            "key_points": [],
            "action_items": []
        }"#;

        let result: ExtractionResult = serde_json::from_str(json).unwrap();
        assert_eq!(result.people[0].resolved_to.as_deref(), Some("Gerardo Gagliardo"));
        assert_eq!(result.organizations[0].resolved_to.as_deref(), Some("Exein"));
    }

    #[test]
    fn test_all_entities() {
        let result = ExtractionResult {
            title: "Test".to_string(),
            summary: "Test summary".to_string(),
            topics: vec![],
            people: vec![ExtractedEntity {
                name: "John".to_string(),
                context: "Engineer".to_string(),
                resolved_to: None,
            }],
            organizations: vec![ExtractedEntity {
                name: "Acme".to_string(),
                context: "Company".to_string(),
                resolved_to: Some("Acme Corp".to_string()),
            }],
            key_points: vec![],
            action_items: vec![],
        };

        let entities = result.all_entities();
        assert_eq!(entities.len(), 2);
        assert_eq!(entities[0].1, "person");
        assert_eq!(entities[1].1, "organization");
        assert_eq!(entities[1].0.resolved_to.as_deref(), Some("Acme Corp"));
    }
}
