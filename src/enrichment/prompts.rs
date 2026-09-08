//! Prompts for knowledge extraction from transcripts.

/// Build the extraction prompt for a transcript.
pub fn build_extraction_prompt(transcript: &str) -> String {
    format!(
        r#"You are a knowledge extraction assistant. Analyze the following transcript and extract structured information.

TRANSCRIPT:
---
{transcript}
---

Extract the following information and return it as a JSON object:

1. "title": A concise, descriptive title for this recording (5-10 words max)
2. "summary": A brief summary of the main content (2-3 sentences)
3. "topics": An array of main topics discussed (3-7 items)
4. "people": An array of INDIVIDUAL HUMAN PERSONS explicitly NAMED in the transcript (NOT companies/organizations), each with:
   - "name": The person's actual name as mentioned in the transcript
   - "context": Brief context about this person from the transcript (role, relationship, etc.)
   - Do NOT include unnamed speakers, narrators, or placeholders like "Author", "Speaker", "Autore", etc.
5. "organizations": An array of companies, organizations, institutions mentioned (e.g., OpenAI, Google, Anthropic, universities, governments), each with:
   - "name": The organization name
   - "context": Brief context about this organization
6. "key_points": An array of key points or insights (3-5 items)
7. "action_items": An array of action items or tasks mentioned (if any)

IMPORTANT: If the same person or organization appears multiple times in the transcript (even with slight spelling variations), include them ONLY ONCE with combined context. Do NOT return duplicate entries.

Return ONLY valid JSON, no additional text. If a field has no relevant content, use an empty array [] or empty string "".

Example JSON structure (DO NOT copy these example values - extract from the actual transcript above):
{{
  "title": "<descriptive title from transcript>",
  "summary": "<summary of actual content>",
  "topics": ["<topic1>", "<topic2>"],
  "people": [
    {{"name": "<person name>", "context": "<their role/context>"}}
  ],
  "organizations": [
    {{"name": "<org name>", "context": "<org context>"}}
  ],
  "key_points": ["<key point from transcript>"],
  "action_items": ["<action item if any>"]
}}

Now analyze the transcript above and return JSON with the ACTUAL content:"#,
        transcript = transcript
    )
}



/// Build the reduce step of chunked extraction: combine the per-chunk results
/// (JSON, in transcript order) into one coherent set of recording-level fields.
pub fn build_extraction_merge_prompt(partials_json: &[String]) -> String {
    let mut parts = String::new();
    for (i, partial) in partials_json.iter().enumerate() {
        parts.push_str(&format!("PART {} OF {}:\n{}\n\n", i + 1, partials_json.len(), partial));
    }
    format!(
        r#"You are a knowledge extraction assistant. A long recording was analyzed in consecutive parts. Below are the PARTIAL RESULTS for each part, in order.

{parts}
Combine them into ONE result for the whole recording and return it as a JSON object:

1. "title": A concise, descriptive title for the whole recording (5-10 words max)
2. "summary": A brief summary of the whole recording (2-4 sentences), not a list of parts
3. "topics": The main topics across all parts, deduplicated (3-7 items)
4. "people": Leave as an empty array [] (entities are merged separately)
5. "organizations": Leave as an empty array []
6. "key_points": The most important points across all parts, deduplicated (3-7 items)
7. "action_items": All action items or tasks mentioned, deduplicated

Return ONLY valid JSON with exactly these keys, no additional text."#,
        parts = parts.trim_end()
    )
}

/// Build a prompt for updating entity context with new information.
pub fn build_context_update_prompt(
    entity_name: &str,
    entity_type: &str,
    existing_context: &str,
    new_mentions: &[(&str, &str)], // (mention_text, context_snippet)
) -> String {
    let mentions_text: String = new_mentions
        .iter()
        .map(|(text, ctx)| format!("- \"{}\" mentioned in: \"{}\"", text, ctx))
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        r#"You are building a knowledge base about people and organizations. Your job is to accumulate factual information about entities over time.

ENTITY TO UPDATE:
- Name: "{entity_name}"
- Type: {entity_type}
- Current known information: "{existing_context}"

NEW TRANSCRIPT EXCERPT(S) mentioning this entity:
{mentions_text}

TASK: Extract NEW factual information about "{entity_name}" from the transcript excerpt(s) and merge it with existing knowledge.

Look for facts like:
- For people: job title, role, company, expertise, relationships, location, responsibilities
- For organizations: industry, products, location, size, what they do

RULES:
1. ONLY include facts that are explicitly stated or clearly implied in the text
2. DO NOT invent or assume information not in the text
3. PRESERVE all existing facts that are still relevant
4. ADD new facts discovered in the transcript
5. If new info contradicts old info, prefer the new info
6. Keep it factual and concise (max 150 words)

Return a JSON object:
{{
  "updated_context": "<merged context with old + new facts>",
  "new_facts": ["<specific new fact 1>", "<specific new fact 2>"]
}}

Return ONLY valid JSON:"#,
        entity_name = entity_name,
        entity_type = entity_type,
        existing_context = existing_context,
        mentions_text = mentions_text
    )
}

/// Build a prompt for updating an entity description with new information.
///
/// Takes what we currently know and new facts, asks the LLM to produce
/// a single updated description.
pub fn build_context_compaction_prompt(
    entity_name: &str,
    _entity_type: &str,
    existing_context: &str,
    new_info: &str,
) -> String {
    format!(
        r#"What we know about "{entity_name}":
{existing_context}

New information:
{new_info}

Write an updated 1-2 sentence description of "{entity_name}" combining everything above. Be concise and factual. Only return the description, nothing else."#,
        entity_name = entity_name,
        existing_context = existing_context,
        new_info = new_info
    )
}

/// Build a prompt for evolving the world description based on a new transcript.
///
/// This prompt asks the LLM to return ONLY the new information to add to the
/// owner's world, as a conservative JSON delta. The merge happens in code.
pub fn build_world_evolution_prompt(
    current_world: &str,
    _transcript: &str,
    extraction_summary: &str,
) -> String {
    format!(
        r#"You maintain a structured knowledge base about a person. Given their current profile and information extracted from a new recording, identify ONLY genuinely new facts to add.

CURRENT WORLD (what we already know):
{current_world}

NEW INFORMATION FROM LATEST RECORDING:
{extraction_summary}

IMPORTANT: Transcripts often contain misspellings or mis-transcriptions of names. Before adding a new person or organization, check carefully if it could be a misspelling of someone/something already in the current world. For example, "Exane" is likely "Exein", "Jon" could be "John". When in doubt, assume it refers to the known entity and do NOT create a duplicate.

YOUR TASK: Return a JSON object containing ONLY new information learned from this recording. Be very conservative:
- Do NOT repeat or rephrase information already in the current world
- Do NOT include entities/organizations already known unless there is genuinely new information about them
- Do NOT create new entries for misspellings of known entities — they are the SAME entity
- Do NOT include transient details (specific numbers, dates, meeting agendas)
- Do NOT invent fields or categories not in the schema below
- ONLY include lasting facts that are NEW: new people, new organizations, new interests, new projects, new beliefs
- If the recording adds nothing meaningful to the world, return an empty object: {{}}

The JSON schema (include only fields with new content):
{{
  "owner": {{
    "name": "only if name needs correction",
    "aliases": ["only NEW aliases not already known"],
    "role": "only if role changed",
    "organization": "only if changed",
    "location": "only if changed"
  }},
  "organizations": [
    {{"name": "org name", "description": "what they do", "aliases": ["misspellings"]}}
  ],
  "people": [
    {{"name": "person name", "relationship": "who they are to the owner", "aliases": ["alternate spellings"]}}
  ],
  "interests": ["only genuinely new recurring interests"],
  "projects": [
    {{"name": "project name", "description": "brief description"}}
  ],
  "beliefs": ["only strong opinions explicitly expressed by the owner"]
}}

RULES:
- Omit any field/section that has no new information
- For people: only include if they are clearly important (not just briefly mentioned)
- For interests: only include if this is a recurring theme, not a one-off topic
- For projects: only include active projects the owner is working on
- For beliefs: only include strong convictions the owner clearly expressed
- Keep descriptions SHORT (under 15 words each)
- ALL string values MUST be plain text, NEVER nested JSON objects or arrays. For example: "relationship": "Giovanni's girlfriend" NOT "relationship": {{"key": "value"}}
- "description" and "relationship" fields are short plain-text sentences, nothing else

Return ONLY valid JSON, nothing else:"#,
        current_world = current_world,
        extraction_summary = extraction_summary
    )
}

/// Build a prompt to extract a structured world profile from a seed description.
///
/// This is used when initializing the world to build the structured JSON
/// from the owner's free-form description.
pub fn build_world_seed_extraction_prompt(world_content: &str) -> String {
    format!(
        r#"You are analyzing a personal description to build a structured profile.

DESCRIPTION:
{world_content}

Extract a structured profile. The first person mentioned is the OWNER of this system.

Return a JSON object matching this EXACT schema:
{{
  "owner": {{
    "name": "full name of the owner",
    "aliases": ["common nicknames or short names"],
    "role": "their job title or role",
    "organization": "their primary organization",
    "location": "where they are based"
  }},
  "organizations": [
    {{"name": "org name", "description": "what they do", "aliases": ["known misspellings"]}}
  ],
  "people": [
    {{"name": "person name", "relationship": "who they are to the owner", "aliases": ["alternate spellings"]}}
  ],
  "interests": [],
  "projects": [],
  "beliefs": []
}}

RULES:
- ONLY extract information explicitly stated in the description
- Do NOT invent or assume information not present
- Leave arrays empty if nothing relevant is mentioned
- Keep descriptions SHORT (under 15 words)
- ALL string values MUST be plain text, NEVER nested JSON objects or arrays

Return ONLY valid JSON, nothing else."#,
        world_content = world_content
    )
}

/// Build a prompt to extract entities from a world description.
///
/// This is used when initializing the world to automatically create
/// entities mentioned in the owner's description.
pub fn build_world_entity_extraction_prompt(world_content: &str) -> String {
    format!(
        r#"You are analyzing a personal world description to extract entities (people and organizations).

WORLD DESCRIPTION:
{world_content}

Extract all people and organizations mentioned. For each entity:
- Identify the canonical name (how they should be referred to)
- Note any aliases or alternative names/spellings mentioned
- Extract context about who they are or what they do

The FIRST person mentioned is typically the OWNER of this system - mark them with "is_owner": true.

Return a JSON object with:
{{
  "people": [
    {{
      "name": "canonical name",
      "aliases": ["alias1", "full name if different"],
      "context": "brief description of who they are",
      "is_owner": true/false
    }}
  ],
  "organizations": [
    {{
      "name": "canonical name",
      "aliases": ["common misspelling", "abbreviation"],
      "context": "brief description of what they do"
    }}
  ]
}}

Return ONLY valid JSON, no additional text."#,
        world_content = world_content
    )
}

/// Build extraction prompt with world context and inline entity resolution.
///
/// The world context contains everything the LLM needs: owner profile,
/// known people, organizations, etc. The LLM resolves entity mentions
/// against the world and indicates matches via `resolved_to`.
pub fn build_full_context_extraction_prompt(
    transcript: &str,
    world_context: Option<&str>,
) -> String {
    if world_context.is_none() || world_context.map(|w| w.trim().is_empty()).unwrap_or(true) {
        return build_extraction_prompt(transcript);
    }

    let world = world_context.unwrap();

    format!(
        r#"You are a knowledge extraction assistant. You have access to the owner's world context, which describes who they are and the people, organizations, and projects they know about.

OWNER'S WORLD:
{world}

Use this world context to understand the transcript, resolve ambiguous or misspelled names, and identify known entities. The owner is involved in every recording as a speaker or listener.

TRANSCRIPT:
---
{transcript}
---

Extract the following information and return it as a JSON object:

1. "title": A concise, descriptive title for this recording (5-10 words max)
2. "summary": A brief summary of the main content (2-3 sentences)
3. "topics": An array of main topics discussed (3-7 items)
4. "people": An array of INDIVIDUAL HUMAN PERSONS explicitly NAMED in the transcript (NOT companies/organizations), each with:
   - "name": The name as it appears in the transcript
   - "context": Brief context about this person from the transcript
   - "resolved_to": If this person matches someone in the OWNER'S WORLD above (even if misspelled or abbreviated), set this to their EXACT name from the world. Otherwise set to null.
   - Do NOT include unnamed speakers, narrators, or placeholders like "Author", "Speaker", "Autore", etc.
5. "organizations": An array of companies, organizations, institutions mentioned, each with:
   - "name": The name as it appears in the transcript
   - "context": Brief context about this organization
   - "resolved_to": If this organization matches one in the OWNER'S WORLD above (even if misspelled — e.g. "Exane" for "Exein"), set this to the EXACT name from the world. Otherwise set to null.
6. "key_points": An array of key points or insights (3-5 items)
7. "action_items": An array of action items or tasks mentioned (if any)

ENTITY RESOLUTION RULES:
- Transcripts often contain misspellings due to speech-to-text errors. "Exane", "Xane", "Hexane" could all be "Exein". Use the world context to resolve these.
- If a name closely resembles a known person or organization from the world, resolve it. Check BOTH names AND aliases in the world context.
- If a name is genuinely new (not in the world at all), set "resolved_to" to null.
- When in doubt, resolve to the known entity rather than creating a new one.
- DEDUPLICATION: If the same person or organization appears multiple times in the transcript (even with slight spelling variations), include them ONLY ONCE with combined context. Do NOT return duplicate entries.

Return ONLY valid JSON, no additional text. If a field has no relevant content, use an empty array [] or empty string "".

{{
  "title": "<title>",
  "summary": "<summary>",
  "topics": ["<topic>"],
  "people": [
    {{"name": "<transcript name>", "context": "<context>", "resolved_to": "<world name or null>"}}
  ],
  "organizations": [
    {{"name": "<transcript name>", "context": "<context>", "resolved_to": "<world name or null>"}}
  ],
  "key_points": ["<point>"],
  "action_items": ["<item>"]
}}

Now analyze the transcript above and return JSON:"#,
        world = world,
        transcript = transcript
    )
}

/// Build a prompt for identifying speakers in a diarized transcript.
///
/// Takes the diarized text with generic labels (Speaker 0, Speaker 1, etc.)
/// and asks the LLM to infer real names from conversational cues and world context.
pub fn build_speaker_identification_prompt(
    diarized_text: &str,
    world_context: Option<&str>,
    num_speakers: usize,
) -> String {
    let world_section = match world_context {
        Some(w) if !w.trim().is_empty() => format!(
            r#"OWNER'S WORLD (people and organizations the owner knows):
{w}

The owner of this recording system is almost certainly one of the speakers. Use the world context to identify them and other known people."#
        ),
        _ => "No world context available. Try to identify speakers from conversational cues only.".to_string(),
    };

    format!(
        r#"You are analyzing a meeting transcript where speakers have been identified by number but not by name. Your job is to figure out who each speaker actually is.

{world_section}

DIARIZED TRANSCRIPT:
---
{diarized_text}
---

There are {num_speakers} distinct speakers labeled Speaker 0 through Speaker {max_speaker}.

IDENTIFICATION STRATEGY:
1. Look for direct address: "Hi Giovanni", "Thanks Sara", "What do you think, Marco?"
2. Look for self-identification: "I'm Giovanni", "This is Sara speaking"
3. Look for role/topic clues: if the world says Giovanni is CTO and a speaker discusses technical architecture, that's likely Giovanni
4. The owner (from the world context) is almost always a participant
5. Check aliases in the world context — speakers may use nicknames

Return a JSON object mapping each speaker label to their identified name, or null if you cannot determine who they are:

{{
  "speakers": {{
    "Speaker 0": "<real name or null>",
    "Speaker 1": "<real name or null>"
  }}
}}

RULES:
- Only identify a speaker if you have reasonable confidence
- If unsure, return null for that speaker (do NOT guess)
- Use the EXACT canonical name from the world context when possible
- Return ONLY valid JSON, nothing else

Return JSON:"#,
        world_section = world_section,
        diarized_text = diarized_text,
        num_speakers = num_speakers,
        max_speaker = num_speakers.saturating_sub(1),
    )
}

/// Build a prompt for resolving unresolved entities using web search results.
///
/// Given entities the LLM couldn't match in the first pass, plus web search
/// results for each, asks the LLM to cross-reference against known world
/// entities and correct misspellings.
pub fn build_search_resolution_prompt(
    unresolved_with_search: &[(String, String, String, String)], // (name, type, context, search_summary)
    world_summary: &str,
) -> String {
    let mut entities_section = String::new();
    for (name, entity_type, context, search_results) in unresolved_with_search {
        entities_section.push_str(&format!(
            r#"
- Name: "{name}"
  Type: {entity_type}
  Transcript context: "{context}"
  Web search results:
{search_results}
"#,
            name = name,
            entity_type = entity_type,
            context = context,
            search_results = search_results
        ));
    }

    format!(
        r#"You are verifying entity names from a speech-to-text transcript. Some names may be misspelled due to transcription errors.

KNOWN ENTITIES (from the owner's world):
{world_summary}

UNRESOLVED ENTITIES (not matched in first pass):
{entities_section}

For each unresolved entity, I searched the web. Cross-reference the search results with the known entities above.

YOUR TASK: For each unresolved entity, determine if it's a misspelling of a known entity or genuinely new.

RULES:
- If web search results suggest the entity is actually a known entity (e.g., search for "Gianni Cuozzi" shows results about "Gianni Cuozzo" who matches a known person), resolve it to the EXACT known entity name.
- If the entity is genuinely new (not a misspelling of any known entity), set the value to null.
- Only resolve to known entities listed above — do not invent new canonical names.

Return ONLY a JSON object mapping each unresolved name to its resolution:

{{
  "resolutions": {{
    "<transcript_name>": "<known_entity_name or null>"
  }}
}}

Return ONLY valid JSON, nothing else."#,
        world_summary = world_summary,
        entities_section = entities_section
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extraction_prompt_contains_transcript() {
        let prompt = build_extraction_prompt("Hello, this is a test transcript.");
        assert!(prompt.contains("Hello, this is a test transcript."));
        assert!(prompt.contains("title"));
        assert!(prompt.contains("summary"));
        assert!(prompt.contains("topics"));
    }

    #[test]
    fn test_world_evolution_prompt() {
        let current = r#"{"owner":{"name":"Giovanni"}}"#;
        let transcript = "Today we met with Luca to discuss the roadmap.";
        let extraction = "People: Luca (team member)";

        let prompt = build_world_evolution_prompt(current, transcript, extraction);

        assert!(prompt.contains("Giovanni"));
        assert!(prompt.contains("Luca"));
        assert!(prompt.contains("ONLY new information"));
        assert!(prompt.contains("conservative"));
    }

    #[test]
    fn test_full_context_extraction_prompt_with_world() {
        let world = r#"{"owner":{"name":"Giovanni"},"organizations":[{"name":"Exein"}]}"#;

        let prompt = build_full_context_extraction_prompt(
            "Meeting transcript here",
            Some(world),
        );

        assert!(prompt.contains("Giovanni"));
        assert!(prompt.contains("Exein"));
        assert!(prompt.contains("Meeting transcript here"));
        assert!(prompt.contains("resolved_to"));
        assert!(prompt.contains("OWNER'S WORLD"));
    }

    #[test]
    fn test_full_context_extraction_prompt_no_context() {
        let prompt = build_full_context_extraction_prompt(
            "Simple transcript",
            None,
        );

        assert!(prompt.contains("Simple transcript"));
        assert!(prompt.contains("title"));
        // Should NOT contain resolved_to (basic prompt)
        assert!(!prompt.contains("resolved_to"));
    }

    #[test]
    fn test_search_resolution_prompt() {
        let unresolved = vec![
            (
                "Gianni Cuozzi".to_string(),
                "person".to_string(),
                "discussing budget".to_string(),
                "    - \"Gianni Cuozzo - LinkedIn\" — Profile of Gianni Cuozzo, CTO".to_string(),
            ),
        ];
        let world_summary = "People: Gianni Cuozzo (aliases: none)\nOrganizations: Exein (aliases: Exane)";

        let prompt = build_search_resolution_prompt(&unresolved, world_summary);

        assert!(prompt.contains("Gianni Cuozzi"));
        assert!(prompt.contains("Gianni Cuozzo"));
        assert!(prompt.contains("Exein"));
        assert!(prompt.contains("resolutions"));
    }
}
