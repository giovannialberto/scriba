//! Agent module for tool-use chat with Scriba.
//!
//! Provides an autonomous agent loop that can call tools to fetch data
//! from the database, world context, and transcripts before answering.

pub mod loop_runner;
pub mod provider;
pub mod tools;

pub use loop_runner::run_agent_loop;
pub use provider::{AgentProvider, AgentTurnResult, create_agent_provider};
pub use tools::all_tool_definitions;
