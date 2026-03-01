//! # rig-claude-advanced
//!
//! Anthropic advanced tool use features for [Rig](https://github.com/0xPlaygrounds/rig).
//!
//! This crate provides a wrapper around Rig's Anthropic provider that adds support for:
//!
//! - **Tool use examples** — Attach realistic examples to tool definitions to improve accuracy
//!   (Anthropic reports 72% -> 90% improvement)
//! - **Cache control on tools** — Mark tool definitions for Anthropic prompt caching
//! - **Deferred tool loading** — Send only a tool search tool upfront, loading full
//!   definitions on demand (up to 85% token reduction)
//!
//! The wrapper implements Rig's `CompletionModel` trait, so it works seamlessly with
//! Rig's agent system, hooks, multi-turn conversations, and streaming.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use rig::providers::anthropic;
//! use rig_claude_advanced::{AdvancedClaudeModel, ToolUseExample};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let client: anthropic::Client<reqwest::Client> = anthropic::Client::builder()
//!     .api_key("your-key")
//!     .anthropic_beta("advanced-tool-use-2025-11-20")
//!     .build()?;
//!
//! let model = AdvancedClaudeModel::builder(client, "claude-sonnet-4-5-20250929")
//!     .tool_examples("search_docs", vec![
//!         ToolUseExample::new(
//!             "Find auth docs",
//!             serde_json::json!({"query": "OAuth2", "limit": 5}),
//!             Some(serde_json::json!({"results": []})),
//!         ),
//!     ])
//!     .with_prompt_caching()
//!     .build();
//! # Ok(())
//! # }
//! ```

mod model;
mod request;
mod tool_search;
mod types;

pub use model::{AdvancedClaudeModel, AdvancedClaudeModelBuilder};
pub use tool_search::{ToolSearchArgs, ToolSearchOutput, ToolSearchResult, ToolSearchTool};
pub use types::{AdvancedToolDefinition, ToolMetadata, ToolUseExample};
