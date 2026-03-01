use rig::providers::anthropic::completion::CacheControl;
use serde::{Deserialize, Serialize};

/// Enhanced tool definition with Anthropic advanced features.
///
/// Extends the standard Anthropic tool definition with:
/// - `tool_use_examples`: Realistic examples showing correct parameter patterns
/// - `cache_control`: Prompt caching directive for tool schemas
/// - `defer_loading`: Marks tools for deferred loading (tool search pattern)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedToolDefinition {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_use_examples: Option<Vec<ToolUseExample>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub defer_loading: Option<bool>,
}

/// Example showing correct tool usage patterns.
///
/// Attach 1-5 realistic examples to tool definitions to improve
/// the model's accuracy. Anthropic reports 72% -> 90% accuracy improvement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolUseExample {
    pub description: String,
    pub parameters: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<serde_json::Value>,
}

impl ToolUseExample {
    /// Create a new tool use example.
    pub fn new(
        description: impl Into<String>,
        parameters: serde_json::Value,
        output: Option<serde_json::Value>,
    ) -> Self {
        Self {
            description: description.into(),
            parameters,
            output,
        }
    }
}

/// Per-tool advanced metadata, keyed by tool name in the model's configuration.
#[derive(Debug, Clone, Default)]
pub struct ToolMetadata {
    pub cache_control: Option<CacheControl>,
    pub examples: Vec<ToolUseExample>,
    pub deferred: bool,
}
