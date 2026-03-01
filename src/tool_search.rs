use std::sync::Arc;

use rig::{completion::ToolDefinition, tool::Tool};
use serde::{Deserialize, Serialize};

/// A built-in tool that lets the model discover deferred tools on demand.
///
/// When tools are marked as deferred via `.defer_tool()`, this tool is
/// automatically added to the agent. The model calls it with a search
/// query to discover matching tool definitions, avoiding the cost of
/// sending all tool schemas upfront.
///
/// Anthropic reports up to 85% token reduction when using this pattern.
pub struct ToolSearchTool {
    /// The full definitions of all deferred tools, searched at runtime.
    deferred_tools: Arc<Vec<ToolDefinition>>,
}

impl ToolSearchTool {
    /// Create a new tool search tool with the given deferred tool definitions.
    pub fn new(deferred_tools: Arc<Vec<ToolDefinition>>) -> Self {
        Self { deferred_tools }
    }
}

/// Input arguments for the tool search tool.
#[derive(Debug, Deserialize)]
pub struct ToolSearchArgs {
    /// The search query to find relevant tools.
    pub query: String,
}

/// A matching tool returned by the search.
#[derive(Debug, Serialize)]
pub struct ToolSearchResult {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// Output of the tool search containing matching tools.
#[derive(Debug, Serialize)]
pub struct ToolSearchOutput {
    pub matches: Vec<ToolSearchResult>,
}

#[derive(Debug, thiserror::Error)]
#[error("Tool search failed: {0}")]
pub struct ToolSearchError(String);

impl Tool for ToolSearchTool {
    const NAME: &'static str = "tool_search";

    type Error = ToolSearchError;
    type Args = ToolSearchArgs;
    type Output = ToolSearchOutput;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "tool_search".to_string(),
            description: "Search for available tools by name or description. \
                Use this when you need a tool that isn't immediately available. \
                Returns matching tool definitions with their parameters."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to find relevant tools by name or description"
                    }
                },
                "required": ["query"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let query_lower = args.query.to_lowercase();
        let query_terms: Vec<&str> = query_lower.split_whitespace().collect();

        let matches: Vec<ToolSearchResult> = self
            .deferred_tools
            .iter()
            .filter(|tool| {
                let name_lower = tool.name.to_lowercase();
                let desc_lower = tool.description.to_lowercase();

                // Match if any query term appears in the tool name or description
                query_terms
                    .iter()
                    .any(|term| name_lower.contains(term) || desc_lower.contains(term))
            })
            .map(|tool| ToolSearchResult {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: tool.parameters.clone(),
            })
            .collect();

        Ok(ToolSearchOutput { matches })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_tool_search_finds_matching_tools() {
        let tools = Arc::new(vec![
            ToolDefinition {
                name: "search_documents".to_string(),
                description: "Search through document storage".to_string(),
                parameters: serde_json::json!({"type": "object"}),
            },
            ToolDefinition {
                name: "analyze_sentiment".to_string(),
                description: "Analyze text sentiment".to_string(),
                parameters: serde_json::json!({"type": "object"}),
            },
            ToolDefinition {
                name: "translate_text".to_string(),
                description: "Translate text between languages".to_string(),
                parameters: serde_json::json!({"type": "object"}),
            },
        ]);

        let search = ToolSearchTool::new(tools);

        let result = search
            .call(ToolSearchArgs {
                query: "search documents".to_string(),
            })
            .await
            .expect("search should succeed");

        assert_eq!(result.matches.len(), 1);
        assert_eq!(result.matches[0].name, "search_documents");
    }

    #[tokio::test]
    async fn test_tool_search_returns_empty_on_no_match() {
        let tools = Arc::new(vec![ToolDefinition {
            name: "calculate".to_string(),
            description: "Perform calculations".to_string(),
            parameters: serde_json::json!({"type": "object"}),
        }]);

        let search = ToolSearchTool::new(tools);

        let result = search
            .call(ToolSearchArgs {
                query: "weather forecast".to_string(),
            })
            .await
            .expect("search should succeed");

        assert!(result.matches.is_empty());
    }

    #[tokio::test]
    async fn test_tool_search_matches_by_description() {
        let tools = Arc::new(vec![ToolDefinition {
            name: "rare_tool".to_string(),
            description: "Analyze images for accessibility compliance".to_string(),
            parameters: serde_json::json!({"type": "object"}),
        }]);

        let search = ToolSearchTool::new(tools);

        let result = search
            .call(ToolSearchArgs {
                query: "accessibility".to_string(),
            })
            .await
            .expect("search should succeed");

        assert_eq!(result.matches.len(), 1);
        assert_eq!(result.matches[0].name, "rare_tool");
    }
}
