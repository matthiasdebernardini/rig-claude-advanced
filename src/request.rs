use std::collections::HashMap;

use rig::{
    completion::{CompletionError, CompletionRequest},
    providers::anthropic::completion::{Message, SystemContent, ToolChoice, apply_cache_control},
};
use serde::{Deserialize, Serialize};

use crate::types::{AdvancedToolDefinition, ToolMetadata};

/// Output format specifier for Anthropic's structured output.
#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OutputFormat {
    JsonSchema { schema: serde_json::Value },
}

/// Configuration for the model's output format.
#[derive(Debug, Deserialize, Serialize)]
struct OutputConfig {
    format: OutputFormat,
}

/// Enhanced Anthropic completion request with advanced tool features.
///
/// Replicates `rig-core`'s private `AnthropicCompletionRequest` but uses
/// `AdvancedToolDefinition` instead of the standard `ToolDefinition`,
/// enabling tool use examples, cache control on tools, and deferred loading.
#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct AdvancedCompletionRequest {
    model: String,
    messages: Vec<Message>,
    max_tokens: u64,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    system: Vec<SystemContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<AdvancedToolDefinition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_config: Option<OutputConfig>,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    additional_params: Option<serde_json::Value>,
}

/// Parameters for building an `AdvancedCompletionRequest`.
pub(crate) struct RequestParams<'a> {
    pub model: &'a str,
    pub request: CompletionRequest,
    pub prompt_caching: bool,
    pub tool_metadata: &'a HashMap<String, ToolMetadata>,
}

impl TryFrom<RequestParams<'_>> for AdvancedCompletionRequest {
    type Error = CompletionError;

    fn try_from(params: RequestParams<'_>) -> Result<Self, Self::Error> {
        let RequestParams {
            model,
            request: req,
            prompt_caching,
            tool_metadata,
        } = params;

        let Some(max_tokens) = req.max_tokens else {
            return Err(CompletionError::RequestError(
                "`max_tokens` must be set for Anthropic".into(),
            ));
        };

        let mut full_history = vec![];
        if let Some(docs) = req.normalized_documents() {
            full_history.push(docs);
        }
        full_history.extend(req.chat_history);

        let mut messages = full_history
            .into_iter()
            .map(Message::try_from)
            .collect::<Result<Vec<Message>, _>>()?;

        // Map tools through metadata to add examples/cache_control/defer_loading
        let tools = req
            .tools
            .into_iter()
            .map(|tool| {
                let metadata = tool_metadata.get(&tool.name);
                AdvancedToolDefinition {
                    name: tool.name,
                    description: Some(tool.description),
                    input_schema: tool.parameters,
                    cache_control: metadata.and_then(|m| m.cache_control.clone()),
                    tool_use_examples: metadata.and_then(|m| {
                        if m.examples.is_empty() {
                            None
                        } else {
                            Some(m.examples.clone())
                        }
                    }),
                    defer_loading: metadata
                        .and_then(|m| if m.deferred { Some(true) } else { None }),
                }
            })
            .collect::<Vec<_>>();

        let mut system = if let Some(preamble) = req.preamble {
            if preamble.is_empty() {
                vec![]
            } else {
                vec![SystemContent::Text {
                    text: preamble,
                    cache_control: None,
                }]
            }
        } else {
            vec![]
        };

        if prompt_caching {
            apply_cache_control(&mut system, &mut messages);
        }

        let output_config = req.output_schema.map(|schema| {
            let mut schema_value = schema.to_value();
            sanitize_schema(&mut schema_value);
            OutputConfig {
                format: OutputFormat::JsonSchema {
                    schema: schema_value,
                },
            }
        });

        Ok(Self {
            model: model.to_string(),
            messages,
            max_tokens,
            system,
            temperature: req.temperature,
            tool_choice: req.tool_choice.and_then(|x| ToolChoice::try_from(x).ok()),
            tools,
            output_config,
            additional_params: req.additional_params,
        })
    }
}

/// Recursively sanitize JSON schemas to comply with Anthropic's structured output restrictions:
/// - `additionalProperties` must be `false` on every object
/// - All properties must be listed in `required`
/// - Numeric constraints are stripped
///
/// Replicated from rig-core (private function).
fn sanitize_schema(schema: &mut serde_json::Value) {
    use serde_json::Value;

    if let Value::Object(obj) = schema {
        let is_object_schema = obj.get("type") == Some(&Value::String("object".to_string()))
            || obj.contains_key("properties");

        if is_object_schema && !obj.contains_key("additionalProperties") {
            obj.insert("additionalProperties".to_string(), Value::Bool(false));
        }

        if let Some(Value::Object(properties)) = obj.get("properties") {
            let prop_keys = properties.keys().cloned().map(Value::String).collect();
            obj.insert("required".to_string(), Value::Array(prop_keys));
        }

        let is_numeric_schema = obj.get("type") == Some(&Value::String("integer".to_string()))
            || obj.get("type") == Some(&Value::String("number".to_string()));

        if is_numeric_schema {
            for key in [
                "minimum",
                "maximum",
                "exclusiveMinimum",
                "exclusiveMaximum",
                "multipleOf",
            ] {
                obj.remove(key);
            }
        }

        if let Some(Value::Object(defs_obj)) = obj.get_mut("$defs") {
            for (_, def_schema) in defs_obj.iter_mut() {
                sanitize_schema(def_schema);
            }
        }

        if let Some(Value::Object(props)) = obj.get_mut("properties") {
            for (_, prop_value) in props.iter_mut() {
                sanitize_schema(prop_value);
            }
        }

        if let Some(items) = obj.get_mut("items") {
            sanitize_schema(items);
        }

        for key in ["anyOf", "oneOf", "allOf"] {
            if let Some(Value::Array(variants_array)) = obj.get_mut(key) {
                for variant in variants_array.iter_mut() {
                    sanitize_schema(variant);
                }
            }
        }
    }
}

/// Merge JSON object `b` into object `a` in-place.
///
/// Replicated from rig-core (crate-private function).
pub(crate) fn merge_inplace(a: &mut serde_json::Value, b: serde_json::Value) {
    if let (serde_json::Value::Object(a_map), serde_json::Value::Object(b_map)) = (a, b) {
        b_map.into_iter().for_each(|(key, value)| {
            a_map.insert(key, value);
        });
    }
}
