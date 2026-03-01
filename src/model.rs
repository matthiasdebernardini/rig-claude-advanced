use std::collections::HashMap;
use std::sync::Arc;

use async_stream::stream;
use bytes::Bytes;
use futures::StreamExt;
use rig::{
    completion::{self, CompletionError, CompletionRequest, ToolDefinition},
    http_client::{
        self, HttpClientExt,
        sse::{Event, GenericEventSource},
    },
    message::ReasoningContent,
    providers::anthropic::{
        self,
        completion::{
            CacheControl, CompletionResponse, Content, Message, SystemContent, ToolChoice,
            apply_cache_control,
        },
        streaming::{ContentDelta, PartialUsage, StreamingCompletionResponse, StreamingEvent},
    },
    streaming::{
        self as rig_streaming, RawStreamingChoice, RawStreamingToolCall, StreamingResult,
        ToolCallDeltaContent,
    },
    wasm_compat::*,
};
use serde::Deserialize;
use tracing::{Level, enabled, info_span};
use tracing_futures::Instrument;

use crate::request::{AdvancedCompletionRequest, RequestParams, merge_inplace};
use crate::tool_search::ToolSearchTool;
use crate::types::{ToolMetadata, ToolUseExample};

/// An enhanced Claude completion model that supports Anthropic's advanced tool use features.
///
/// Wraps an Anthropic client and intercepts `CompletionRequest` to enhance tool definitions
/// with examples, cache control, and deferred loading before sending to the API.
///
/// # Features
///
/// - **Tool use examples**: Attach realistic examples to improve accuracy
/// - **Cache control on tools**: Mark tool definitions for prompt caching
/// - **Deferred tool loading**: Send only a tool search tool upfront; load full definitions on demand
///
/// # Example
///
/// ```rust,no_run
/// use rig::providers::anthropic;
/// use rig_claude_advanced::{AdvancedClaudeModel, ToolUseExample};
///
/// let client: anthropic::Client<reqwest::Client> = anthropic::Client::builder()
///     .api_key("your-key")
///     .anthropic_beta("advanced-tool-use-2025-11-20")
///     .build()
///     .unwrap();
///
/// let model = AdvancedClaudeModel::builder(client, "claude-sonnet-4-5-20250929")
///     .tool_examples("search_docs", vec![
///         ToolUseExample::new("Find auth docs", serde_json::json!({"query": "OAuth2"}), None),
///     ])
///     .build();
/// ```
#[derive(Clone)]
pub struct AdvancedClaudeModel<T = reqwest::Client> {
    client: anthropic::Client<T>,
    model: String,
    default_max_tokens: Option<u64>,
    prompt_caching: bool,
    /// Per-tool advanced metadata, keyed by tool name.
    tool_metadata: HashMap<String, ToolMetadata>,
    /// Definitions of deferred tools, used by the built-in tool search.
    deferred_tool_defs: Arc<Vec<ToolDefinition>>,
}

/// Builder for constructing an `AdvancedClaudeModel`.
pub struct AdvancedClaudeModelBuilder<T = reqwest::Client> {
    client: anthropic::Client<T>,
    model: String,
    default_max_tokens: Option<u64>,
    prompt_caching: bool,
    tool_metadata: HashMap<String, ToolMetadata>,
    deferred_tool_defs: Vec<ToolDefinition>,
}

impl<T> AdvancedClaudeModelBuilder<T> {
    /// Add tool use examples for a named tool.
    ///
    /// These examples are serialized in the request as `tool_use_examples` on the
    /// tool definition, showing the model correct parameter patterns.
    pub fn tool_examples(mut self, tool_name: &str, examples: Vec<ToolUseExample>) -> Self {
        self.tool_metadata
            .entry(tool_name.to_string())
            .or_default()
            .examples = examples;
        self
    }

    /// Add cache control to a tool definition.
    ///
    /// Adds `cache_control: {"type": "ephemeral"}` to the tool definition,
    /// enabling Anthropic prompt caching for large tool schemas.
    pub fn tool_cache_control(mut self, tool_name: &str, cache_control: CacheControl) -> Self {
        self.tool_metadata
            .entry(tool_name.to_string())
            .or_default()
            .cache_control = Some(cache_control);
        self
    }

    /// Mark a tool as deferred (not sent in full upfront).
    ///
    /// The tool will have `defer_loading: true` in its definition, and the model
    /// will receive a built-in `tool_search` tool to discover deferred tools on demand.
    ///
    /// Call this with the tool's definition so the tool search can find it.
    pub fn defer_tool(mut self, tool_name: &str, definition: ToolDefinition) -> Self {
        self.tool_metadata
            .entry(tool_name.to_string())
            .or_default()
            .deferred = true;
        self.deferred_tool_defs.push(definition);
        self
    }

    /// Enable automatic prompt caching.
    ///
    /// When enabled, cache_control breakpoints are added to the system prompt
    /// and the last message content block.
    pub fn with_prompt_caching(mut self) -> Self {
        self.prompt_caching = true;
        self
    }

    /// Set the default max tokens for requests.
    pub fn default_max_tokens(mut self, max_tokens: u64) -> Self {
        self.default_max_tokens = Some(max_tokens);
        self
    }

    /// Build the `AdvancedClaudeModel`.
    pub fn build(self) -> AdvancedClaudeModel<T> {
        AdvancedClaudeModel {
            client: self.client,
            model: self.model,
            default_max_tokens: self.default_max_tokens,
            prompt_caching: self.prompt_caching,
            tool_metadata: self.tool_metadata,
            deferred_tool_defs: Arc::new(self.deferred_tool_defs),
        }
    }
}

impl<T> AdvancedClaudeModel<T> {
    /// Create a builder for an `AdvancedClaudeModel`.
    pub fn builder(
        client: anthropic::Client<T>,
        model: impl Into<String>,
    ) -> AdvancedClaudeModelBuilder<T> {
        let model = model.into();
        let default_max_tokens = calculate_max_tokens(&model);

        AdvancedClaudeModelBuilder {
            client,
            model,
            default_max_tokens,
            prompt_caching: false,
            tool_metadata: HashMap::new(),
            deferred_tool_defs: Vec::new(),
        }
    }

    /// Get a `ToolSearchTool` if any tools are deferred.
    ///
    /// Returns `Some(ToolSearchTool)` when deferred tools exist, intended
    /// to be added to the agent alongside normal tools.
    pub fn tool_search_tool(&self) -> Option<ToolSearchTool> {
        if self.deferred_tool_defs.is_empty() {
            None
        } else {
            Some(ToolSearchTool::new(self.deferred_tool_defs.clone()))
        }
    }
}

/// Anthropic requires a `max_tokens` parameter. These defaults match rig-core's logic.
fn calculate_max_tokens(model: &str) -> Option<u64> {
    if model.starts_with("claude-opus-4") {
        Some(32000)
    } else if model.starts_with("claude-sonnet-4") || model.starts_with("claude-3-7-sonnet") {
        Some(64000)
    } else if model.starts_with("claude-3-5-sonnet") || model.starts_with("claude-3-5-haiku") {
        Some(8192)
    } else if model.starts_with("claude-3-opus")
        || model.starts_with("claude-3-sonnet")
        || model.starts_with("claude-3-haiku")
    {
        Some(4096)
    } else {
        None
    }
}

#[derive(Debug, Deserialize)]
struct ApiErrorResponse {
    message: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ApiResponse<T> {
    Message(T),
    Error(ApiErrorResponse),
}

/// State for accumulating tool call JSON chunks during streaming.
#[derive(Default)]
struct ToolCallState {
    name: String,
    id: String,
    internal_call_id: String,
    input_json: String,
}

/// State for accumulating thinking/reasoning chunks during streaming.
#[derive(Default)]
struct ThinkingState {
    thinking: String,
    signature: String,
}

impl<T> completion::CompletionModel for AdvancedClaudeModel<T>
where
    T: HttpClientExt + Clone + Default + WasmCompatSend + WasmCompatSync + 'static,
{
    type Response = CompletionResponse;
    type StreamingResponse = StreamingCompletionResponse;
    type Client = anthropic::Client<T>;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        let model = model.into();
        let default_max_tokens = calculate_max_tokens(&model);
        Self {
            client: client.clone(),
            model,
            default_max_tokens,
            prompt_caching: false,
            tool_metadata: HashMap::new(),
            deferred_tool_defs: Arc::new(Vec::new()),
        }
    }

    async fn completion(
        &self,
        mut completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
        let request_model = completion_request
            .model
            .clone()
            .unwrap_or_else(|| self.model.clone());

        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat",
                gen_ai.operation.name = "chat",
                gen_ai.provider.name = "anthropic_advanced",
                gen_ai.request.model = &request_model,
                gen_ai.system_instructions = &completion_request.preamble,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        // Apply default max_tokens if not set
        if completion_request.max_tokens.is_none() {
            if let Some(tokens) = self.default_max_tokens {
                completion_request.max_tokens = Some(tokens);
            } else {
                return Err(CompletionError::RequestError(
                    "`max_tokens` must be set for Anthropic".into(),
                ));
            }
        }

        let request = AdvancedCompletionRequest::try_from(RequestParams {
            model: &request_model,
            request: completion_request,
            prompt_caching: self.prompt_caching,
            tool_metadata: &self.tool_metadata,
        })?;

        if enabled!(Level::TRACE) {
            tracing::trace!(
                target: "rig::completions",
                "Advanced Anthropic completion request: {}",
                serde_json::to_string_pretty(&request)?
            );
        }

        async move {
            let request: Vec<u8> = serde_json::to_vec(&request)?;

            let req = self
                .client
                .post("/v1/messages")?
                .body(request)
                .map_err(|e| CompletionError::HttpError(e.into()))?;

            let response = self
                .client
                .send::<_, Bytes>(req)
                .await
                .map_err(CompletionError::HttpError)?;

            if response.status().is_success() {
                match serde_json::from_slice::<ApiResponse<CompletionResponse>>(
                    response
                        .into_body()
                        .await
                        .map_err(CompletionError::HttpError)?
                        .to_vec()
                        .as_slice(),
                )? {
                    ApiResponse::Message(completion) => {
                        if enabled!(Level::TRACE) {
                            tracing::trace!(
                                target: "rig::completions",
                                "Advanced Anthropic completion response: {}",
                                serde_json::to_string_pretty(&completion)?
                            );
                        }
                        completion.try_into()
                    }
                    ApiResponse::Error(ApiErrorResponse { message }) => {
                        Err(CompletionError::ResponseError(message))
                    }
                }
            } else {
                let text: String = String::from_utf8_lossy(
                    &response
                        .into_body()
                        .await
                        .map_err(CompletionError::HttpError)?,
                )
                .into();
                Err(CompletionError::ProviderError(text))
            }
        }
        .instrument(span)
        .await
    }

    async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<
        rig_streaming::StreamingCompletionResponse<StreamingCompletionResponse>,
        CompletionError,
    > {
        let request_model = completion_request
            .model
            .clone()
            .unwrap_or_else(|| self.model.clone());

        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat_streaming",
                gen_ai.operation.name = "chat_streaming",
                gen_ai.provider.name = "anthropic_advanced",
                gen_ai.request.model = &request_model,
                gen_ai.system_instructions = &completion_request.preamble,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = &request_model,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        let max_tokens = if let Some(tokens) = completion_request.max_tokens {
            tokens
        } else if let Some(tokens) = self.default_max_tokens {
            tokens
        } else {
            return Err(CompletionError::RequestError(
                "`max_tokens` must be set for Anthropic".into(),
            ));
        };

        // Build the streaming request body using JSON (matching rig-core's streaming approach)
        let mut full_history = vec![];
        if let Some(docs) = completion_request.normalized_documents() {
            full_history.push(docs);
        }
        full_history.extend(completion_request.chat_history);

        let mut messages = full_history
            .into_iter()
            .map(Message::try_from)
            .collect::<Result<Vec<Message>, _>>()?;

        let mut system: Vec<SystemContent> =
            if let Some(preamble) = completion_request.preamble.as_ref() {
                if preamble.is_empty() {
                    vec![]
                } else {
                    vec![SystemContent::Text {
                        text: preamble.clone(),
                        cache_control: None,
                    }]
                }
            } else {
                vec![]
            };

        if self.prompt_caching {
            apply_cache_control(&mut system, &mut messages);
        }

        let mut body = serde_json::json!({
            "model": request_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": true,
        });

        if !system.is_empty() {
            merge_inplace(&mut body, serde_json::json!({ "system": system }));
        }

        if let Some(temperature) = completion_request.temperature {
            merge_inplace(&mut body, serde_json::json!({ "temperature": temperature }));
        }

        if !completion_request.tools.is_empty() {
            // Build enhanced tool definitions
            let tools: Vec<serde_json::Value> = completion_request
                .tools
                .into_iter()
                .map(|tool| {
                    let metadata = self.tool_metadata.get(&tool.name);
                    let mut tool_json = serde_json::json!({
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.parameters,
                    });

                    if let Some(meta) = metadata {
                        if let Some(cc) = &meta.cache_control {
                            tool_json["cache_control"] =
                                serde_json::to_value(cc).unwrap_or_default();
                        }
                        if !meta.examples.is_empty() {
                            tool_json["tool_use_examples"] =
                                serde_json::to_value(&meta.examples).unwrap_or_default();
                        }
                        if meta.deferred {
                            tool_json["defer_loading"] = serde_json::json!(true);
                        }
                    }

                    tool_json
                })
                .collect();

            merge_inplace(
                &mut body,
                serde_json::json!({
                    "tools": tools,
                    "tool_choice": ToolChoice::Auto,
                }),
            );
        }

        if let Some(ref params) = completion_request.additional_params {
            merge_inplace(&mut body, params.clone());
        }

        if enabled!(Level::TRACE) {
            tracing::trace!(
                target: "rig::completions",
                "Advanced Anthropic streaming request: {}",
                serde_json::to_string_pretty(&body)?
            );
        }

        let body: Vec<u8> = serde_json::to_vec(&body)?;

        let req = self
            .client
            .post("/v1/messages")?
            .body(body)
            .map_err(http_client::Error::Protocol)?;

        let sse_stream = GenericEventSource::new(self.client.clone(), req);

        let stream: StreamingResult<StreamingCompletionResponse> = Box::pin(
            stream! {
                let mut current_tool_call: Option<ToolCallState> = None;
                let mut current_thinking: Option<ThinkingState> = None;
                let mut sse_stream = Box::pin(sse_stream);
                let mut input_tokens: u64 = 0;
                let mut final_usage = None;

                while let Some(sse_result) = sse_stream.next().await {
                    match sse_result {
                        Ok(Event::Open) => {}
                        Ok(Event::Message(sse)) => {
                            match serde_json::from_str::<StreamingEvent>(&sse.data) {
                                Ok(event) => {
                                    match &event {
                                        StreamingEvent::MessageStart { message } => {
                                            input_tokens = message.usage.input_tokens;
                                        }
                                        StreamingEvent::MessageDelta { delta, usage } => {
                                            if delta.stop_reason.is_some() {
                                                let usage = PartialUsage {
                                                    output_tokens: usage.output_tokens,
                                                    input_tokens: Some(
                                                        input_tokens
                                                            .try_into()
                                                            .expect("Failed to convert input_tokens to usize"),
                                                    ),
                                                };
                                                final_usage = Some(usage);
                                                break;
                                            }
                                        }
                                        _ => {}
                                    }

                                    if let Some(result) = handle_event(
                                        &event,
                                        &mut current_tool_call,
                                        &mut current_thinking,
                                    ) {
                                        yield result;
                                    }
                                }
                                Err(e) => {
                                    if !sse.data.trim().is_empty() {
                                        yield Err(CompletionError::ResponseError(format!(
                                            "Failed to parse JSON: {} (Data: {})",
                                            e, sse.data
                                        )));
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            yield Err(CompletionError::ProviderError(format!("SSE Error: {e}")));
                            break;
                        }
                    }
                }

                sse_stream.close();

                yield Ok(RawStreamingChoice::FinalResponse(
                    StreamingCompletionResponse {
                        usage: final_usage.unwrap_or_default(),
                    },
                ));
            }
            .instrument(span),
        );

        Ok(rig_streaming::StreamingCompletionResponse::stream(stream))
    }
}

/// Handle a streaming event and produce the appropriate `RawStreamingChoice`.
///
/// Replicated from rig-core's private `handle_event` function.
fn handle_event(
    event: &StreamingEvent,
    current_tool_call: &mut Option<ToolCallState>,
    current_thinking: &mut Option<ThinkingState>,
) -> Option<Result<RawStreamingChoice<StreamingCompletionResponse>, CompletionError>> {
    match event {
        StreamingEvent::ContentBlockDelta { delta, .. } => match delta {
            ContentDelta::TextDelta { text } => {
                if current_tool_call.is_none() {
                    return Some(Ok(RawStreamingChoice::Message(text.clone())));
                }
                None
            }
            ContentDelta::InputJsonDelta { partial_json } => {
                if let Some(tool_call) = current_tool_call {
                    tool_call.input_json.push_str(partial_json);
                    return Some(Ok(RawStreamingChoice::ToolCallDelta {
                        id: tool_call.id.clone(),
                        internal_call_id: tool_call.internal_call_id.clone(),
                        content: ToolCallDeltaContent::Delta(partial_json.clone()),
                    }));
                }
                None
            }
            ContentDelta::ThinkingDelta { thinking } => {
                current_thinking
                    .get_or_insert_with(ThinkingState::default)
                    .thinking
                    .push_str(thinking);

                Some(Ok(RawStreamingChoice::ReasoningDelta {
                    id: None,
                    reasoning: thinking.clone(),
                }))
            }
            ContentDelta::SignatureDelta { signature } => {
                current_thinking
                    .get_or_insert_with(ThinkingState::default)
                    .signature
                    .push_str(signature);
                None
            }
        },
        StreamingEvent::ContentBlockStart { content_block, .. } => match content_block {
            Content::ToolUse { id, name, .. } => {
                let internal_call_id = nanoid::nanoid!();
                *current_tool_call = Some(ToolCallState {
                    name: name.clone(),
                    id: id.clone(),
                    internal_call_id: internal_call_id.clone(),
                    input_json: String::new(),
                });
                Some(Ok(RawStreamingChoice::ToolCallDelta {
                    id: id.clone(),
                    internal_call_id,
                    content: ToolCallDeltaContent::Name(name.clone()),
                }))
            }
            Content::Thinking { .. } => {
                *current_thinking = Some(ThinkingState::default());
                None
            }
            Content::RedactedThinking { data } => Some(Ok(RawStreamingChoice::Reasoning {
                id: None,
                content: ReasoningContent::Redacted { data: data.clone() },
            })),
            _ => None,
        },
        StreamingEvent::ContentBlockStop { .. } => {
            if let Some(thinking_state) = Option::take(current_thinking) {
                if !thinking_state.thinking.is_empty() {
                    let signature = if thinking_state.signature.is_empty() {
                        None
                    } else {
                        Some(thinking_state.signature)
                    };

                    return Some(Ok(RawStreamingChoice::Reasoning {
                        id: None,
                        content: ReasoningContent::Text {
                            text: thinking_state.thinking,
                            signature,
                        },
                    }));
                }
            }

            if let Some(tool_call) = Option::take(current_tool_call) {
                let json_str = if tool_call.input_json.is_empty() {
                    "{}"
                } else {
                    &tool_call.input_json
                };
                match serde_json::from_str(json_str) {
                    Ok(json_value) => {
                        let raw_tool_call =
                            RawStreamingToolCall::new(tool_call.id, tool_call.name, json_value)
                                .with_internal_call_id(tool_call.internal_call_id);
                        Some(Ok(RawStreamingChoice::ToolCall(raw_tool_call)))
                    }
                    Err(e) => Some(Err(CompletionError::from(e))),
                }
            } else {
                None
            }
        }
        StreamingEvent::MessageStart { .. }
        | StreamingEvent::MessageDelta { .. }
        | StreamingEvent::MessageStop
        | StreamingEvent::Ping
        | StreamingEvent::Unknown => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_max_tokens() {
        assert_eq!(calculate_max_tokens("claude-opus-4-0"), Some(32000));
        assert_eq!(
            calculate_max_tokens("claude-sonnet-4-5-20250929"),
            Some(64000)
        );
        assert_eq!(calculate_max_tokens("claude-3-5-sonnet-latest"), Some(8192));
        assert_eq!(calculate_max_tokens("unknown-model"), None);
    }

    #[test]
    fn test_builder_tool_metadata() {
        let client: anthropic::Client<reqwest::Client> = anthropic::Client::builder()
            .api_key("test-key")
            .build()
            .expect("client build should succeed");

        let model = AdvancedClaudeModel::builder(client, "claude-sonnet-4-5-20250929")
            .tool_examples(
                "search",
                vec![ToolUseExample::new(
                    "Search for docs",
                    serde_json::json!({"query": "test"}),
                    None,
                )],
            )
            .tool_cache_control("search", CacheControl::Ephemeral)
            .defer_tool(
                "rare_tool",
                ToolDefinition {
                    name: "rare_tool".to_string(),
                    description: "A rarely used tool".to_string(),
                    parameters: serde_json::json!({"type": "object"}),
                },
            )
            .with_prompt_caching()
            .build();

        assert!(model.prompt_caching);
        assert_eq!(model.tool_metadata.len(), 2);

        let search_meta = model.tool_metadata.get("search").expect("search metadata");
        assert_eq!(search_meta.examples.len(), 1);
        assert!(search_meta.cache_control.is_some());
        assert!(!search_meta.deferred);

        let rare_meta = model
            .tool_metadata
            .get("rare_tool")
            .expect("rare_tool metadata");
        assert!(rare_meta.deferred);

        assert_eq!(model.deferred_tool_defs.len(), 1);
        assert!(model.tool_search_tool().is_some());
    }

    #[test]
    fn test_no_deferred_tools_no_search() {
        let client: anthropic::Client<reqwest::Client> = anthropic::Client::builder()
            .api_key("test-key")
            .build()
            .expect("client build should succeed");

        let model = AdvancedClaudeModel::builder(client, "claude-sonnet-4-5-20250929").build();

        assert!(model.tool_search_tool().is_none());
    }
}
