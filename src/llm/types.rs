//! Universal LLM types. Provider-agnostic.

use serde::{Deserialize, Serialize};

/// A chat message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

/// Message role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::System => write!(f, "system"),
            Self::User => write!(f, "user"),
            Self::Assistant => write!(f, "assistant"),
            Self::Tool => write!(f, "tool"),
        }
    }
}

/// Request to an LLM.
#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<usize>,
    pub system: Option<String>,
    pub stop: Vec<String>,
}

impl ChatRequest {
    pub fn new(model: impl Into<String>, messages: Vec<Message>) -> Self {
        Self {
            model: model.into(),
            messages,
            temperature: None,
            max_tokens: None,
            system: None,
            stop: Vec::new(),
        }
    }
}

/// Response from an LLM.
#[derive(Debug, Clone)]
pub struct ChatResponse {
    pub content: String,
    pub model: String,
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub finish_reason: FinishReason,
}

/// Why the model stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    MaxTokens,
    Error,
}

impl std::fmt::Display for FinishReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stop => write!(f, "stop"),
            Self::MaxTokens => write!(f, "max_tokens"),
            Self::Error => write!(f, "error"),
        }
    }
}

/// LLM provider configuration.
#[derive(Debug, Clone)]
pub struct ProviderConfig {
    pub name: String,
    pub base_url: String,
    pub api_key: Option<String>,
    pub default_model: String,
    pub headers: Vec<(String, String)>,
}

/// Error from an LLM call.
#[derive(Debug, Clone)]
pub struct LLMError {
    pub message: String,
    pub status_code: Option<u16>,
    pub provider: String,
}

impl std::fmt::Display for LLMError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(code) = self.status_code {
            write!(f, "[{}] {} (HTTP {})", self.provider, self.message, code)
        } else {
            write!(f, "[{}] {}", self.provider, self.message)
        }
    }
}

impl std::error::Error for LLMError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_creation() {
        let msg = Message {
            role: Role::User,
            content: "hello".into(),
        };
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.content, "hello");
    }

    #[test]
    fn role_display() {
        assert_eq!(Role::System.to_string(), "system");
        assert_eq!(Role::User.to_string(), "user");
        assert_eq!(Role::Assistant.to_string(), "assistant");
    }

    #[test]
    fn chat_request() {
        let req = ChatRequest::new(
            "gpt-4o",
            vec![Message {
                role: Role::User,
                content: "hi".into(),
            }],
        );
        assert_eq!(req.model, "gpt-4o");
        assert_eq!(req.messages.len(), 1);
    }

    #[test]
    fn finish_reason_display() {
        assert_eq!(FinishReason::Stop.to_string(), "stop");
        assert_eq!(FinishReason::MaxTokens.to_string(), "max_tokens");
    }

    #[test]
    fn llm_error_display() {
        let e = LLMError {
            message: "rate limited".into(),
            status_code: Some(429),
            provider: "openai".into(),
        };
        assert!(e.to_string().contains("429"));
        assert!(e.to_string().contains("openai"));
    }
}
