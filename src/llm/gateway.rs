//! Universal LLM gateway.
//!
//! One constructor, any provider. Switch models with one line.

use crate::llm::provider;
use crate::llm::types::*;

/// Universal LLM handle. One type for all providers.
#[derive(Debug, Clone)]
pub struct LLM {
    config: ProviderConfig,
    provider_type: ProviderType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProviderType {
    OpenAI,
    Anthropic,
    Ollama,
}

impl LLM {
    // ---------------------------------------------------------------
    // Constructors: one per provider family
    // ---------------------------------------------------------------

    /// OpenAI (GPT-4o, GPT-4, GPT-3.5).
    /// Also works with: Azure OpenAI, Groq, Together, Fireworks, Perplexity, vLLM, LiteLLM.
    pub fn openai(model: impl Into<String>) -> Self {
        Self {
            config: ProviderConfig {
                name: "openai".into(),
                base_url: std::env::var("OPENAI_BASE_URL")
                    .unwrap_or_else(|_| "https://api.openai.com/v1".into()),
                api_key: std::env::var("OPENAI_API_KEY").ok(),
                default_model: model.into(),
                headers: Vec::new(),
            },
            provider_type: ProviderType::OpenAI,
        }
    }

    /// Anthropic (Claude 3.5, Claude 3).
    pub fn anthropic(model: impl Into<String>) -> Self {
        Self {
            config: ProviderConfig {
                name: "anthropic".into(),
                base_url: std::env::var("ANTHROPIC_BASE_URL")
                    .unwrap_or_else(|_| "https://api.anthropic.com/v1".into()),
                api_key: std::env::var("ANTHROPIC_API_KEY").ok(),
                default_model: model.into(),
                headers: Vec::new(),
            },
            provider_type: ProviderType::Anthropic,
        }
    }

    /// Ollama (local models: llama3, mistral, codellama, etc).
    pub fn ollama(model: impl Into<String>) -> Self {
        Self {
            config: ProviderConfig {
                name: "ollama".into(),
                base_url: std::env::var("OLLAMA_BASE_URL")
                    .unwrap_or_else(|_| "http://localhost:11434/api".into()),
                api_key: None,
                default_model: model.into(),
                headers: Vec::new(),
            },
            provider_type: ProviderType::Ollama,
        }
    }

    /// Any OpenAI-compatible API with custom base URL.
    pub fn custom(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            config: ProviderConfig {
                name: "custom".into(),
                base_url: base_url.into(),
                api_key: std::env::var("LLM_API_KEY").ok(),
                default_model: model.into(),
                headers: Vec::new(),
            },
            provider_type: ProviderType::OpenAI, // custom APIs use OpenAI format
        }
    }

    // ---------------------------------------------------------------
    // Configuration
    // ---------------------------------------------------------------

    /// Set the API key explicitly (instead of env var).
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.config.api_key = Some(key.into());
        self
    }

    /// Set the base URL explicitly.
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.config.base_url = url.into();
        self
    }

    /// Add a custom header.
    pub fn header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.config.headers.push((key.into(), value.into()));
        self
    }

    /// Get the model name.
    pub fn model(&self) -> &str {
        &self.config.default_model
    }

    /// Get the provider name.
    pub fn provider(&self) -> &str {
        &self.config.name
    }

    // ---------------------------------------------------------------
    // Call the LLM
    // ---------------------------------------------------------------

    /// Send a chat request. This is the universal entry point.
    pub fn chat(&self, request: &ChatRequest) -> Result<ChatResponse, LLMError> {
        match self.provider_type {
            ProviderType::OpenAI => provider::call_openai(&self.config, request),
            ProviderType::Anthropic => provider::call_anthropic(&self.config, request),
            ProviderType::Ollama => provider::call_ollama(&self.config, request),
        }
    }

    /// Simple one-shot: send a user message, get a response.
    pub fn ask(&self, prompt: &str) -> Result<ChatResponse, LLMError> {
        let request = ChatRequest::new(
            &self.config.default_model,
            vec![Message {
                role: Role::User,
                content: prompt.to_string(),
            }],
        );
        self.chat(&request)
    }

    /// Send with system prompt + user message.
    pub fn ask_with_system(&self, system: &str, prompt: &str) -> Result<ChatResponse, LLMError> {
        let mut request = ChatRequest::new(
            &self.config.default_model,
            vec![Message {
                role: Role::User,
                content: prompt.to_string(),
            }],
        );
        request.system = Some(system.to_string());
        self.chat(&request)
    }

    /// Chat with full message history.
    pub fn complete(&self, messages: Vec<Message>) -> Result<ChatResponse, LLMError> {
        let request = ChatRequest::new(&self.config.default_model, messages);
        self.chat(&request)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_constructor() {
        let llm = LLM::openai("gpt-4o");
        assert_eq!(llm.model(), "gpt-4o");
        assert_eq!(llm.provider(), "openai");
    }

    #[test]
    fn anthropic_constructor() {
        let llm = LLM::anthropic("claude-3-5-sonnet-20241022");
        assert_eq!(llm.model(), "claude-3-5-sonnet-20241022");
        assert_eq!(llm.provider(), "anthropic");
    }

    #[test]
    fn ollama_constructor() {
        let llm = LLM::ollama("llama3");
        assert_eq!(llm.model(), "llama3");
        assert_eq!(llm.provider(), "ollama");
    }

    #[test]
    fn custom_constructor() {
        let llm = LLM::custom("https://my-api.com/v1", "my-model")
            .api_key("my-key")
            .header("X-Custom", "value");
        assert_eq!(llm.model(), "my-model");
        assert_eq!(llm.config.api_key, Some("my-key".into()));
        assert_eq!(llm.config.headers.len(), 1);
    }

    #[test]
    fn model_switch() {
        // Prove that switching is one line:
        let _llm1 = LLM::openai("gpt-4o");
        let _llm2 = LLM::anthropic("claude-3-5-sonnet-20241022");
        let _llm3 = LLM::ollama("llama3");
        // All have the same interface: .chat(), .ask(), .complete()
    }
}
