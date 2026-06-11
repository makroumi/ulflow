//! Universal LLM gateway.
//!
//! One trait, any model. Switch providers by changing one line.
//!
//! ```rust,ignore
//! let llm = LLM::openai("gpt-4o");
//! let llm = LLM::anthropic("claude-3-5-sonnet-20241022");
//! let llm = LLM::ollama("llama3");
//! let llm = LLM::custom("https://my-api.com/v1", "my-model");
//! ```

pub mod gateway;
pub mod provider;
pub mod types;

pub use gateway::LLM;
pub use types::*;
