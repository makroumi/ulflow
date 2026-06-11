//! Agent abstraction: LLM agent calls with context management.

use crate::context::{ContextValue, ExecutionContext};

#[derive(Debug, Clone)]
pub struct AgentCall {
    pub name: String,
    pub prompt: String,
    pub model: Option<String>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<usize>,
}

impl AgentCall {
    pub fn new(name: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            prompt: prompt.into(),
            model: None,
            temperature: None,
            max_tokens: None,
        }
    }
    pub fn model(mut self, m: impl Into<String>) -> Self {
        self.model = Some(m.into());
        self
    }
    pub fn temperature(mut self, t: f64) -> Self {
        self.temperature = Some(t);
        self
    }
    pub fn max_tokens(mut self, n: usize) -> Self {
        self.max_tokens = Some(n);
        self
    }
    pub fn render_prompt(&self, ctx: &ExecutionContext) -> String {
        ctx.render(&self.prompt)
    }
}

#[derive(Debug, Clone)]
pub struct AgentResult {
    pub agent_name: String,
    pub output: String,
    pub confidence: f64,
    pub tokens_used: usize,
    pub model: String,
    pub finish_reason: FinishReason,
}

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

impl From<AgentResult> for ContextValue {
    fn from(r: AgentResult) -> Self {
        ContextValue::String(r.output)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoordinationPattern {
    Sequential,
    Parallel,
    Hierarchical,
    Debate,
    Refinement,
}

impl std::fmt::Display for CoordinationPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sequential => write!(f, "sequential"),
            Self::Parallel => write!(f, "parallel"),
            Self::Hierarchical => write!(f, "hierarchical"),
            Self::Debate => write!(f, "debate"),
            Self::Refinement => write!(f, "refinement"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AgentGroup {
    pub name: String,
    pub agents: Vec<AgentCall>,
    pub pattern: CoordinationPattern,
    pub max_rounds: usize,
}

impl AgentGroup {
    pub fn new(name: impl Into<String>, pattern: CoordinationPattern) -> Self {
        Self {
            name: name.into(),
            agents: Vec::new(),
            pattern,
            max_rounds: 1,
        }
    }
    pub fn agent(mut self, call: AgentCall) -> Self {
        self.agents.push(call);
        self
    }
    pub fn max_rounds(mut self, n: usize) -> Self {
        self.max_rounds = n;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::ExecutionContext;

    #[test]
    fn agent_call_builder() {
        let call = AgentCall::new("analyst", "Analyze {{file}}")
            .model("gpt-4o")
            .temperature(0.2)
            .max_tokens(2048);
        assert_eq!(call.model.as_deref(), Some("gpt-4o"));
    }

    #[test]
    fn prompt_rendering() {
        let mut ctx = ExecutionContext::unlimited("r1");
        ctx.set("file", "auth.py");
        let call = AgentCall::new("test", "Review {{file}}");
        assert_eq!(call.render_prompt(&ctx), "Review auth.py");
    }

    #[test]
    fn finish_reason_display() {
        assert_eq!(FinishReason::Stop.to_string(), "stop");
        assert_eq!(FinishReason::MaxTokens.to_string(), "max_tokens");
    }

    #[test]
    fn coordination_patterns() {
        assert_eq!(CoordinationPattern::Parallel.to_string(), "parallel");
        assert_eq!(CoordinationPattern::Debate.to_string(), "debate");
    }
}
