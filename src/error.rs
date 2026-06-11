//! Rich error types with recovery strategies.

use std::fmt;

/// Top-level flow error.
#[derive(Debug, Clone)]
pub enum FlowError {
    /// A step failed.
    StepFailed { step: String, error: StepError },
    /// Budget exceeded.
    BudgetExceeded { budget: usize, used: usize },
    /// Workflow definition is invalid.
    InvalidFlow(String),
    /// Checkpoint save/load failed.
    CheckpointError(String),
    /// Tool invocation error.
    ToolError { tool: String, message: String },
    /// Agent execution error.
    AgentError { agent: String, message: String },
    /// Storage error.
    StorageError(String),
    /// Timeout.
    Timeout { step: String, timeout_ms: u64 },
}

/// Step-level error.
#[derive(Debug, Clone)]
pub enum StepError {
    /// Tool not found in registry.
    ToolNotFound(String),
    /// Tool returned an error.
    ToolFailed(String),
    /// Required input not available.
    MissingInput(String),
    /// Output type mismatch.
    TypeMismatch { expected: String, got: String },
    /// Condition evaluation failed.
    ConditionError(String),
    /// Step timed out.
    Timeout,
    /// Step was cancelled.
    Cancelled,
}

impl fmt::Display for FlowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StepFailed { step, error } => write!(f, "step {:?} failed: {}", step, error),
            Self::BudgetExceeded { budget, used } => {
                write!(f, "context budget exceeded: {}/{} tokens", used, budget)
            }
            Self::InvalidFlow(m) => write!(f, "invalid flow: {}", m),
            Self::CheckpointError(m) => write!(f, "checkpoint error: {}", m),
            Self::ToolError { tool, message } => write!(f, "tool {:?} error: {}", tool, message),
            Self::AgentError { agent, message } => {
                write!(f, "agent {:?} error: {}", agent, message)
            }
            Self::StorageError(m) => write!(f, "storage error: {}", m),
            Self::Timeout { step, timeout_ms } => {
                write!(f, "step {:?} timed out after {}ms", step, timeout_ms)
            }
        }
    }
}

impl fmt::Display for StepError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ToolNotFound(t) => write!(f, "tool not found: {:?}", t),
            Self::ToolFailed(m) => write!(f, "tool failed: {}", m),
            Self::MissingInput(k) => write!(f, "missing input: {:?}", k),
            Self::TypeMismatch { expected, got } => {
                write!(f, "type mismatch: expected {}, got {}", expected, got)
            }
            Self::ConditionError(m) => write!(f, "condition error: {}", m),
            Self::Timeout => write!(f, "timed out"),
            Self::Cancelled => write!(f, "cancelled"),
        }
    }
}

impl std::error::Error for FlowError {}
impl std::error::Error for StepError {}

/// Retry policy for failed steps.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    pub max_attempts: usize,
    pub backoff_ms: u64,
    pub max_backoff_ms: u64,
}

impl RetryPolicy {
    pub fn none() -> Self {
        Self {
            max_attempts: 1,
            backoff_ms: 0,
            max_backoff_ms: 0,
        }
    }
    pub fn attempts(n: usize) -> Self {
        Self {
            max_attempts: n,
            backoff_ms: 100,
            max_backoff_ms: 5000,
        }
    }
    pub fn backoff_for(&self, attempt: usize) -> u64 {
        let ms = self.backoff_ms * (1u64 << attempt.min(10));
        ms.min(self.max_backoff_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flow_error_display() {
        let e = FlowError::StepFailed {
            step: "search".into(),
            error: StepError::ToolNotFound("code_search".into()),
        };
        let s = e.to_string();
        assert!(s.contains("search"));
        assert!(s.contains("code_search"));
    }

    #[test]
    fn budget_exceeded() {
        let e = FlowError::BudgetExceeded {
            budget: 4096,
            used: 5000,
        };
        assert!(e.to_string().contains("5000"));
    }

    #[test]
    fn retry_backoff() {
        let p = RetryPolicy::attempts(3);
        assert!(p.backoff_for(1) > p.backoff_for(0));
        assert!(p.backoff_for(100) <= p.max_backoff_ms);
    }

    #[test]
    fn retry_none() {
        let p = RetryPolicy::none();
        assert_eq!(p.max_attempts, 1);
        assert_eq!(p.backoff_for(5), 0);
    }
}
