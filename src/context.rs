//! Execution context: variables, token budget, step outputs.

use std::collections::HashMap;
use ulmen_core::tokens;

/// Value in the execution context.
#[derive(Debug, Clone, PartialEq)]
pub enum ContextValue {
    Null,
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    List(Vec<ContextValue>),
    Map(HashMap<String, ContextValue>),
    Bytes(Vec<u8>),
}

impl ContextValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Integer(i) => Some(*i),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Float(f) => Some(*f),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Boolean(b) => Some(*b),
            _ => None,
        }
    }
    pub fn is_truthy(&self) -> bool {
        match self {
            Self::Null => false,
            Self::Boolean(b) => *b,
            Self::Integer(i) => *i != 0,
            Self::Float(f) => *f != 0.0,
            Self::String(s) => !s.is_empty(),
            Self::List(l) => !l.is_empty(),
            Self::Map(m) => !m.is_empty(),
            Self::Bytes(b) => !b.is_empty(),
        }
    }

    pub fn token_estimate(&self) -> usize {
        match self {
            Self::Null => 1,
            Self::String(s) => tokens::count_tokens(s),
            Self::Integer(i) => tokens::count_tokens(&i.to_string()),
            Self::Float(f) => tokens::count_tokens(&format!("{:?}", f)),
            Self::Boolean(_) => 1,
            Self::List(l) => l.iter().map(|v| v.token_estimate()).sum::<usize>() + 2,
            Self::Map(m) => {
                m.iter()
                    .map(|(k, v)| tokens::count_tokens(k) + v.token_estimate() + 1)
                    .sum::<usize>()
                    + 2
            }
            Self::Bytes(b) => b.len() / 4 + 1,
        }
    }
}

impl From<String> for ContextValue {
    fn from(s: String) -> Self {
        Self::String(s)
    }
}
impl From<&str> for ContextValue {
    fn from(s: &str) -> Self {
        Self::String(s.to_string())
    }
}
impl From<i64> for ContextValue {
    fn from(i: i64) -> Self {
        Self::Integer(i)
    }
}
impl From<f64> for ContextValue {
    fn from(f: f64) -> Self {
        Self::Float(f)
    }
}
impl From<bool> for ContextValue {
    fn from(b: bool) -> Self {
        Self::Boolean(b)
    }
}

/// Token budget tracking.
#[derive(Debug, Clone)]
pub struct ContextBudget {
    pub total: usize,
    pub used: usize,
}

impl ContextBudget {
    pub fn new(total: usize) -> Self {
        Self { total, used: 0 }
    }
    pub fn unlimited() -> Self {
        Self {
            total: usize::MAX,
            used: 0,
        }
    }
    pub fn remaining(&self) -> usize {
        self.total.saturating_sub(self.used)
    }
    pub fn use_tokens(&mut self, n: usize) -> bool {
        if self.used + n > self.total {
            return false;
        }
        self.used += n;
        true
    }
    pub fn usage_ratio(&self) -> f64 {
        if self.total == 0 || self.total == usize::MAX {
            return 0.0;
        }
        self.used as f64 / self.total as f64
    }
    pub fn is_near_limit(&self, threshold: f64) -> bool {
        self.usage_ratio() >= threshold
    }
}

/// The execution context for a running workflow.
/// Holds variables, step outputs, and token budget.
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    /// Named variables (set by user or steps).
    vars: HashMap<String, ContextValue>,
    /// Step outputs (keyed by step_name.field).
    outputs: HashMap<String, ContextValue>,
    /// Token budget for the entire workflow.
    pub budget: ContextBudget,
    /// Workflow run ID.
    pub run_id: String,
    /// Session ID (links to uldb agent workspace).
    pub session_id: Option<String>,
}

impl ExecutionContext {
    pub fn new(run_id: impl Into<String>, budget: usize) -> Self {
        Self {
            vars: HashMap::new(),
            outputs: HashMap::new(),
            budget: ContextBudget::new(budget),
            run_id: run_id.into(),
            session_id: None,
        }
    }

    pub fn unlimited(run_id: impl Into<String>) -> Self {
        Self {
            vars: HashMap::new(),
            outputs: HashMap::new(),
            budget: ContextBudget::unlimited(),
            run_id: run_id.into(),
            session_id: None,
        }
    }

    pub fn with_session(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Set a named variable.
    pub fn set(&mut self, key: impl Into<String>, value: impl Into<ContextValue>) {
        self.vars.insert(key.into(), value.into());
    }

    /// Get a variable or step output by dot-path.
    /// "my_var" -> looks in vars
    /// "step_name.field" -> looks in outputs
    pub fn get(&self, path: &str) -> Option<&ContextValue> {
        if path.contains('.') {
            self.outputs.get(path)
        } else {
            self.vars.get(path).or_else(|| self.outputs.get(path))
        }
    }

    pub fn get_str(&self, path: &str) -> Option<&str> {
        self.get(path)?.as_str()
    }

    pub fn get_i64(&self, path: &str) -> Option<i64> {
        self.get(path)?.as_i64()
    }

    pub fn get_f64(&self, path: &str) -> Option<f64> {
        self.get(path)?.as_f64()
    }

    /// Record a step output.
    pub fn set_output(&mut self, step: &str, field: &str, value: impl Into<ContextValue>) {
        let key = format!("{}.{}", step, field);
        let val = value.into();
        let tokens = val.token_estimate();
        self.budget.use_tokens(tokens);
        self.outputs.insert(key, val);
    }

    /// Get all outputs for a step.
    pub fn step_outputs(&self, step: &str) -> HashMap<&str, &ContextValue> {
        let prefix = format!("{}.", step);
        self.outputs
            .iter()
            .filter(|(k, _)| k.starts_with(&prefix))
            .map(|(k, v)| (k.strip_prefix(&prefix).unwrap_or(k), v))
            .collect()
    }

    /// Render a template string with context values.
    /// {{var_name}} is replaced with the value.
    pub fn render(&self, template: &str) -> String {
        let mut result = template.to_string();
        for (key, value) in &self.vars {
            let placeholder = format!("{{{{{}}}}}", key);
            if let ContextValue::String(s) = value {
                result = result.replace(&placeholder, s);
            }
        }
        for (key, value) in &self.outputs {
            let placeholder = format!("{{{{{}}}}}", key);
            if let ContextValue::String(s) = value {
                result = result.replace(&placeholder, s);
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_set_get() {
        let mut ctx = ExecutionContext::new("run_1", 4096);
        ctx.set("task", "refactor auth");
        assert_eq!(ctx.get_str("task"), Some("refactor auth"));
    }

    #[test]
    fn context_step_output() {
        let mut ctx = ExecutionContext::new("run_1", 4096);
        ctx.set_output("search", "results", "auth/jwt.py::validate");
        assert_eq!(ctx.get_str("search.results"), Some("auth/jwt.py::validate"));
    }

    #[test]
    fn context_render_template() {
        let mut ctx = ExecutionContext::new("run_1", 4096);
        ctx.set("file", "auth.py");
        ctx.set("concern", "security");
        let rendered = ctx.render("Review {{file}} for {{concern}} issues");
        assert_eq!(rendered, "Review auth.py for security issues");
    }

    #[test]
    fn budget_tracking() {
        let mut ctx = ExecutionContext::new("run_1", 100);
        assert!(ctx.budget.use_tokens(50));
        assert!(ctx.budget.use_tokens(49));
        assert!(!ctx.budget.use_tokens(2)); // exceeds
        assert_eq!(ctx.budget.used, 99);
    }

    #[test]
    fn budget_ratio() {
        let mut b = ContextBudget::new(100);
        b.use_tokens(75);
        assert!((b.usage_ratio() - 0.75).abs() < 0.01);
        assert!(b.is_near_limit(0.7));
        assert!(!b.is_near_limit(0.8));
    }

    #[test]
    fn context_value_truthy() {
        assert!(!ContextValue::Null.is_truthy());
        assert!(!ContextValue::Boolean(false).is_truthy());
        assert!(ContextValue::Boolean(true).is_truthy());
        assert!(!ContextValue::String(String::new()).is_truthy());
        assert!(ContextValue::String("x".into()).is_truthy());
        assert!(ContextValue::Integer(1).is_truthy());
        assert!(!ContextValue::Integer(0).is_truthy());
    }

    #[test]
    fn context_value_tokens() {
        assert!(ContextValue::String("hello world".into()).token_estimate() >= 1);
        assert_eq!(ContextValue::Null.token_estimate(), 1);
    }

    #[test]
    fn unlimited_budget() {
        let mut ctx = ExecutionContext::unlimited("run_1");
        for _ in 0..1000 {
            assert!(ctx.budget.use_tokens(10000));
        }
    }
}
