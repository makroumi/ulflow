//! Step abstraction: the atomic unit of a workflow.

use crate::context::ContextValue;
use crate::error::RetryPolicy;
use std::collections::HashMap;
use std::time::Duration;

/// How to resolve an input value for a step.
#[derive(Debug, Clone)]
pub enum Input {
    /// A literal value.
    Literal(ContextValue),
    /// From a context variable: "my_var"
    FromVar(String),
    /// From a step output: "step_name.field"
    FromStep(String),
    /// A template string with {{placeholders}}.
    Template(String),
    /// Concatenate multiple inputs.
    Join(Vec<Input>, String),
}

impl Input {
    pub fn literal(v: impl Into<ContextValue>) -> Self {
        Self::Literal(v.into())
    }
    pub fn from_var(name: impl Into<String>) -> Self {
        Self::FromVar(name.into())
    }
    pub fn from_step(path: impl Into<String>) -> Self {
        Self::FromStep(path.into())
    }
    pub fn template(t: impl Into<String>) -> Self {
        Self::Template(t.into())
    }

    /// Resolve this input against the execution context.
    pub fn resolve(&self, ctx: &crate::context::ExecutionContext) -> Option<ContextValue> {
        match self {
            Self::Literal(v) => Some(v.clone()),
            Self::FromVar(name) => ctx.get(name).cloned(),
            Self::FromStep(path) => ctx.get(path).cloned(),
            Self::Template(t) => Some(ContextValue::String(ctx.render(t))),
            Self::Join(inputs, sep) => {
                let parts: Vec<String> = inputs
                    .iter()
                    .filter_map(|i| i.resolve(ctx))
                    .filter_map(|v| match v {
                        ContextValue::String(s) => Some(s),
                        _ => None,
                    })
                    .collect();
                Some(ContextValue::String(parts.join(sep)))
            }
        }
    }
}

/// The kind of step.
#[derive(Debug, Clone)]
pub enum StepKind {
    /// Call a tool from the ulmcp registry.
    Tool {
        tool_name: String,
        inputs: HashMap<String, Input>,
        output_field: String,
    },
    /// Call an LLM agent with a prompt.
    Agent {
        prompt: String,
        context_inputs: Vec<String>,
        output_field: String,
    },
    /// Conditional branch.
    Condition {
        /// Expression to evaluate (dot-path into context).
        test: String,
        /// Steps to run if true.
        then_steps: Vec<String>,
        /// Steps to run if false.
        else_steps: Vec<String>,
    },
    /// Run steps in parallel.
    Parallel {
        step_names: Vec<String>,
        /// How to merge results: "first" | "all" | "any"
        merge: String,
    },
    /// Transform: apply a function to context values.
    Transform {
        input: String,
        operation: TransformOp,
        output_field: String,
    },
    /// Wait for an external event (via ulmp watch).
    Wait {
        event_pattern: String,
        timeout_ms: Option<u64>,
        output_field: String,
    },
}

/// Transform operations.
#[derive(Debug, Clone)]
pub enum TransformOp {
    /// Extract first element of a list.
    First,
    /// Extract last element.
    Last,
    /// Get element by index.
    Index(usize),
    /// Convert to string.
    ToString,
    /// String slice.
    Slice(usize, Option<usize>),
    /// Count elements.
    Len,
}

/// Status of a completed step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepStatus {
    Pending,
    Running,
    Succeeded,
    Failed,
    Skipped,
    Cancelled,
}

impl std::fmt::Display for StepStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pending => write!(f, "pending"),
            Self::Running => write!(f, "running"),
            Self::Succeeded => write!(f, "succeeded"),
            Self::Failed => write!(f, "failed"),
            Self::Skipped => write!(f, "skipped"),
            Self::Cancelled => write!(f, "cancelled"),
        }
    }
}

/// Result of a completed step.
#[derive(Debug, Clone)]
pub struct StepResult {
    pub step_name: String,
    pub status: StepStatus,
    pub output: Option<ContextValue>,
    pub error: Option<String>,
    pub tokens_used: usize,
    pub latency_ms: u64,
    pub attempts: usize,
}

/// A workflow step definition.
#[derive(Debug, Clone)]
pub struct Step {
    pub name: String,
    pub kind: StepKind,
    pub depends_on: Vec<String>,
    pub retry: RetryPolicy,
    pub timeout: Option<Duration>,
    pub skip_on_error: bool,
    pub condition: Option<String>,
}

impl Step {
    /// Create a tool step.
    pub fn tool(name: impl Into<String>) -> StepBuilder {
        StepBuilder::new(name)
    }

    /// Create an agent step.
    pub fn agent(name: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            kind: StepKind::Agent {
                prompt: prompt.into(),
                context_inputs: Vec::new(),
                output_field: "output".into(),
            },
            depends_on: Vec::new(),
            retry: RetryPolicy::none(),
            timeout: None,
            skip_on_error: false,
            condition: None,
        }
    }

    /// Create a parallel step group.
    pub fn parallel(name: impl Into<String>, steps: Vec<&str>) -> Self {
        Self {
            name: name.into(),
            kind: StepKind::Parallel {
                step_names: steps.iter().map(|s| s.to_string()).collect(),
                merge: "all".into(),
            },
            depends_on: Vec::new(),
            retry: RetryPolicy::none(),
            timeout: None,
            skip_on_error: false,
            condition: None,
        }
    }

    /// Create a condition step.
    pub fn condition(name: impl Into<String>, test: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            kind: StepKind::Condition {
                test: test.into(),
                then_steps: Vec::new(),
                else_steps: Vec::new(),
            },
            depends_on: Vec::new(),
            retry: RetryPolicy::none(),
            timeout: None,
            skip_on_error: false,
            condition: None,
        }
    }
}

/// Builder for tool steps.
pub struct StepBuilder {
    name: String,
    tool_name: String,
    inputs: HashMap<String, Input>,
    output_field: String,
    depends_on: Vec<String>,
    retry: RetryPolicy,
    timeout: Option<Duration>,
    skip_on_error: bool,
    when: Option<String>,
}

impl StepBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            tool_name: String::new(),
            inputs: HashMap::new(),
            output_field: "output".into(),
            depends_on: Vec::new(),
            retry: RetryPolicy::none(),
            timeout: None,
            skip_on_error: false,
            when: None,
        }
    }

    pub fn tool(mut self, name: impl Into<String>) -> Self {
        self.tool_name = name.into();
        self
    }

    pub fn input(mut self, key: impl Into<String>, input: Input) -> Self {
        self.inputs.insert(key.into(), input);
        self
    }

    pub fn input_literal(mut self, key: impl Into<String>, value: impl Into<ContextValue>) -> Self {
        self.inputs.insert(key.into(), Input::Literal(value.into()));
        self
    }

    pub fn output(mut self, field: impl Into<String>) -> Self {
        self.output_field = field.into();
        self
    }

    pub fn depends_on(mut self, step: impl Into<String>) -> Self {
        self.depends_on.push(step.into());
        self
    }

    pub fn retry(mut self, policy: RetryPolicy) -> Self {
        self.retry = policy;
        self
    }

    pub fn timeout(mut self, d: Duration) -> Self {
        self.timeout = Some(d);
        self
    }

    pub fn skip_on_error(mut self) -> Self {
        self.skip_on_error = true;
        self
    }

    pub fn when(mut self, condition: impl Into<String>) -> Self {
        self.when = Some(condition.into());
        self
    }

    pub fn build(self) -> Step {
        Step {
            name: self.name,
            kind: StepKind::Tool {
                tool_name: self.tool_name,
                inputs: self.inputs,
                output_field: self.output_field,
            },
            depends_on: self.depends_on,
            retry: self.retry,
            timeout: self.timeout,
            skip_on_error: self.skip_on_error,
            condition: self.when,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::ExecutionContext;

    #[test]
    fn input_literal() {
        let ctx = ExecutionContext::unlimited("r1");
        let input = Input::literal("hello");
        assert_eq!(
            input.resolve(&ctx),
            Some(ContextValue::String("hello".into()))
        );
    }

    #[test]
    fn input_from_var() {
        let mut ctx = ExecutionContext::unlimited("r1");
        ctx.set("task", "refactor auth");
        let input = Input::from_var("task");
        assert_eq!(
            input.resolve(&ctx),
            Some(ContextValue::String("refactor auth".into()))
        );
    }

    #[test]
    fn input_from_step() {
        let mut ctx = ExecutionContext::unlimited("r1");
        ctx.set_output("search", "result", "auth/jwt.py");
        let input = Input::from_step("search.result");
        assert_eq!(
            input.resolve(&ctx),
            Some(ContextValue::String("auth/jwt.py".into()))
        );
    }

    #[test]
    fn input_template() {
        let mut ctx = ExecutionContext::unlimited("r1");
        ctx.set("file", "auth.py");
        let input = Input::template("Review {{file}} carefully");
        assert_eq!(
            input.resolve(&ctx),
            Some(ContextValue::String("Review auth.py carefully".into()))
        );
    }

    #[test]
    fn input_missing() {
        let ctx = ExecutionContext::unlimited("r1");
        let input = Input::from_var("nonexistent");
        assert_eq!(input.resolve(&ctx), None);
    }

    #[test]
    fn step_builder() {
        let step = Step::tool("search")
            .tool("code_search")
            .input("query", Input::from_var("task"))
            .output("results")
            .depends_on("previous")
            .timeout(Duration::from_secs(5))
            .build();
        assert_eq!(step.name, "search");
        assert!(step.timeout.is_some());
        assert_eq!(step.depends_on, vec!["previous"]);
    }

    #[test]
    fn step_status_display() {
        assert_eq!(StepStatus::Succeeded.to_string(), "succeeded");
        assert_eq!(StepStatus::Failed.to_string(), "failed");
        assert_eq!(StepStatus::Skipped.to_string(), "skipped");
    }
}
