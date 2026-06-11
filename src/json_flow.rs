//! Build workflows from JSON.
//!
//! Lets users define workflows via API without writing Rust.
//!
//! ```json
//! {
//!   "name": "code_review",
//!   "steps": [
//!     {"name": "search", "tool": "code_search", "inputs": {"query": "$task"}},
//!     {"name": "analyze", "agent": "Review {{search.output}} for: {{task}}"}
//!   ]
//! }
//! ```

use crate::error::FlowError;
use crate::step::{Input, Step};
use crate::workflow::Flow;

/// Build a Flow from a JSON value.
pub fn flow_from_json(json: &serde_json::Value) -> Result<Flow, FlowError> {
    let name = json["name"].as_str().unwrap_or("unnamed").to_string();
    let budget = json["context_budget"].as_u64().unwrap_or(8192) as usize;

    let steps_arr = json["steps"]
        .as_array()
        .ok_or_else(|| FlowError::InvalidFlow("missing 'steps' array".into()))?;

    if steps_arr.is_empty() {
        return Err(FlowError::InvalidFlow("'steps' array is empty".into()));
    }

    let mut steps = Vec::new();
    let mut prev_name: Option<String> = None;

    for (i, step_json) in steps_arr.iter().enumerate() {
        let step_name = step_json["name"]
            .as_str()
            .unwrap_or(&format!("step_{}", i))
            .to_string();

        let step = if let Some(tool_name) = step_json["tool"].as_str() {
            // Tool step
            let mut builder = Step::tool(&step_name).tool(tool_name);

            if let Some(inputs) = step_json["inputs"].as_object() {
                for (key, val) in inputs {
                    let input = parse_input(val);
                    builder = builder.input(key.clone(), input);
                }
            }

            if let Some(dep) = step_json["depends_on"].as_str() {
                builder = builder.depends_on(dep);
            } else if let Some(ref prev) = prev_name {
                builder = builder.depends_on(prev.clone());
            }

            builder.build()
        } else if let Some(prompt) = step_json["agent"].as_str() {
            // Agent step
            let mut step = Step::agent(&step_name, prompt);
            if let Some(dep) = step_json["depends_on"].as_str() {
                step.depends_on.push(dep.to_string());
            } else if let Some(ref prev) = prev_name {
                step.depends_on.push(prev.clone());
            }
            step
        } else {
            return Err(FlowError::InvalidFlow(format!(
                "step {} must have 'tool' or 'agent' field",
                i
            )));
        };

        prev_name = Some(step_name);
        steps.push(step);
    }

    // Remove the first step's dependency (it should have none)
    if !steps.is_empty() {
        steps[0].depends_on.clear();
    }

    let mut builder = Flow::pipeline(&name).context_budget(budget);
    for step in steps {
        builder = builder.step(step);
    }
    builder.build()
}

/// Parse an input value from JSON.
/// "$var_name" -> Input::FromVar
/// "{{step.output}}" -> kept as template
/// "literal" -> Input::Literal
fn parse_input(val: &serde_json::Value) -> Input {
    match val {
        serde_json::Value::String(s) => {
            if s.starts_with('$') {
                Input::from_var(&s[1..])
            } else if s.contains("{{") {
                Input::template(s.as_str())
            } else {
                Input::literal(s.clone())
            }
        }
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Input::literal(crate::context::ContextValue::Integer(i))
            } else if let Some(f) = n.as_f64() {
                Input::literal(crate::context::ContextValue::Float(f))
            } else {
                Input::literal("0")
            }
        }
        serde_json::Value::Bool(b) => Input::literal(crate::context::ContextValue::Boolean(*b)),
        _ => Input::literal(""),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_workflow() {
        let json = serde_json::json!({
            "name": "test",
            "steps": [
                {"name": "s1", "tool": "echo", "inputs": {"text": "$task"}},
                {"name": "s2", "agent": "Analyze: {{s1.output}}"}
            ]
        });
        let flow = flow_from_json(&json).unwrap();
        assert_eq!(flow.name, "test");
        assert_eq!(flow.steps.len(), 2);
    }

    #[test]
    fn empty_steps_error() {
        let json = serde_json::json!({"name": "bad", "steps": []});
        assert!(flow_from_json(&json).is_err());
    }

    #[test]
    fn missing_steps_error() {
        let json = serde_json::json!({"name": "bad"});
        assert!(flow_from_json(&json).is_err());
    }

    #[test]
    fn auto_naming() {
        let json = serde_json::json!({
            "name": "auto",
            "steps": [
                {"tool": "echo", "inputs": {"text": "hello"}},
                {"agent": "Review: {{step_0.output}}"}
            ]
        });
        let flow = flow_from_json(&json).unwrap();
        assert_eq!(flow.steps[0].name, "step_0");
        assert_eq!(flow.steps[1].name, "step_1");
    }

    #[test]
    fn var_input() {
        let json = serde_json::json!({
            "name": "vars",
            "steps": [
                {"name": "s1", "tool": "search", "inputs": {"query": "$task", "limit": 5}}
            ]
        });
        let flow = flow_from_json(&json).unwrap();
        assert_eq!(flow.steps.len(), 1);
    }

    #[test]
    fn custom_budget() {
        let json = serde_json::json!({
            "name": "budget",
            "context_budget": 2048,
            "steps": [{"name": "s1", "tool": "echo", "inputs": {"text": "hi"}}]
        });
        let flow = flow_from_json(&json).unwrap();
        assert_eq!(flow.context_budget, 2048);
    }

    #[test]
    fn explicit_depends_on() {
        let json = serde_json::json!({
            "name": "deps",
            "steps": [
                {"name": "a", "tool": "echo", "inputs": {"text": "hi"}},
                {"name": "b", "tool": "echo", "inputs": {"text": "hi"}},
                {"name": "c", "tool": "echo", "inputs": {"text": "hi"}, "depends_on": "a"}
            ]
        });
        let flow = flow_from_json(&json).unwrap();
        assert_eq!(flow.steps[2].depends_on, vec!["a"]);
    }
}
