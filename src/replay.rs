//! Deterministic replay system.
//!
//! Record every step's inputs, outputs, and LLM responses during a run.
//! Replay the run later using recorded data instead of live calls.
//! Compare runs to understand behavior changes.
//!
//! ```rust,ignore
//! // Record a run
//! let mut recorder = RunRecorder::new("run_001");
//! recorder.record_step("search", &input, &output, 150);
//! recorder.record_llm("analyze", &prompt, &response, 500);
//! let recording = recorder.finish();
//!
//! // Replay without LLM calls
//! let replayer = RunReplayer::from_recording(&recording);
//! let response = replayer.get_llm_response("analyze", &prompt);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;


// ---------------------------------------------------------------------------
// Recording
// ---------------------------------------------------------------------------

/// A recorded step execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepRecord {
    pub step_name: String,
    pub step_type: String,
    pub input: HashMap<String, String>,
    pub output: Option<String>,
    pub tokens_used: usize,
    pub latency_ms: u64,
    pub status: String,
    pub error: Option<String>,
    pub attempt: usize,
}

/// A recorded LLM call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMRecord {
    pub step_name: String,
    pub prompt: String,
    pub response: String,
    pub model: String,
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub latency_ms: u64,
}

/// A complete run recording.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunRecording {
    pub run_id: String,
    pub flow_name: String,
    pub timestamp_ms: u64,
    pub steps: Vec<StepRecord>,
    pub llm_calls: Vec<LLMRecord>,
    pub total_tokens: usize,
    pub total_latency_ms: u64,
    pub status: String,
    pub input_vars: HashMap<String, String>,
}

impl RunRecording {
    /// Serialize to JSON bytes for storage.
    pub fn to_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).unwrap_or_default()
    }

    /// Deserialize from JSON bytes.
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        serde_json::from_slice(data).ok()
    }

    /// Get the LLM response for a step+prompt (for replay).
    pub fn get_llm_response(&self, step_name: &str) -> Option<&LLMRecord> {
        self.llm_calls.iter().find(|r| r.step_name == step_name)
    }

    /// Get a step record by name.
    pub fn get_step(&self, step_name: &str) -> Option<&StepRecord> {
        self.steps.iter().find(|s| s.step_name == step_name)
    }

    /// Summary for display.
    pub fn summary(&self) -> String {
        format!(
            "run={} flow={} steps={} llm_calls={} tokens={} latency={}ms status={}",
            self.run_id,
            self.flow_name,
            self.steps.len(),
            self.llm_calls.len(),
            self.total_tokens,
            self.total_latency_ms,
            self.status
        )
    }
}

// ---------------------------------------------------------------------------
// Recorder
// ---------------------------------------------------------------------------

/// Records a run as it executes.
pub struct RunRecorder {
    run_id: String,
    flow_name: String,
    start_ms: u64,
    steps: Vec<StepRecord>,
    llm_calls: Vec<LLMRecord>,
    input_vars: HashMap<String, String>,
}

impl RunRecorder {
    pub fn new(run_id: impl Into<String>, flow_name: impl Into<String>) -> Self {
        Self {
            run_id: run_id.into(),
            flow_name: flow_name.into(),
            start_ms: now_ms(),
            steps: Vec::new(),
            llm_calls: Vec::new(),
            input_vars: HashMap::new(),
        }
    }

    /// Record input variables.
    pub fn set_input(&mut self, key: &str, value: &str) {
        self.input_vars.insert(key.into(), value.into());
    }

    /// Record a step execution.
    pub fn record_step(
        &mut self,
        step_name: &str,
        step_type: &str,
        input: HashMap<String, String>,
        output: Option<String>,
        tokens: usize,
        latency_ms: u64,
        status: &str,
        error: Option<String>,
    ) {
        self.steps.push(StepRecord {
            step_name: step_name.into(),
            step_type: step_type.into(),
            input,
            output,
            tokens_used: tokens,
            latency_ms,
            status: status.into(),
            error,
            attempt: 1,
        });
    }

    /// Record an LLM call.
    pub fn record_llm(
        &mut self,
        step_name: &str,
        prompt: &str,
        response: &str,
        model: &str,
        input_tokens: usize,
        output_tokens: usize,
        latency_ms: u64,
    ) {
        self.llm_calls.push(LLMRecord {
            step_name: step_name.into(),
            prompt: prompt.into(),
            response: response.into(),
            model: model.into(),
            input_tokens,
            output_tokens,
            latency_ms,
        });
    }

    /// Finish recording and return the complete recording.
    pub fn finish(self, status: &str) -> RunRecording {
        let total_tokens: usize = self.steps.iter().map(|s| s.tokens_used).sum();
        let total_latency = now_ms() - self.start_ms;

        RunRecording {
            run_id: self.run_id,
            flow_name: self.flow_name,
            timestamp_ms: self.start_ms,
            steps: self.steps,
            llm_calls: self.llm_calls,
            total_tokens,
            total_latency_ms: total_latency,
            status: status.into(),
            input_vars: self.input_vars,
        }
    }
}

// ---------------------------------------------------------------------------
// Replayer
// ---------------------------------------------------------------------------

/// Replay a recorded run using stored LLM responses.
pub struct RunReplayer {
    recording: RunRecording,
}

impl RunReplayer {
    pub fn from_recording(recording: RunRecording) -> Self {
        Self { recording }
    }

    /// Get the recorded LLM response for a step.
    /// Returns the response text if found, None if the step wasn't recorded.
    pub fn get_llm_response(&self, step_name: &str) -> Option<String> {
        self.recording
            .get_llm_response(step_name)
            .map(|r| r.response.clone())
    }

    /// Get the recorded tool output for a step.
    pub fn get_tool_output(&self, step_name: &str) -> Option<String> {
        self.recording
            .get_step(step_name)
            .and_then(|s| s.output.clone())
    }

    /// Check if a step was recorded.
    pub fn has_step(&self, step_name: &str) -> bool {
        self.recording.get_step(step_name).is_some()
    }

    /// Get the original input vars.
    pub fn input_vars(&self) -> &HashMap<String, String> {
        &self.recording.input_vars
    }

    /// Get the recording.
    pub fn recording(&self) -> &RunRecording {
        &self.recording
    }
}

// ---------------------------------------------------------------------------
// Run Diff
// ---------------------------------------------------------------------------

/// Difference between two runs.
#[derive(Debug, Serialize)]
pub struct RunDiff {
    pub run_a: String,
    pub run_b: String,
    pub step_diffs: Vec<StepDiff>,
    pub tokens_diff: i64,
    pub latency_diff: i64,
}

/// Difference in a single step between two runs.
#[derive(Debug, Serialize)]
pub struct StepDiff {
    pub step_name: String,
    pub status_a: String,
    pub status_b: String,
    pub output_changed: bool,
    pub tokens_diff: i64,
    pub latency_diff: i64,
}

/// Compare two run recordings.
pub fn diff_runs(a: &RunRecording, b: &RunRecording) -> RunDiff {
    let mut step_diffs = Vec::new();

    // Compare steps that exist in both
    for step_a in &a.steps {
        if let Some(step_b) = b.get_step(&step_a.step_name) {
            step_diffs.push(StepDiff {
                step_name: step_a.step_name.clone(),
                status_a: step_a.status.clone(),
                status_b: step_b.status.clone(),
                output_changed: step_a.output != step_b.output,
                tokens_diff: step_b.tokens_used as i64 - step_a.tokens_used as i64,
                latency_diff: step_b.latency_ms as i64 - step_a.latency_ms as i64,
            });
        } else {
            step_diffs.push(StepDiff {
                step_name: step_a.step_name.clone(),
                status_a: step_a.status.clone(),
                status_b: "missing".into(),
                output_changed: true,
                tokens_diff: -(step_a.tokens_used as i64),
                latency_diff: -(step_a.latency_ms as i64),
            });
        }
    }

    // Steps only in B
    for step_b in &b.steps {
        if a.get_step(&step_b.step_name).is_none() {
            step_diffs.push(StepDiff {
                step_name: step_b.step_name.clone(),
                status_a: "missing".into(),
                status_b: step_b.status.clone(),
                output_changed: true,
                tokens_diff: step_b.tokens_used as i64,
                latency_diff: step_b.latency_ms as i64,
            });
        }
    }

    RunDiff {
        run_a: a.run_id.clone(),
        run_b: b.run_id.clone(),
        step_diffs,
        tokens_diff: b.total_tokens as i64 - a.total_tokens as i64,
        latency_diff: b.total_latency_ms as i64 - a.total_latency_ms as i64,
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recorder_basic() {
        let mut rec = RunRecorder::new("run_1", "my_flow");
        rec.set_input("task", "review auth");
        rec.record_step(
            "search",
            "tool",
            HashMap::new(),
            Some("found 3 results".into()),
            50,
            100,
            "succeeded",
            None,
        );
        rec.record_llm(
            "analyze",
            "Review this code",
            "The code has a bug",
            "gpt-4o",
            100,
            50,
            500,
        );

        let recording = rec.finish("succeeded");
        assert_eq!(recording.run_id, "run_1");
        assert_eq!(recording.flow_name, "my_flow");
        assert_eq!(recording.steps.len(), 1);
        assert_eq!(recording.llm_calls.len(), 1);
        assert_eq!(recording.status, "succeeded");
        assert_eq!(
            recording.input_vars.get("task").map(|s| s.as_str()),
            Some("review auth")
        );
    }

    #[test]
    fn recording_serialization() {
        let mut rec = RunRecorder::new("run_2", "flow");
        rec.record_step(
            "s1",
            "tool",
            HashMap::new(),
            Some("output".into()),
            10,
            50,
            "succeeded",
            None,
        );
        let recording = rec.finish("succeeded");

        let bytes = recording.to_bytes();
        let restored = RunRecording::from_bytes(&bytes).unwrap();
        assert_eq!(restored.run_id, "run_2");
        assert_eq!(restored.steps.len(), 1);
    }

    #[test]
    fn replayer_gets_responses() {
        let mut rec = RunRecorder::new("run_3", "flow");
        rec.record_llm("analyze", "prompt", "response text", "gpt-4o", 50, 25, 300);
        rec.record_step(
            "search",
            "tool",
            HashMap::new(),
            Some("search result".into()),
            10,
            50,
            "succeeded",
            None,
        );
        let recording = rec.finish("succeeded");

        let replayer = RunReplayer::from_recording(recording);
        assert_eq!(
            replayer.get_llm_response("analyze"),
            Some("response text".into())
        );
        assert_eq!(
            replayer.get_tool_output("search"),
            Some("search result".into())
        );
        assert!(replayer.has_step("search"));
        assert!(!replayer.has_step("nonexistent"));
    }

    #[test]
    fn diff_identical_runs() {
        let mut rec1 = RunRecorder::new("a", "flow");
        rec1.record_step(
            "s1",
            "tool",
            HashMap::new(),
            Some("out".into()),
            10,
            50,
            "succeeded",
            None,
        );
        let r1 = rec1.finish("succeeded");

        let mut rec2 = RunRecorder::new("b", "flow");
        rec2.record_step(
            "s1",
            "tool",
            HashMap::new(),
            Some("out".into()),
            10,
            50,
            "succeeded",
            None,
        );
        let r2 = rec2.finish("succeeded");

        let diff = diff_runs(&r1, &r2);
        assert_eq!(diff.step_diffs.len(), 1);
        assert!(!diff.step_diffs[0].output_changed);
        assert_eq!(diff.step_diffs[0].tokens_diff, 0);
    }

    #[test]
    fn diff_changed_output() {
        let mut rec1 = RunRecorder::new("a", "flow");
        rec1.record_step(
            "s1",
            "tool",
            HashMap::new(),
            Some("old output".into()),
            10,
            50,
            "succeeded",
            None,
        );
        let r1 = rec1.finish("succeeded");

        let mut rec2 = RunRecorder::new("b", "flow");
        rec2.record_step(
            "s1",
            "tool",
            HashMap::new(),
            Some("new output".into()),
            15,
            60,
            "succeeded",
            None,
        );
        let r2 = rec2.finish("succeeded");

        let diff = diff_runs(&r1, &r2);
        assert!(diff.step_diffs[0].output_changed);
        assert_eq!(diff.step_diffs[0].tokens_diff, 5);
    }

    #[test]
    fn diff_missing_step() {
        let mut rec1 = RunRecorder::new("a", "flow");
        rec1.record_step(
            "s1",
            "tool",
            HashMap::new(),
            None,
            10,
            50,
            "succeeded",
            None,
        );
        rec1.record_step(
            "s2",
            "tool",
            HashMap::new(),
            None,
            10,
            50,
            "succeeded",
            None,
        );
        let r1 = rec1.finish("succeeded");

        let mut rec2 = RunRecorder::new("b", "flow");
        rec2.record_step(
            "s1",
            "tool",
            HashMap::new(),
            None,
            10,
            50,
            "succeeded",
            None,
        );
        let r2 = rec2.finish("succeeded");

        let diff = diff_runs(&r1, &r2);
        assert_eq!(diff.step_diffs.len(), 2);
        assert_eq!(diff.step_diffs[1].status_b, "missing");
    }

    #[test]
    fn recording_summary() {
        let mut rec = RunRecorder::new("run_x", "code_review");
        rec.record_step(
            "s1",
            "tool",
            HashMap::new(),
            None,
            100,
            200,
            "succeeded",
            None,
        );
        let recording = rec.finish("succeeded");
        let summary = recording.summary();
        assert!(summary.contains("run_x"));
        assert!(summary.contains("code_review"));
    }

    #[test]
    fn recording_get_step() {
        let mut rec = RunRecorder::new("r", "f");
        rec.record_step(
            "search",
            "tool",
            HashMap::new(),
            Some("data".into()),
            10,
            50,
            "ok",
            None,
        );
        let recording = rec.finish("ok");
        assert!(recording.get_step("search").is_some());
        assert!(recording.get_step("missing").is_none());
    }
}
