//! Resource scheduler: decides how, when, and with what resources execution happens.
//!
//! Features:
//!   - Priority-based step ordering
//!   - Multi-model LLM routing (cheap for search, expensive for analysis)
//!   - Retry with exponential backoff
//!   - Timeout enforcement
//!   - Backpressure (configurable concurrency limit)
//!   - Cost tracking per model
//!
//! The scheduler sits between the FlowRunner and actual execution.
//! It wraps step execution with retry, timeout, and routing logic.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::error::RetryPolicy;
use crate::llm::LLM;

// ---------------------------------------------------------------------------
// Priority
// ---------------------------------------------------------------------------

/// Step priority (lower number = higher priority).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Critical = 0,
    High = 1,
    Normal = 2,
    Low = 3,
    Background = 4,
}

impl Priority {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "critical" => Self::Critical,
            "high" => Self::High,
            "low" => Self::Low,
            "background" => Self::Background,
            _ => Self::Normal,
        }
    }
}

impl std::fmt::Display for Priority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Critical => write!(f, "critical"),
            Self::High => write!(f, "high"),
            Self::Normal => write!(f, "normal"),
            Self::Low => write!(f, "low"),
            Self::Background => write!(f, "background"),
        }
    }
}

// ---------------------------------------------------------------------------
// Model Router
// ---------------------------------------------------------------------------

/// Route LLM calls to different models based on the task.
#[derive(Debug, Clone)]
pub struct ModelRoute {
    pub pattern: String,
    pub model: String,
    pub provider: String,
}

/// Multi-model router: picks the right model for each step.
#[derive(Debug, Clone)]
pub struct ModelRouter {
    routes: Vec<ModelRoute>,
    default_provider: String,
    default_model: String,
}

impl ModelRouter {
    pub fn new(default_provider: &str, default_model: &str) -> Self {
        Self {
            routes: Vec::new(),
            default_provider: default_provider.into(),
            default_model: default_model.into(),
        }
    }

    /// Add a routing rule: steps matching pattern use this model.
    pub fn route(
        mut self,
        pattern: impl Into<String>,
        provider: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        self.routes.push(ModelRoute {
            pattern: pattern.into(),
            provider: provider.into(),
            model: model.into(),
        });
        self
    }

    /// Pick the model for a step name.
    pub fn resolve(&self, step_name: &str) -> (&str, &str) {
        for route in &self.routes {
            if step_name.contains(&route.pattern) {
                return (&route.provider, &route.model);
            }
        }
        (&self.default_provider, &self.default_model)
    }

    /// Build an LLM for the resolved model.
    pub fn llm_for(&self, step_name: &str) -> LLM {
        let (provider, model) = self.resolve(step_name);
        let llm = match provider {
            "openai" => LLM::openai(model),
            "anthropic" => LLM::anthropic(model),
            "groq" => LLM::groq(model),
            "gemini" => LLM::gemini(model),
            "ollama" => LLM::ollama(model),
            "together" => LLM::together(model),
            "fireworks" => LLM::fireworks(model),
            "mistral" => LLM::mistral(model),
            "mock" => LLM::mock(model),
            _ => LLM::custom(provider, model),
        };
        // API keys are resolved from env vars by the LLM constructors
        llm
    }
}

// ---------------------------------------------------------------------------
// Retry Executor
// ---------------------------------------------------------------------------

/// Execute a function with retry and timeout.
pub fn execute_with_retry<F, T, E>(
    retry: &RetryPolicy,
    timeout: Option<Duration>,
    mut f: F,
) -> Result<T, String>
where
    F: FnMut() -> Result<T, E>,
    E: std::fmt::Display,
{
    let deadline = timeout.map(|t| Instant::now() + t);
    let mut last_error = String::new();

    for attempt in 0..retry.max_attempts {
        // Check timeout
        if let Some(dl) = deadline {
            if Instant::now() > dl {
                return Err(format!("timeout after {}ms", timeout.unwrap().as_millis()));
            }
        }

        match f() {
            Ok(result) => return Ok(result),
            Err(e) => {
                last_error = e.to_string();
                if attempt + 1 < retry.max_attempts {
                    let backoff = retry.backoff_for(attempt);
                    if backoff > 0 {
                        std::thread::sleep(Duration::from_millis(backoff));
                    }
                }
            }
        }
    }

    Err(format!(
        "failed after {} attempts: {}",
        retry.max_attempts, last_error
    ))
}

// ---------------------------------------------------------------------------
// Cost Tracker
// ---------------------------------------------------------------------------

/// Track LLM costs per model.
#[derive(Debug, Default)]
pub struct CostTracker {
    usage: std::sync::Mutex<HashMap<String, ModelUsage>>,
}

#[derive(Debug, Default, Clone)]
pub struct ModelUsage {
    pub calls: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_latency_ms: u64,
    pub errors: u64,
}

impl ModelUsage {
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }

    pub fn avg_latency_ms(&self) -> u64 {
        if self.calls > 0 {
            self.total_latency_ms / self.calls
        } else {
            0
        }
    }
}

impl CostTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&self, model: &str, input: u64, output: u64, latency_ms: u64, is_error: bool) {
        let mut usage = self.usage.lock().unwrap();
        let entry = usage.entry(model.to_string()).or_default();
        entry.calls += 1;
        entry.input_tokens += input;
        entry.output_tokens += output;
        entry.total_latency_ms += latency_ms;
        if is_error {
            entry.errors += 1;
        }
    }

    pub fn get(&self, model: &str) -> ModelUsage {
        self.usage
            .lock()
            .unwrap()
            .get(model)
            .cloned()
            .unwrap_or_default()
    }

    pub fn all(&self) -> HashMap<String, ModelUsage> {
        self.usage.lock().unwrap().clone()
    }

    pub fn total_tokens(&self) -> u64 {
        self.usage
            .lock()
            .unwrap()
            .values()
            .map(|u| u.total_tokens())
            .sum()
    }

    pub fn total_cost_estimate(&self, cost_per_1k_tokens: f64) -> f64 {
        self.total_tokens() as f64 / 1000.0 * cost_per_1k_tokens
    }
}

// ---------------------------------------------------------------------------
// Backpressure
// ---------------------------------------------------------------------------

/// Simple concurrency limiter.
#[derive(Debug)]
pub struct ConcurrencyLimit {
    max: usize,
    current: std::sync::atomic::AtomicUsize,
}

impl ConcurrencyLimit {
    pub fn new(max: usize) -> Self {
        Self {
            max,
            current: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    pub fn unlimited() -> Self {
        Self::new(usize::MAX)
    }

    /// Try to acquire a slot. Returns false if at capacity.
    pub fn try_acquire(&self) -> bool {
        let current = self.current.load(std::sync::atomic::Ordering::SeqCst);
        if current >= self.max {
            return false;
        }
        self.current
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        true
    }

    /// Release a slot.
    pub fn release(&self) {
        self.current
            .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }

    pub fn current(&self) -> usize {
        self.current.load(std::sync::atomic::Ordering::SeqCst)
    }

    pub fn available(&self) -> usize {
        self.max.saturating_sub(self.current())
    }
}

// ---------------------------------------------------------------------------
// Scheduler Config
// ---------------------------------------------------------------------------

/// Complete scheduler configuration.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub max_concurrency: usize,
    pub default_retry: RetryPolicy,
    pub default_timeout: Option<Duration>,
    pub model_router: Option<ModelRouter>,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_concurrency: 10,
            default_retry: RetryPolicy::attempts(3),
            default_timeout: Some(Duration::from_secs(30)),
            model_router: None,
        }
    }
}

impl SchedulerConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn max_concurrency(mut self, n: usize) -> Self {
        self.max_concurrency = n;
        self
    }

    pub fn default_retry(mut self, policy: RetryPolicy) -> Self {
        self.default_retry = policy;
        self
    }

    pub fn default_timeout(mut self, t: Duration) -> Self {
        self.default_timeout = Some(t);
        self
    }

    pub fn no_timeout(mut self) -> Self {
        self.default_timeout = None;
        self
    }

    pub fn model_router(mut self, router: ModelRouter) -> Self {
        self.model_router = Some(router);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn priority_ordering() {
        assert!(Priority::Critical < Priority::High);
        assert!(Priority::High < Priority::Normal);
        assert!(Priority::Normal < Priority::Low);
        assert!(Priority::Low < Priority::Background);
    }

    #[test]
    fn priority_from_str() {
        assert_eq!(Priority::from_str("critical"), Priority::Critical);
        assert_eq!(Priority::from_str("HIGH"), Priority::High);
        assert_eq!(Priority::from_str("unknown"), Priority::Normal);
    }

    #[test]
    fn model_router_default() {
        let router = ModelRouter::new("groq", "llama-3.3-70b-versatile");
        let (p, m) = router.resolve("any_step");
        assert_eq!(p, "groq");
        assert_eq!(m, "llama-3.3-70b-versatile");
    }

    #[test]
    fn model_router_matching() {
        let router = ModelRouter::new("groq", "llama-3.3-70b-versatile")
            .route("search", "groq", "llama-3.1-8b-instant")
            .route("analyze", "openai", "gpt-4o");

        let (p1, m1) = router.resolve("search_code");
        assert_eq!(p1, "groq");
        assert_eq!(m1, "llama-3.1-8b-instant");

        let (p2, m2) = router.resolve("analyze_results");
        assert_eq!(p2, "openai");
        assert_eq!(m2, "gpt-4o");

        let (p3, _m3) = router.resolve("write_file");
        assert_eq!(p3, "groq"); // default
    }

    #[test]
    fn retry_succeeds_eventually() {
        let mut attempts = 0;
        let result =
            execute_with_retry(&RetryPolicy::attempts(3), None, || -> Result<&str, &str> {
                attempts += 1;
                if attempts < 3 {
                    Err("not yet")
                } else {
                    Ok("done")
                }
            });
        assert_eq!(result.unwrap(), "done");
        assert_eq!(attempts, 3);
    }

    #[test]
    fn retry_exhausted() {
        let result = execute_with_retry(&RetryPolicy::attempts(2), None, || -> Result<(), &str> {
            Err("always fails")
        });
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("2 attempts"));
    }

    #[test]
    fn retry_with_timeout() {
        let result = execute_with_retry(
            &RetryPolicy::attempts(100),
            Some(Duration::from_millis(1)),
            || -> Result<(), &str> {
                std::thread::sleep(Duration::from_millis(5));
                Err("slow")
            },
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("timeout"));
    }

    #[test]
    fn cost_tracker() {
        let tracker = CostTracker::new();
        tracker.record("gpt-4o", 100, 50, 500, false);
        tracker.record("gpt-4o", 200, 100, 800, false);
        tracker.record("llama3", 50, 25, 200, false);

        let gpt = tracker.get("gpt-4o");
        assert_eq!(gpt.calls, 2);
        assert_eq!(gpt.total_tokens(), 450);
        assert_eq!(gpt.avg_latency_ms(), 650);

        assert_eq!(tracker.total_tokens(), 525);
    }

    #[test]
    fn cost_estimate() {
        let tracker = CostTracker::new();
        tracker.record("gpt-4o", 1000, 500, 100, false);
        // $0.01 per 1K tokens
        let cost = tracker.total_cost_estimate(0.01);
        assert!((cost - 0.015).abs() < 0.001);
    }

    #[test]
    fn concurrency_limit() {
        let limiter = ConcurrencyLimit::new(2);
        assert!(limiter.try_acquire());
        assert!(limiter.try_acquire());
        assert!(!limiter.try_acquire()); // at capacity
        assert_eq!(limiter.current(), 2);
        assert_eq!(limiter.available(), 0);

        limiter.release();
        assert_eq!(limiter.available(), 1);
        assert!(limiter.try_acquire());
    }

    #[test]
    fn concurrency_unlimited() {
        let limiter = ConcurrencyLimit::unlimited();
        for _ in 0..10000 {
            assert!(limiter.try_acquire());
        }
    }

    #[test]
    fn scheduler_config_builder() {
        let config = SchedulerConfig::new()
            .max_concurrency(5)
            .default_retry(RetryPolicy::attempts(3))
            .default_timeout(Duration::from_secs(10))
            .model_router(ModelRouter::new("groq", "llama3").route(
                "search",
                "groq",
                "llama-3.1-8b-instant",
            ));

        assert_eq!(config.max_concurrency, 5);
        assert!(config.model_router.is_some());
    }
}
