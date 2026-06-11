//! Live LLM test with Groq.
//! Run: GROQ_API_KEY=your_key cargo run --example live_llm_test --release

use std::sync::{Arc, RwLock};
use std::time::Instant;

use ulmen_core::*;
use ulmcp::tool::*;
use ulmcp::registry::Registry;
use ulflow::prelude::*;
use ulflow::step::Input;
use ulflow::context::ContextValue;
use ulflow::llm::LLM;
use uldb::engine::{Engine, EngineConfig};

const SEP: &str = "======================================================================";

fn main() {
    println!("\n{SEP}");
    println!("  LIVE LLM TEST (Groq)");
    println!("{SEP}");

    // Check API key
    let api_key = std::env::var("GROQ_API_KEY").expect("Set GROQ_API_KEY");
    println!("  API key: {}...{}", &api_key[..8], &api_key[api_key.len()-4..]);

    let mut pass = 0u32;
    let mut fail = 0u32;
    macro_rules! check {
        ($name:expr, $cond:expr) => {
            if $cond { pass += 1; println!("  [PASS] {}", $name); }
            else     { fail += 1; println!("  [FAIL] {}", $name); }
        };
    }

    // === 1. Direct LLM call ===
    println!("\n--- 1. Direct LLM Call ---");

    let llm = LLM::custom("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile")
        .api_key(&api_key);

    println!("  Provider: {}", llm.provider());
    println!("  Model:    {}", llm.model());

    let start = Instant::now();
    let response = llm.ask("What is 2+2? Reply with just the number.").unwrap();
    let latency = start.elapsed().as_millis();

    check!("LLM responds", !response.content.is_empty());
    check!("correct answer", response.content.contains("4"));
    println!("  Response:     {:?}", response.content.trim());
    println!("  Model used:   {}", response.model);
    println!("  Input tokens:  {}", response.input_tokens);
    println!("  Output tokens: {}", response.output_tokens);
    println!("  Latency:       {} ms", latency);
    println!("  Finish:        {}", response.finish_reason);

    // === 2. System prompt + user message ===
    println!("\n--- 2. System Prompt ---");

    let start = Instant::now();
    let response = llm.ask_with_system(
        "You are a senior Rust developer. Be concise. No markdown.",
        "What is the difference between Arc and Rc in Rust? One sentence."
    ).unwrap();
    let latency = start.elapsed().as_millis();

    check!("system prompt works", !response.content.is_empty());
    check!("mentions thread", response.content.to_lowercase().contains("thread")
        || response.content.to_lowercase().contains("concurrent")
        || response.content.to_lowercase().contains("arc"));
    println!("  Response: {}", response.content.trim());
    println!("  Tokens:   {} in + {} out = {} total",
        response.input_tokens, response.output_tokens,
        response.input_tokens + response.output_tokens);
    println!("  Latency:  {} ms", latency);

    // === 3. Multi-turn conversation ===
    println!("\n--- 3. Multi-turn Conversation ---");

    use ulflow::llm::{Message, Role};

    let messages = vec![
        Message { role: Role::User, content: "Remember this number: 42".into() },
        Message { role: Role::Assistant, content: "Got it, I'll remember 42.".into() },
        Message { role: Role::User, content: "What number did I ask you to remember? Just the number.".into() },
    ];

    let start = Instant::now();
    let response = llm.complete(messages).unwrap();
    let latency = start.elapsed().as_millis();

    check!("multi-turn works", !response.content.is_empty());
    check!("remembers number", response.content.contains("42"));
    println!("  Response: {:?}", response.content.trim());
    println!("  Latency:  {} ms", latency);

    // === 4. Code analysis with uldb context ===
    println!("\n--- 4. Code Analysis with uldb ---");

    let db_dir = std::env::temp_dir().join(format!("llm_test_{}", std::process::id()));
    std::fs::create_dir_all(&db_dir).unwrap();
    let engine = Arc::new(RwLock::new(Engine::open(EngineConfig::new(&db_dir)).unwrap()));

    // Ingest code
    {
        let mut eng = engine.write().unwrap();
        eng.put(b"auth/jwt.py::validate_token",
            b"def validate_token(token):\n    key = load_rsa_key()  # Loads from disk every call!\n    return jwt.decode(token, key, algorithms=['RS256'])").unwrap();
        eng.put(b"auth/jwt.py::create_token",
            b"def create_token(user_id, expiry=3600):\n    return jwt.encode({'sub': user_id, 'exp': time.time() + expiry}, load_rsa_key())").unwrap();
    }

    // Register search tool
    let eng_s: Arc<RwLock<Engine>> = Arc::clone(&engine);
    let mut registry = Registry::new();
    registry.register_tool(
        ToolDef::new("code_search", "Search code").param("query", "Q", ParamType::String, true),
        Box::new(move |call| {
            let q = call.arguments.get("query").and_then(|v| v.as_str()).unwrap_or("");
            let mut eng = eng_s.write().unwrap();
            let hits = eng.indices.query(&uldb::query::planner::QuerySpec {
                text: q.to_string(), top_k: 3, ..Default::default() });
            let results: Vec<String> = hits.iter()
                .map(|h| {
                    let key = String::from_utf8_lossy(&h.key).to_string();
                    let val = eng.get(&h.key).unwrap_or_default();
                    format!("{}:\n{}", key, String::from_utf8_lossy(&val))
                }).collect();
            ToolResult { call_id: call.call_id.clone(), status: ToolStatus::Success,
                output: ToolValue::String(results.join("\n\n")),
                error: None, tokens_used: Some(results.len() * 50), latency_ms: None }
        }),
    );

    // Build workflow: search -> LLM analyzes
    let flow = Flow::pipeline("code_review")
        .context_budget(4096)
        .step(Step::tool("search").tool("code_search")
            .input("query", Input::from_var("question"))
            .build())
        .step(Step::agent("analyze",
            "You are a senior engineer reviewing Python code.\n\nCode found:\n{{search.output}}\n\nQuestion: {{question}}\n\nProvide a concise analysis (2-3 sentences max). No markdown."))
        .build().unwrap();

    let start = Instant::now();
    let mut runner = FlowRunner::new(registry).with_llm(
        LLM::custom("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile")
            .api_key(&api_key)
    );

    let result = runner.run(flow, FlowInput::new()
        .var("question", "Is there a performance issue in the JWT validation code?")
    ).unwrap();
    let total_latency = start.elapsed().as_millis();

    check!("workflow succeeded", result.succeeded());
    check!("2 steps done", result.steps_completed == 2);

    if let Some(ContextValue::String(analysis)) = result.get("analyze.output") {
        check!("LLM analyzed code", !analysis.is_empty());
        check!("found the issue", analysis.to_lowercase().contains("key")
            || analysis.to_lowercase().contains("load")
            || analysis.to_lowercase().contains("disk")
            || analysis.to_lowercase().contains("performance")
            || analysis.to_lowercase().contains("cache"));
        println!("\n  LLM Analysis:");
        for line in analysis.trim().lines() {
            println!("    {}", line);
        }
    }
    println!("\n  Workflow tokens: {}", result.tokens_used);
    println!("  Workflow latency: {} ms", total_latency);

    // === 5. Serialize agent conversation ===
    println!("\n--- 5. Persist Agent Conversation ---");

    let search_output: String = result.get("search.output")
        .and_then(|v| if let ContextValue::String(s) = v { Some(s.clone()) } else { None })
        .unwrap_or_default();
    let analysis_output: String = result.get("analyze.output")
        .and_then(|v| if let ContextValue::String(s) = v { Some(s.clone()) } else { None })
        .unwrap_or_default();

    let agent_records = vec![
        AgentRecord {
            record_type: RecordType::Msg, id: "m1".into(), thread_id: "review_001".into(), step: 1,
            fields: vec![FieldValue::Str("user".into()), FieldValue::Int(1),
                FieldValue::Str("Is there a perf issue in JWT validation?".into()),
                FieldValue::Int(10), FieldValue::Bool(false)],
            meta: MetaFields::default(),
        },
        AgentRecord {
            record_type: RecordType::Tool, id: "t1".into(), thread_id: "review_001".into(), step: 2,
            fields: vec![FieldValue::Str("code_search".into()),
                FieldValue::Str("{\"query\":\"JWT validation\"}".into()),
                FieldValue::Str("done".into())],
            meta: MetaFields::default(),
        },
        AgentRecord {
            record_type: RecordType::Res, id: "t1".into(), thread_id: "review_001".into(), step: 3,
            fields: vec![FieldValue::Str("code_search".into()),
                FieldValue::Str(search_output[..search_output.len().min(200)].to_string()),
                FieldValue::Str("done".into()), FieldValue::Int(total_latency as i64)],
            meta: MetaFields::default(),
        },
        AgentRecord {
            record_type: RecordType::Msg, id: "m2".into(), thread_id: "review_001".into(), step: 4,
            fields: vec![FieldValue::Str("assistant".into()), FieldValue::Int(2),
                FieldValue::Str(analysis_output[..analysis_output.len().min(500)].to_string()),
                FieldValue::Int(result.tokens_used as i64), FieldValue::Bool(false)],
            meta: MetaFields::default(),
        },
    ];

    let payload = AgentPayload {
        header: AgentHeader {
            thread_id: Some("review_001".into()),
            record_count: agent_records.len(),
            ..Default::default()
        },
        records: agent_records,
    };

    let encoded = payload.encode();
    check!("encode conversation", encoded.starts_with("ULMEN-AGENT v1\n"));
    check!("validate conversation", validate_payload(&payload).is_ok());

    // Store in uldb
    {
        let mut eng = engine.write().unwrap();
        eng.put(b"agent:review_001", encoded.as_bytes()).unwrap();
    }

    // Retrieve and verify
    {
        let eng = engine.read().unwrap();
        if let Some(raw) = eng.get(b"agent:review_001") {
            let restored = AgentPayload::decode(std::str::from_utf8(&raw).unwrap()).unwrap();
            check!("conversation persisted", restored.records.len() == 4);
        }
    }

    let json_equiv = serde_json::to_string(&serde_json::json!({
        "messages": [
            {"role": "user", "content": "Is there a perf issue?"},
            {"role": "tool", "name": "code_search", "result": "..."},
            {"role": "assistant", "content": "..."}
        ]
    })).unwrap();

    println!("  ULMEN payload: {} bytes", encoded.len());
    println!("  JSON equiv:    ~{} bytes", json_equiv.len());
    println!("  Savings:       {}%", ((1.0 - encoded.len() as f64 / (json_equiv.len() as f64 * 2.0)) * 100.0) as i32);

    // === 6. Benchmarks ===
    println!("\n--- 6. LLM Benchmarks ---");

    // Multiple calls to measure consistency
    let mut latencies = Vec::new();
    let mut total_tokens = 0usize;

    for i in 0..3 {
        let start = Instant::now();
        let resp = llm.ask(&format!("What is {}+{}? Just the number.", i+1, i+2)).unwrap();
        let ms = start.elapsed().as_millis() as u64;
        latencies.push(ms);
        total_tokens += resp.input_tokens + resp.output_tokens;
        println!("    Call {}: {} ms, {} tokens, response: {:?}",
            i+1, ms, resp.input_tokens + resp.output_tokens, resp.content.trim());
    }

    let avg_latency = latencies.iter().sum::<u64>() / latencies.len() as u64;
    let min_latency = *latencies.iter().min().unwrap();
    let max_latency = *latencies.iter().max().unwrap();

    println!("\n  Latency: avg={} ms, min={} ms, max={} ms", avg_latency, min_latency, max_latency);
    println!("  Total tokens across {} calls: {}", latencies.len(), total_tokens);
    println!("  Tokens/sec: {:.0}", total_tokens as f64 / (latencies.iter().sum::<u64>() as f64 / 1000.0));

    // === Results ===
    println!("\n  Passed: {}  Failed: {}", pass, fail);
    if fail == 0 {
        println!("\n{SEP}");
        println!("  ALL {} LIVE TESTS PASSED", pass);
        println!("{SEP}");
    } else {
        println!("\n{SEP}");
        println!("  {} TESTS FAILED", fail);
        println!("{SEP}");
    }

    let _ = std::fs::remove_dir_all(&db_dir);
}
