//! Real-world end-to-end smoke test.
//! Run: cargo run --example real_world_e2e --release

use std::sync::{Arc, RwLock};
use std::time::Instant;

use uldb::engine::{Engine, EngineConfig};
use ulflow::checkpoint::Checkpoint;
use ulflow::context::ContextValue;
use ulflow::memory::{Memory, MemoryScope};
use ulflow::prelude::*;
use ulflow::step::Input;
use ulmcp::context::ContextTracker;
use ulmcp::registry::Registry;
use ulmcp::tool::*;
use ulmen_core::*;

const SEP: &str = "======================================================================";

fn bench<F: FnMut()>(name: &str, iters: u32, mut f: F) -> f64 {
    for _ in 0..50 {
        f();
    }
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let ns = start.elapsed().as_nanos() as f64 / iters as f64;
    if ns >= 1000.0 {
        println!("    {:<45} {:>10.1} us", name, ns / 1000.0);
    } else {
        println!("    {:<45} {:>10.0} ns", name, ns);
    }
    ns
}

fn main() {
    println!("\n{SEP}");
    println!("  ULMEN ECOSYSTEM: Real-World End-to-End Test");
    println!("{SEP}");

    let mut pass = 0u32;
    let mut fail = 0u32;
    macro_rules! check {
        ($name:expr, $cond:expr) => {
            if $cond {
                pass += 1;
                println!("  [PASS] {}", $name);
            } else {
                fail += 1;
                println!("  [FAIL] {}", $name);
            }
        };
    }

    // === 1. ULDB: Ingest ===
    println!("\n--- 1. ULDB: Ingest Codebase ---");
    let db_dir = std::env::temp_dir().join(format!("ulflow_e2e_{}", std::process::id()));
    std::fs::create_dir_all(&db_dir).unwrap();
    let engine = Arc::new(RwLock::new(
        Engine::open(EngineConfig::new(&db_dir)).unwrap(),
    ));

    let files = vec![
        ("auth/jwt.py::validate_token", "def validate_token(token):\n    key = load_rsa_key()\n    return jwt.decode(token, key)"),
        ("auth/jwt.py::create_token", "def create_token(uid): return jwt.encode({'sub': uid}, load_rsa_key())"),
        ("auth/jwt.py::refresh_token", "def refresh_token(t): return create_token(validate_token(t)['sub'])"),
        ("auth/middleware.py::check_auth", "def check_auth(req): return validate_token(req.headers['Auth'])"),
        ("models/user.py::User", "class User: id: int; email: str; role: str = 'user'"),
        ("api/routes.py::login", "async def login(c): return create_token(authenticate(c).id)"),
        ("tests/test_auth.py::test_valid", "def test_valid(): assert validate_token(create_token(1))['sub']==1"),
    ];
    {
        let mut eng = engine.write().unwrap();
        for (k, v) in &files {
            eng.put(k.as_bytes(), v.as_bytes()).unwrap();
        }
    }
    check!("ingest codebase", true);
    println!("    {} files ingested", files.len());

    // === 2. ULMCP: Register Tools ===
    println!("\n--- 2. ULMCP: Register Tools ---");
    let mut registry = Registry::new();

    let eng_s: Arc<RwLock<Engine>> = Arc::clone(&engine);
    registry.register_tool(
        ToolDef::new("code_search", "Search")
            .param("query", "Q", ParamType::String, true)
            .tag("search"),
        Box::new(move |call| {
            let q = call
                .arguments
                .get("query")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let mut eng = eng_s.write().unwrap();
            let hits = eng.indices.query(&uldb::query::planner::QuerySpec {
                text: q.to_string(),
                top_k: 5,
                ..Default::default()
            });
            let r: Vec<String> = hits
                .iter()
                .map(|h| String::from_utf8_lossy(&h.key).to_string())
                .collect();
            ToolResult {
                call_id: call.call_id.clone(),
                status: ToolStatus::Success,
                output: ToolValue::String(r.join("\n")),
                error: None,
                tokens_used: Some(r.len() * 20),
                latency_ms: None,
            }
        }),
    );

    let eng_r: Arc<RwLock<Engine>> = Arc::clone(&engine);
    registry.register_tool(
        ToolDef::new("file_read", "Read")
            .param("path", "P", ParamType::String, true)
            .tag("io"),
        Box::new(move |call| {
            let p = call
                .arguments
                .get("path")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let eng = eng_r.read().unwrap();
            match eng.get(p.as_bytes()) {
                Some(d) => {
                    let text = String::from_utf8_lossy(&d).to_string();
                    let len = d.len();
                    ToolResult {
                        call_id: call.call_id.clone(),
                        status: ToolStatus::Success,
                        output: ToolValue::String(text),
                        error: None,
                        tokens_used: Some(len / 4),
                        latency_ms: None,
                    }
                }
                None => ToolResult {
                    call_id: call.call_id.clone(),
                    status: ToolStatus::Error,
                    output: ToolValue::Null,
                    error: Some(format!("not found: {}", p)),
                    tokens_used: None,
                    latency_ms: None,
                },
            }
        }),
    );

    let eng_w: Arc<RwLock<Engine>> = Arc::clone(&engine);
    registry.register_tool(
        ToolDef::new("file_write", "Write")
            .param("path", "P", ParamType::String, true)
            .param("content", "C", ParamType::String, true)
            .tag("io"),
        Box::new(move |call| {
            let p = call
                .arguments
                .get("path")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let c = call
                .arguments
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let mut eng = eng_w.write().unwrap();
            match eng.put(p.as_bytes(), c.as_bytes()) {
                Ok(()) => ToolResult {
                    call_id: call.call_id.clone(),
                    status: ToolStatus::Success,
                    output: ToolValue::String(format!("wrote {} bytes", c.len())),
                    error: None,
                    tokens_used: Some(5),
                    latency_ms: None,
                },
                Err(e) => ToolResult {
                    call_id: call.call_id.clone(),
                    status: ToolStatus::Error,
                    output: ToolValue::Null,
                    error: Some(e.to_string()),
                    tokens_used: None,
                    latency_ms: None,
                },
            }
        }),
    );
    check!("register 3 tools", registry.tool_count() == 3);

    // === 3. ULFLOW: Execute Workflow ===
    println!("\n--- 3. ULFLOW: Execute Workflow ---");
    let flow = Flow::pipeline("auth_refactor")
        .context_budget(8192)
        .step(Step::tool("search").tool("code_search")
            .input("query", Input::from_var("task")).build())
        .step(Step::tool("read").tool("file_read")
            .input("path", Input::literal("auth/jwt.py::validate_token"))
            .depends_on("search").build())
        .step(Step::agent("analyze", "Review {{read.code}} for: {{task}}"))
        .step(Step::tool("write").tool("file_write")
            .input("path", Input::literal("auth/jwt.py::validate_token"))
            .input("content", Input::literal("_key=None\ndef validate_token(t):\n    global _key\n    if not _key: _key=load_rsa_key()\n    return jwt.decode(t,_key)"))
            .depends_on("analyze").build())
        .build().unwrap();

    let events: Arc<std::sync::Mutex<Vec<String>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
    let ec = Arc::clone(&events);
    let mut runner = FlowRunner::new(registry);
    runner.on_event(move |e| {
        ec.lock().unwrap().push(e.kind.to_string());
    });

    let result = runner
        .run(flow, FlowInput::new().var("task", "Cache RSA key"))
        .unwrap();
    check!("workflow succeeded", result.succeeded());
    check!("4 steps done", result.steps_completed == 4);
    println!(
        "    Status: {}, Steps: {}, Tokens: {}, Latency: {}ms",
        result.status, result.steps_completed, result.tokens_used, result.latency_ms
    );

    // Verify search found results
    if let Some(ContextValue::String(s)) = result.get("search.output") {
        check!("search found code", s.contains("validate_token"));
    } else {
        check!("search found code", false);
    }

    // Verify code written to uldb
    {
        let eng = engine.read().unwrap();
        if let Some(d) = eng.get(b"auth/jwt.py::validate_token") {
            check!("code updated", String::from_utf8_lossy(&d).contains("_key"));
        } else {
            check!("code updated", false);
        }
    }

    {
        let ev = events.lock().unwrap();
        check!("events emitted", ev.len() >= 4);
    }

    let tel = runner.telemetry_summary();
    check!("telemetry", tel.total_spans >= 4);

    // === 4. ULMEN-CORE: Serialize ===
    println!("\n--- 4. ULMEN-CORE: Serialize ---");
    let records = vec![
        AgentRecord {
            record_type: RecordType::Msg,
            id: "m1".into(),
            thread_id: "s42".into(),
            step: 1,
            fields: vec![
                FieldValue::Str("user".into()),
                FieldValue::Int(1),
                FieldValue::Str("Refactor auth".into()),
                FieldValue::Int(5),
                FieldValue::Bool(false),
            ],
            meta: MetaFields::default(),
        },
        AgentRecord {
            record_type: RecordType::Tool,
            id: "t1".into(),
            thread_id: "s42".into(),
            step: 2,
            fields: vec![
                FieldValue::Str("search".into()),
                FieldValue::Str("{}".into()),
                FieldValue::Str("done".into()),
            ],
            meta: MetaFields::default(),
        },
        AgentRecord {
            record_type: RecordType::Res,
            id: "t1".into(),
            thread_id: "s42".into(),
            step: 3,
            fields: vec![
                FieldValue::Str("search".into()),
                FieldValue::Str("auth.py".into()),
                FieldValue::Str("done".into()),
                FieldValue::Int(12),
            ],
            meta: MetaFields::default(),
        },
    ];
    let payload = AgentPayload {
        header: AgentHeader {
            thread_id: Some("s42".into()),
            record_count: 3,
            ..Default::default()
        },
        records,
    };
    let encoded = payload.encode();
    check!("encode", encoded.starts_with("ULMEN-AGENT v1\n"));
    check!("validate", validate_payload(&payload).is_ok());
    let dec = AgentPayload::decode(&encoded).unwrap();
    check!("decode roundtrip", dec.records.len() == 3);
    let comp = compress_context(
        &dec.records,
        CompressStrategy::CompletedSequences,
        2,
        None,
        None,
        false,
    );
    check!("compress", comp.len() < 3);
    println!(
        "    {} bytes, {} -> {} records",
        encoded.len(),
        3,
        comp.len()
    );

    // === 5. Persistence ===
    println!("\n--- 5. Persistence ---");
    {
        let mut eng = engine.write().unwrap();
        eng.put(b"agent:s42", encoded.as_bytes()).unwrap();
    }
    drop(engine);
    let eng2 = Engine::open(EngineConfig::new(&db_dir)).unwrap();
    if let Some(raw) = eng2.get(b"agent:s42") {
        let r = AgentPayload::decode(std::str::from_utf8(&raw).unwrap()).unwrap();
        check!("survives restart", r.records.len() == 3);
    } else {
        check!("survives restart", false);
    }

    // === 6. Memory + Checkpoint ===
    println!("\n--- 6. Memory + Checkpoint ---");
    let mut mem = Memory::new();
    mem.store("fix", "cached key", MemoryScope::Session, 0.95);
    check!(
        "memory",
        mem.get_value("fix").and_then(|v| v.as_str()) == Some("cached key")
    );
    let mut cp = Checkpoint::new("r1", "auth_refactor");
    cp.mark_completed("search", ulflow::step::StepStatus::Succeeded);
    cp.tokens_used = 200;
    let cp2 = Checkpoint::from_bytes(&cp.to_bytes()).unwrap();
    check!(
        "checkpoint",
        cp2.is_step_completed("search") && cp2.tokens_used == 200
    );

    // === 7. Benchmarks ===
    println!("\n--- 7. Benchmarks ---");
    bench("ulmen-core encode 3 records", 10_000, || {
        let _ = payload.encode();
    });
    bench("ulmen-core decode 3 records", 10_000, || {
        let _ = AgentPayload::decode(&encoded);
    });
    bench("ulmen-core validate", 10_000, || {
        let _ = validate_payload(&payload);
    });
    bench("context tracker cycle", 100_000, || {
        let mut c = ContextTracker::new(4096);
        c.use_tokens(100);
        c.reserve(200);
        let _ = c.available();
    });
    bench("memory store+get", 100_000, || {
        let mut m = Memory::new();
        m.store("k", "v", MemoryScope::Session, 0.9);
        let _ = m.get_value("k");
    });
    bench("checkpoint save+load", 10_000, || {
        let b = cp.to_bytes();
        let _ = Checkpoint::from_bytes(&b);
    });

    // Results
    println!("\n  Passed: {pass}  Failed: {fail}");
    if fail == 0 {
        println!("\n{SEP}");
        println!("  ALL {pass} TESTS PASSED");
        println!("{SEP}");
    } else {
        println!("\n{SEP}");
        println!("  {fail} TESTS FAILED");
        println!("{SEP}");
        std::process::exit(1);
    }
    let _ = std::fs::remove_dir_all(&db_dir);
}
