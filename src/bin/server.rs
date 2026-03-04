//! qwen-asr-server — Qwen3-ASR HTTP inference server (Rust port)
//!
//! Endpoints:
//!   GET  /          — built-in HTML status page
//!   POST /inference — transcribe uploaded audio, returns JSON or plain text
//!   POST /load      — hot-swap the loaded model directory at runtime
//!   GET  /health    — readiness probe
//!
//! Requests are serialized: only one inference runs at a time.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use axum::{
    Router,
    extract::{Multipart, Request, State},
    http::{HeaderValue, StatusCode, header},
    middleware::{self, Next},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
};
use clap::Parser;
use rayon::ThreadPoolBuilder;
use serde_json::json;

use qwen_asr_rs::transcribe::Pipeline;

// ── CLI ───────────────────────────────────────────────────────────────────────

/// Qwen3-ASR HTTP inference server (Rust port)
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Model directory (*.safetensors + vocab.json)
    #[arg(short = 'd', long)]
    model_dir: PathBuf,

    /// Hostname / IP to bind
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Port number
    #[arg(long, default_value_t = 8080)]
    port: u16,

    /// Number of threads for inference (0 = auto)
    #[arg(short = 't', long, default_value_t = 0)]
    threads: usize,

    /// Default forced language (empty = auto-detect)
    #[arg(long, default_value = "")]
    language: String,

    /// Directory for static files (serves index.html at GET /)
    #[arg(long)]
    public: Option<PathBuf>,
}

// ── Shared state ──────────────────────────────────────────────────────────────

struct ServerState {
    pipeline: Option<Pipeline>,
    model_dir: PathBuf,
    /// Server-level default language (empty = auto-detect).
    language: String,
    /// Contents of public/index.html, if --public was given and the file exists.
    index_html: Option<String>,
}

type Shared = Arc<Mutex<ServerState>>;

// ── CORS middleware ───────────────────────────────────────────────────────────

async fn cors_middleware(req: Request, next: Next) -> Response {
    let mut res = next.run(req).await;
    let h = res.headers_mut();
    h.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN,  HeaderValue::from_static("*"));
    h.insert(header::ACCESS_CONTROL_ALLOW_HEADERS, HeaderValue::from_static("content-type, authorization"));
    res
}

// ── Response helpers ──────────────────────────────────────────────────────────

fn json_err(status: StatusCode, msg: &str) -> Response {
    (status, [(header::CONTENT_TYPE, "application/json")], json!({"error": msg}).to_string())
        .into_response()
}

fn json_ok(v: serde_json::Value) -> Response {
    (StatusCode::OK, [(header::CONTENT_TYPE, "application/json")], v.to_string())
        .into_response()
}

// ── Handlers ──────────────────────────────────────────────────────────────────

/// Embedded at compile time from `public/index.html`.
/// Served by default; `--public` overrides it at runtime.
const PUBLIC_INDEX_HTML: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/public/index.html"));

/// Minimal built-in page, used only if `public/index.html` is removed from the project.
#[allow(dead_code)]
const BUILTIN_HTML: &str = r#"<!DOCTYPE html>
<html>
<head>
  <title>Qwen3-ASR Server</title>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width">
  <style>
    body { font-family: sans-serif }
    pre  { background: #f4f4f4; padding: 1em }
    form label { display: block; margin: .6em 0 }
    button { margin-top: .8em }
  </style>
</head>
<body>
<h1>Qwen3-ASR Server (Rust)</h1>
<h2>POST /inference</h2>
<pre>curl 127.0.0.1:8080/inference \
  -F file="@audio.wav" \
  -F response_format="json"</pre>
<h2>POST /load</h2>
<pre>curl 127.0.0.1:8080/load \
  -F model="/path/to/model_dir"</pre>
<h2>Try it</h2>
<form action="/inference" method="POST" enctype="multipart/form-data">
  <label>Audio file: <input type="file" name="file" accept="audio/wav" required></label>
  <label>Language (optional): <input type="text" name="language" placeholder="e.g. English"></label>
  <label>Response format:
    <select name="response_format">
      <option value="json">JSON</option>
      <option value="text">Text</option>
    </select>
  </label>
  <button type="submit">Transcribe</button>
</form>
</body>
</html>"#;

async fn get_root(State(shared): State<Shared>) -> Response {
    let st = shared.lock().unwrap();
    // Priority: --public runtime file → compiled-in public/index.html → BUILTIN_HTML
    let html: &str = st.index_html.as_deref().unwrap_or(PUBLIC_INDEX_HTML);
    Html(html.to_owned()).into_response()
}

async fn get_health(State(shared): State<Shared>) -> Response {
    if shared.lock().unwrap().pipeline.is_some() {
        json_ok(json!({"status": "ok"}))
    } else {
        (StatusCode::SERVICE_UNAVAILABLE,
         [(header::CONTENT_TYPE, "application/json")],
         json!({"status": "loading model"}).to_string())
            .into_response()
    }
}

async fn options_inference() -> StatusCode {
    StatusCode::NO_CONTENT
}

async fn post_inference(
    State(shared): State<Shared>,
    mut multipart: Multipart,
) -> Response {
    // ── Collect multipart fields ───────────────────────────────────────────
    let mut audio_bytes: Option<Vec<u8>> = None;
    let mut filename       = String::new();
    let mut req_language   = String::new();
    let mut response_format = String::from("json");

    loop {
        match multipart.next_field().await {
            Ok(Some(field)) => match field.name() {
                Some("file") => {
                    filename = field.file_name().unwrap_or("unknown").to_string();
                    match field.bytes().await {
                        Ok(b)  => audio_bytes = Some(b.to_vec()),
                        Err(e) => return json_err(StatusCode::BAD_REQUEST,
                                                  &format!("read error: {e}")),
                    }
                }
                Some("language")        => { req_language    = field.text().await.unwrap_or_default(); }
                Some("response_format") => { response_format = field.text().await.unwrap_or_default(); }
                _ => { let _ = field.bytes().await; }
            },
            Ok(None)   => break,
            Err(e)     => return json_err(StatusCode::BAD_REQUEST,
                                          &format!("multipart error: {e}")),
        }
    }

    let audio_bytes = match audio_bytes {
        Some(b) => b,
        None    => return json_err(StatusCode::BAD_REQUEST, "no 'file' field in the request"),
    };

    eprintln!("Received: {} ({} bytes)", filename, audio_bytes.len());

    // ── Run inference (serialized via Mutex) ──────────────────────────────
    let mut st = shared.lock().unwrap();

    // Per-request language falls back to server default when not supplied.
    let _effective_language = if req_language.is_empty() { st.language.clone() } else { req_language };

    let pipeline = match st.pipeline.as_mut() {
        Some(p) => p,
        None    => return json_err(StatusCode::SERVICE_UNAVAILABLE, "model not loaded"),
    };

    let (mel, audio_ms) = match pipeline.mel_from_bytes(&audio_bytes) {
        Ok(r)  => r,
        Err(e) => return json_err(StatusCode::BAD_REQUEST, &format!("audio error: {e}")),
    };

    let (text, timing) = match pipeline.transcribe_mel(&mel, audio_ms) {
        Ok(r)  => r,
        Err(e) => return json_err(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
    };

    drop(st); // release lock before building response

    if response_format == "text" {
        return (StatusCode::OK,
                [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
                text)
            .into_response();
    }

    let total_ms  = timing.encode_ms + timing.decode_ms;
    let tok_s     = if timing.decode_ms > 0.0 {
        timing.n_tokens as f64 / (timing.decode_ms / 1000.0)
    } else { 0.0 };
    let rt_factor = if timing.audio_ms > 0.0 { total_ms / timing.audio_ms } else { 0.0 };

    json_ok(json!({
        "text":      text,
        "total_ms":  total_ms,
        "encode_ms": timing.encode_ms,
        "decode_ms": timing.decode_ms,
        "tokens":    timing.n_tokens,
        "tok_s":     tok_s,
        "rt_factor": rt_factor,
    }))
}

async fn post_load(
    State(shared): State<Shared>,
    mut multipart: Multipart,
) -> Response {
    // ── Collect model path ─────────────────────────────────────────────────
    let mut new_model = String::new();

    loop {
        match multipart.next_field().await {
            Ok(Some(field)) => {
                if field.name() == Some("model") {
                    new_model = field.text().await.unwrap_or_default();
                } else {
                    let _ = field.bytes().await;
                }
            }
            Ok(None)   => break,
            Err(e)     => return json_err(StatusCode::BAD_REQUEST,
                                          &format!("multipart error: {e}")),
        }
    }

    if new_model.is_empty() {
        return json_err(StatusCode::BAD_REQUEST, "no 'model' field in the request");
    }

    let path = PathBuf::from(&new_model);
    eprintln!("Loading new model from {} ...", path.display());

    let mut st = shared.lock().unwrap();

    match Pipeline::load(&path) {
        Ok(p) => {
            st.pipeline  = Some(p);
            st.model_dir = path;
            (StatusCode::OK, "Load successful!").into_response()
        }
        Err(e) => {
            eprintln!("error: failed to load model from '{}': {e}", path.display());
            json_err(StatusCode::INTERNAL_SERVER_ERROR, &format!("failed to load model: {e}"))
        }
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    let args = Args::parse();

    if args.threads > 0 {
        ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .unwrap_or_else(|e| eprintln!("warning: could not set thread count: {e}"));
    }

    eprintln!("Loading model from {} ...", args.model_dir.display());
    let pipeline = Pipeline::load(&args.model_dir)
        .unwrap_or_else(|e| { eprintln!("error: {e}"); std::process::exit(1) });

    let index_html = args.public.as_deref().map(|p| p.join("index.html")).and_then(|p| {
        match std::fs::read_to_string(&p) {
            Ok(s)  => { eprintln!("Serving {} at GET /", p.display()); Some(s) }
            Err(e) => { eprintln!("warning: could not read {}: {e}", p.display()); None }
        }
    });

    let shared: Shared = Arc::new(Mutex::new(ServerState {
        pipeline: Some(pipeline),
        model_dir: args.model_dir,
        language: args.language,
        index_html,
    }));

    let app = Router::new()
        .route("/",          get(get_root))
        .route("/health",    get(get_health))
        .route("/inference", post(post_inference).options(options_inference))
        .route("/load",      post(post_load))
        .layer(middleware::from_fn(cors_middleware))
        .with_state(shared);

    let addr = format!("{}:{}", args.host, args.port);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .unwrap_or_else(|e| {
            eprintln!("error: couldn't bind to {addr}: {e}");
            std::process::exit(1);
        });

    eprintln!("\nqwen-asr server listening at http://{addr}\n");

    axum::serve(listener, app)
        .await
        .unwrap_or_else(|e| eprintln!("server error: {e}"));
}
