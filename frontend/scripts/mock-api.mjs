import { createServer } from "node:http";

const port = Number(process.env.L2C_MOCK_API_PORT || 8000);
const runs = [];

function json(response, status, payload) {
  response.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
  });
  response.end(JSON.stringify(payload));
}

async function readJson(request) {
  let body = "";
  for await (const chunk of request) {
    body += chunk;
  }
  return body ? JSON.parse(body) : {};
}

function runEvent(runId, phase, message, status = "running", data = {}) {
  return {
    run_id: runId,
    event: "stage",
    phase,
    status,
    message,
    data,
    timestamp: new Date().toISOString(),
  };
}

function writeFrame(response, prefix, payload = null) {
  const suffix = payload === null ? "" : ` ${JSON.stringify(payload)}`;
  response.write(`data: ${prefix}${suffix}\n\n`);
}

const server = createServer(async (request, response) => {
  const url = new URL(request.url || "/", `http://${request.headers.host}`);

  if (request.method === "GET" && url.pathname === "/api/operations") {
    json(response, 200, {
      runs,
      retention: {
        inventory: {
          job_postings: 34,
          job_versions: 51,
          recipe_candidates: 8,
          active_recipes: 5,
        },
        files: {
          log_count: 2,
          artifact_count: 3,
          reclaimable_bytes: 2841000,
        },
        database: {
          run_events: 4,
          expired_snapshots: 1,
        },
      },
    });
    return;
  }

  if (
    request.method === "POST" &&
    url.pathname === "/api/operations/retention"
  ) {
    json(response, 200, { deleted_files: 5, deleted_rows: 5 });
    return;
  }

  const jobMatch = url.pathname.match(/^\/api\/jobs\/(\d+)$/);
  if (request.method === "GET" && jobMatch) {
    const jobId = Number(jobMatch[1]);
    const jobs = {
      101: {
        id: 101,
        company_name: "보이저엑스",
        position: "[Vrew/vFlat] iOS 개발자",
        url: "https://www.wanted.co.kr/wd/101",
        posted_at_text: "상시 채용",
        evidence_hash: "da4ad6d26d4935c3",
        collected_at: "2026-07-24T03:12:00Z",
        source_platform: "Wanted",
        raw_text:
          "주요업무\n- iOS 애플리케이션 개발 및 운영\n- 영상 편집 기능 고도화\n\n자격요건\n- Swift 기반 iOS 개발 경험\n- 제품 품질과 사용자 경험에 대한 관심\n\n우대사항\n- SwiftUI, AVFoundation 사용 경험",
      },
      102: {
        id: 102,
        company_name: "넛지헬스케어",
        position: "iOS 개발자",
        url: "https://www.wanted.co.kr/wd/102",
        posted_at_text: "2026.07.22",
        evidence_hash: "52936af7bfc4352a",
        collected_at: "2026-07-24T03:13:00Z",
        source_platform: "Wanted",
        raw_text:
          "주요업무\n- 캐시워크 iOS 앱 기능 개발\n- 서비스 안정성과 성능 개선\n\n자격요건\n- Swift 실무 경험\n- 비동기 프로그래밍 이해\n\n우대사항\n- 대규모 사용자 서비스 운영 경험",
      },
    };
    const job = jobs[jobId];
    if (!job) {
      json(response, 404, { detail: "Job not found" });
      return;
    }
    json(response, 200, job);
    return;
  }

  if (
    request.method === "POST" &&
    /^\/api\/runs\/[^/]+\/cancel$/.test(url.pathname)
  ) {
    json(response, 200, { cancel_requested: true, status: "running" });
    return;
  }

  if (request.method === "POST" && url.pathname === "/api/chat") {
    const payload = await readJson(request);
    const runId = `mock-${Date.now()}`;
    const now = new Date().toISOString();
    runs.unshift({
      run_id: runId,
      user_query: String(payload.query || ""),
      query: String(payload.query || ""),
      status: "running",
      phase: "received",
      message: "요청을 접수했습니다.",
      created_at: now,
      updated_at: now,
    });

    response.writeHead(200, {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    });
    writeFrame(response, "[PROCESSING]", { run_id: runId });

    if (String(payload.query || "").includes("채용공고 찾아줘")) {
      const clarificationEvent = runEvent(
        runId,
        "clarification",
        "조사할 업무 영역을 먼저 확인해야 합니다.",
        "waiting_input",
      );
      setTimeout(() => {
        writeFrame(response, "[EVENT]", clarificationEvent);
        writeFrame(response, "[FINAL]", {
          run_id: runId,
          text: "어떤 업무 영역의 공고를 찾을지 선택해 주세요.",
          status: "waiting_input",
          clarification: {
            question_id: "occupation-domain",
            field: "occupation_domain",
            question: "어떤 업무 영역의 채용공고를 찾을까요?",
            reason: "검색 범위가 넓어 업무 영역을 확정해야 합니다.",
            allow_custom: true,
            options: [
              {
                option_id: "office",
                label: "사무·기획",
                value: "office",
                description: "경영, 기획, 마케팅, 재무 직무",
                matching_count: 18,
              },
              {
                option_id: "software",
                label: "IT·소프트웨어",
                value: "software",
                description: "개발, 데이터, AI, 보안 직무",
                matching_count: 34,
              },
              {
                option_id: "manufacturing",
                label: "제조·생산",
                value: "manufacturing",
                description: "생산, 품질, 설비, 공정 직무",
                matching_count: 12,
              },
            ],
          },
          investigation_id: "mock-investigation-clarification",
          conversation_id: payload.conversation_id || "",
          metrics: {
            run_id: runId,
            duration_sec: 1.24,
            steps: [
              {
                phase: "clarification",
                component: "commander",
                duration_sec: 1.08,
                action_source: "commander",
                success: true,
              },
            ],
            llm: {
              totals: {
                input_tokens: 420,
                output_tokens: 88,
                total_tokens: 508,
              },
              calls: [{ model: "mock-commander" }],
              cost: {
                currency: "USD",
                estimated_total: 0.00031,
                unpriced_models: [],
              },
            },
            outcome: {
              status: "waiting_input",
              last_phase: "clarification",
            },
          },
        });
        writeFrame(response, "[DONE]");
        response.end();
        const run = runs.find((item) => item.run_id === runId);
        if (run) {
          run.status = "waiting_input";
          run.phase = "clarification";
          run.message = "사용자 선택을 기다리고 있습니다.";
          run.updated_at = new Date().toISOString();
        }
      }, 450);
      return;
    }

    const events = [
      runEvent(runId, "planning", "질문의 조사 조건을 확인하고 있습니다."),
      runEvent(runId, "database", "로컬 DB에서 기존 근거를 확인하고 있습니다."),
      runEvent(runId, "collection", "원티드에서 공고 2개를 수집했습니다."),
      runEvent(runId, "review", "수집한 공고의 직무와 출처를 검증하고 있습니다."),
      runEvent(runId, "answering", "DB 근거를 이용해 답변을 작성하고 있습니다."),
    ];

    events.forEach((event, index) => {
      setTimeout(() => {
        writeFrame(response, "[EVENT]", event);
      }, 250 + index * 360);
    });

    setTimeout(
      () => {
        const metrics = {
          run_id: runId,
          duration_sec: 8.74,
          steps: events.map((event, index) => ({
            phase: event.phase,
            component: event.phase,
            duration_sec: 0.6 + index * 0.41,
            action_source:
              event.phase === "collection"
                ? "job_card_queue"
                : event.phase === "review"
                  ? "reflex"
                  : "commander",
            success: true,
          })),
          llm: {
            totals: {
              input_tokens: 3490,
              output_tokens: 672,
              total_tokens: 4162,
            },
            calls: [{ model: "mock-commander" }, { model: "mock-worker" }],
            cost: {
              currency: "USD",
              estimated_total: 0.00382,
              unpriced_models: [],
            },
          },
          outcome: {
            status: "completed",
            last_phase: "completed",
          },
        };
        writeFrame(response, "[FINAL]", {
          run_id: runId,
          text:
            "확인된 iOS 개발자 공고는 2건입니다.\n\n## 비교 결과\n\n- **보이저엑스**는 Swift 기반 제품 개발과 영상 처리 경험을 중요하게 봅니다. SwiftUI와 AVFoundation 경험이 있으면 유리합니다. [job_id:101]\n- **넛지헬스케어**는 대규모 소비자 앱의 안정성과 비동기 프로그래밍 경험을 강조합니다. [job_id:102]\n\n두 공고 모두 iOS 실무 경험을 요구하지만, 제품 기능 개발은 보이저엑스, 서비스 운영 경험은 넛지헬스케어 쪽에 더 가깝습니다.",
          status: "completed",
          clarification: null,
          investigation_id: "mock-investigation",
          conversation_id: payload.conversation_id || "",
          metrics,
        });
        writeFrame(response, "[DONE]");
        response.end();
        const run = runs.find((item) => item.run_id === runId);
        if (run) {
          run.status = "completed";
          run.phase = "completed";
          run.message = "조사가 완료됐습니다.";
          run.updated_at = new Date().toISOString();
        }
      },
      250 + events.length * 360,
    );
    return;
  }

  json(response, 404, { detail: "Not found" });
});

server.listen(port, "127.0.0.1", () => {
  console.log(`L2C mock API: http://127.0.0.1:${port}`);
});
