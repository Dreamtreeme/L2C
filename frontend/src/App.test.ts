import { describe, expect, it } from "vitest";

import { workspaceReducer, type WorkspaceState } from "./App";
import type { ChatMessage, ClarificationQuestion } from "./types";

const clarification: ClarificationQuestion = {
  question_id: "occupation",
  field: "occupation",
  question: "어떤 직무를 찾을까요?",
  options: [],
  allow_custom: true,
};

const assistant: ChatMessage = {
  id: "assistant-1",
  role: "assistant",
  text: "",
  createdAt: "2026-08-18T00:00:00Z",
  state: "pending",
  events: [],
};

function state(messages: ChatMessage[] = []): WorkspaceState {
  return {
    messages,
    isRunning: false,
    activeRunId: null,
    activePhase: null,
    pendingResume: {
      runId: "run-old",
      investigationId: "investigation-old",
      clarification,
      status: "waiting_input",
    },
    selectedMessageId: null,
    selectedJobId: 999,
    cancelRequested: false,
    cancelError: null,
  };
}

describe("workspaceReducer", () => {
  it("보완 답변을 시작할 때 이전 대기 상태를 해제한다", () => {
    const next = workspaceReducer(state(), {
      type: "start_turn",
      user: {
        id: "user-1",
        role: "user",
        text: "AI 엔지니어",
        createdAt: "2026-08-18T00:00:00Z",
        state: "complete",
      },
      assistant,
    });

    expect(next.pendingResume).toBeNull();
    expect(next.selectedMessageId).toBe("assistant-1");
    expect(next.selectedJobId).toBeNull();
  });

  it("완료 답변의 구조화된 근거와 실행 정보를 해당 메시지에 저장한다", () => {
    const next = workspaceReducer(state([assistant]), {
      type: "final",
      assistantId: "assistant-1",
      payload: {
        run_id: "run-1",
        text: "완료 [job_id:999]",
        status: "completed",
        grounded_answer: {
          lines: [],
          citations: [
            {
              citation_id: 1,
              document_id: 101,
              field: "raw_text",
              evidence_text: "근거",
            },
          ],
        },
        metrics: { run_id: "run-1", duration_sec: 3.2 },
      },
    });

    expect(next.selectedJobId).toBe(101);
    expect(next.messages[0]).toMatchObject({
      state: "complete",
      citationIds: [101],
      metrics: { run_id: "run-1", duration_sec: 3.2 },
    });
  });
});
