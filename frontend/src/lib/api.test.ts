import { afterEach, describe, expect, it, vi } from "vitest";

import { parseStreamData, streamChat } from "./api";

function streamResponse(frames: string): Response {
  const body = new ReadableStream({
    start(controller) {
      controller.enqueue(new TextEncoder().encode(frames));
      controller.close();
    },
  });
  return new Response(body, { status: 200 });
}

function streamHandlers() {
  return {
    onProcessing: vi.fn(),
    onEvent: vi.fn(),
    onFinal: vi.fn(),
    onError: vi.fn(),
  };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("parseStreamData", () => {
  it("실행 시작 프레임을 파싱한다", () => {
    expect(
      parseStreamData('[PROCESSING] {"run_id":"chat-1"}'),
    ).toEqual({
      type: "processing",
      payload: { run_id: "chat-1" },
    });
  });

  it("진행 이벤트 프레임을 파싱한다", () => {
    const frame = parseStreamData(
      '[EVENT] {"run_id":"chat-1","event":"stage","phase":"database","status":"running","message":"DB 확인","data":{},"timestamp":"2026-07-24T00:00:00Z"}',
    );
    expect(frame.type).toBe("event");
    if (frame.type === "event") {
      expect(frame.payload.phase).toBe("database");
      expect(frame.payload.message).toBe("DB 확인");
    }
  });

  it("완료 프레임을 구분한다", () => {
    expect(parseStreamData("[DONE]")).toEqual({ type: "done" });
  });

  it("최종 프레임 없이 스트림이 닫히면 실패로 처리한다", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        streamResponse('data: [PROCESSING] {"run_id":"chat-1"}\n\n'),
      ),
    );

    await expect(
      streamChat(
        { query: "테스트", conversation_id: "conversation-1" },
        streamHandlers(),
      ),
    ).rejects.toThrow("최종 결과를 받기 전에 응답 연결이 종료됐습니다.");
  });

  it("최종 프레임을 받은 뒤 연결이 닫히면 정상 완료한다", async () => {
    const handlers = streamHandlers();
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        streamResponse(
          'data: [FINAL] {"run_id":"chat-1","text":"완료","status":"completed"}\n\n',
        ),
      ),
    );

    await expect(
      streamChat(
        { query: "테스트", conversation_id: "conversation-1" },
        handlers,
      ),
    ).resolves.toBeUndefined();
    expect(handlers.onFinal).toHaveBeenCalledOnce();
  });
});
