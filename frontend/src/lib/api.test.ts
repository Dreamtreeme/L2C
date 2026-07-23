import { describe, expect, it } from "vitest";

import { parseStreamData } from "./api";

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
});
