import { describe, expect, it } from "vitest";

import {
  collapseInvestigationRuns,
  extractCitationIds,
  formatDuration,
  hostLabel,
} from "./format";

describe("format helpers", () => {
  it("중복 출처 식별자를 제거한다", () => {
    expect(
      extractCitationIds("첫 공고 [job_id:12], 다시 [job_id:12], 다음 [job_id:27]"),
    ).toEqual([12, 27]);
  });

  it("분 단위 실행시간을 표시한다", () => {
    expect(formatDuration(105.45)).toBe("1분 45초");
  });

  it("원본 사이트 호스트명을 정리한다", () => {
    expect(hostLabel("https://www.wanted.co.kr/wd/123")).toBe("wanted.co.kr");
  });

  it("같은 조사의 확인 응답은 최초 질문과 최신 상태로 묶는다", () => {
    const runs = collapseInvestigationRuns([
      {
        run_id: "run-3",
        query: "머신러닝 엔지니어",
        status: "completed",
        phase: "completed",
        updated_at: "2026-07-24T03:00:03Z",
        result: { investigation_id: "investigation-1" },
      },
      {
        run_id: "run-2",
        query: "IT·데이터",
        status: "waiting_input",
        phase: "clarification",
        updated_at: "2026-07-24T03:00:02Z",
        result: { investigation_id: "investigation-1" },
      },
      {
        run_id: "run-1",
        query: "채용공고 찾아줘",
        status: "waiting_input",
        phase: "clarification",
        updated_at: "2026-07-24T03:00:01Z",
        result: { investigation_id: "investigation-1" },
      },
    ]);

    expect(runs).toHaveLength(1);
    expect(runs[0]).toMatchObject({
      run_id: "run-3",
      query: "채용공고 찾아줘",
      status: "completed",
      updated_at: "2026-07-24T03:00:03Z",
    });
  });
});
