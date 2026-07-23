import { describe, expect, it } from "vitest";

import { extractCitationIds, formatDuration, hostLabel } from "./format";

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
});
