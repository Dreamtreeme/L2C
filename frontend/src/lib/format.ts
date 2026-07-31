import type { RunPhase, RunRecord, RunStatus } from "../types";

export const PHASE_LABELS: Record<RunPhase, string> = {
  received: "요청 접수",
  database: "DB 근거 확인",
  planning: "조사 계획",
  clarification: "조건 확인",
  collection: "웹 수집",
  review: "수집 결과 검토",
  persistence: "DB 저장",
  answering: "답변 작성",
  completed: "완료",
  partial: "부분 완료",
  failed: "실패",
  cancelled: "취소",
};

export const STATUS_LABELS: Record<RunStatus, string> = {
  queued: "대기",
  running: "실행 중",
  waiting_approval: "승인 대기",
  waiting_input: "사용자 선택 필요",
  completed: "완료",
  partial: "부분 완료",
  failed: "실패",
  cancelled: "취소",
};

export function formatDuration(seconds?: number): string {
  const value = Number(seconds || 0);
  if (value < 1) {
    return `${Math.round(value * 1000)}ms`;
  }
  if (value < 60) {
    return `${value.toFixed(1)}초`;
  }
  const minutes = Math.floor(value / 60);
  const remainder = Math.round(value % 60);
  return `${minutes}분 ${remainder}초`;
}

export function formatNumber(value?: number): string {
  return new Intl.NumberFormat("ko-KR").format(Number(value || 0));
}

export function formatBytes(value?: number): string {
  const bytes = Number(value || 0);
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 ** 2) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  if (bytes < 1024 ** 3) {
    return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  }
  return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
}

export function formatDate(value?: string | null): string {
  if (!value) {
    return "-";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return new Intl.DateTimeFormat("ko-KR", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

export function relativeDate(value?: string): string {
  if (!value) {
    return "";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "";
  }
  const elapsed = Date.now() - date.getTime();
  if (elapsed < 60_000) {
    return "방금 전";
  }
  if (elapsed < 3_600_000) {
    return `${Math.floor(elapsed / 60_000)}분 전`;
  }
  if (elapsed < 86_400_000) {
    return `${Math.floor(elapsed / 3_600_000)}시간 전`;
  }
  return `${Math.floor(elapsed / 86_400_000)}일 전`;
}

export function extractCitationIds(text: string): number[] {
  const matches = text.matchAll(/\[job_id:(\d+)\]/g);
  return [...new Set(Array.from(matches, (match) => Number(match[1])))];
}

export function collapseInvestigationRuns(runs: RunRecord[]): RunRecord[] {
  const grouped = new Map<string, RunRecord>();

  for (const run of runs) {
    const investigationId =
      typeof run.result?.investigation_id === "string"
        ? run.result.investigation_id
        : "";
    const key = investigationId || run.run_id;
    const latest = grouped.get(key);

    if (!latest) {
      grouped.set(key, { ...run });
      continue;
    }

    grouped.set(key, {
      ...latest,
      query: run.query || latest.query,
      user_query: run.user_query || latest.user_query,
      created_at: run.created_at || latest.created_at,
    });
  }

  return Array.from(grouped.values());
}

export function hostLabel(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return "원본 사이트";
  }
}
