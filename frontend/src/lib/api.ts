import type {
  ChatErrorPayload,
  ChatFinalPayload,
  ChatRequest,
  JobDetail,
  OperationsResponse,
  ProcessingPayload,
  RunEvent,
  RunRecord,
} from "../types";

const JSON_HEADERS = {
  "Content-Type": "application/json",
};

export interface ChatStreamHandlers {
  onProcessing: (payload: ProcessingPayload) => void;
  onEvent: (event: RunEvent) => void;
  onFinal: (payload: ChatFinalPayload) => void;
  onError: (payload: ChatErrorPayload) => void;
}

type StreamFrame =
  | { type: "processing"; payload: ProcessingPayload }
  | { type: "event"; payload: RunEvent }
  | { type: "final"; payload: ChatFinalPayload }
  | { type: "error"; payload: ChatErrorPayload }
  | { type: "done" }
  | { type: "unknown"; payload: string };

async function errorMessage(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as {
      detail?: string;
      message?: string;
    };
    return body.detail || body.message || `요청 실패 (${response.status})`;
  } catch {
    return `요청 실패 (${response.status})`;
  }
}

export function parseStreamData(data: string): StreamFrame {
  const value = data.trim();
  if (value === "[DONE]") {
    return { type: "done" };
  }

  const prefixes = [
    ["[PROCESSING]", "processing"],
    ["[EVENT]", "event"],
    ["[FINAL]", "final"],
    ["[ERROR]", "error"],
  ] as const;

  for (const [prefix, type] of prefixes) {
    if (!value.startsWith(prefix)) {
      continue;
    }
    const payload = JSON.parse(value.slice(prefix.length).trim()) as never;
    return { type, payload } as StreamFrame;
  }

  return { type: "unknown", payload: value };
}

function dispatchFrame(frame: StreamFrame, handlers: ChatStreamHandlers): boolean {
  switch (frame.type) {
    case "processing":
      handlers.onProcessing(frame.payload);
      return false;
    case "event":
      handlers.onEvent(frame.payload);
      return false;
    case "final":
      handlers.onFinal(frame.payload);
      return false;
    case "error":
      handlers.onError(frame.payload);
      return false;
    case "done":
      return true;
    default:
      return false;
  }
}

export async function streamChat(
  request: ChatRequest,
  handlers: ChatStreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(request),
    signal,
  });

  if (!response.ok) {
    throw new Error(await errorMessage(response));
  }
  if (!response.body) {
    throw new Error("응답 스트림을 열지 못했습니다.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let finished = false;

  while (!finished) {
    const chunk = await reader.read();
    buffer += decoder.decode(chunk.value, { stream: !chunk.done });
    const frames = buffer.split(/\r?\n\r?\n/);
    buffer = frames.pop() ?? "";

    for (const rawFrame of frames) {
      const data = rawFrame
        .split(/\r?\n/)
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n");
      if (!data) {
        continue;
      }
      finished = dispatchFrame(parseStreamData(data), handlers);
      if (finished) {
        break;
      }
    }

    if (chunk.done) {
      break;
    }
  }

  const tail = buffer
    .split(/\r?\n/)
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trimStart())
    .join("\n");
  if (tail && !finished) {
    dispatchFrame(parseStreamData(tail), handlers);
  }
}

export async function cancelRun(runId: string): Promise<void> {
  const response = await fetch(`/api/runs/${encodeURIComponent(runId)}/cancel`, {
    method: "POST",
  });
  if (!response.ok) {
    throw new Error(await errorMessage(response));
  }
}

export async function getRun(runId: string): Promise<RunRecord> {
  const response = await fetch(`/api/runs/${encodeURIComponent(runId)}`);
  if (!response.ok) {
    throw new Error(await errorMessage(response));
  }
  return (await response.json()) as RunRecord;
}

export async function getJob(jobId: number): Promise<JobDetail> {
  const response = await fetch(`/api/jobs/${jobId}`);
  if (!response.ok) {
    throw new Error(await errorMessage(response));
  }
  return (await response.json()) as JobDetail;
}

export async function getOperations(): Promise<OperationsResponse> {
  const response = await fetch("/api/operations");
  if (!response.ok) {
    throw new Error(await errorMessage(response));
  }
  return (await response.json()) as OperationsResponse;
}
