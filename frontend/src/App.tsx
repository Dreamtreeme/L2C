import { useCallback, useEffect, useMemo, useReducer, useState } from "react";

import { cancelRun, getOperations, streamChat } from "./lib/api";
import { extractCitationIds } from "./lib/format";
import type {
  ChatFinalPayload,
  ChatMessage,
  ChatRequest,
  ClarificationAnswer,
  OperationsResponse,
  PendingResume,
  RunEvent,
  RunMetrics,
  RunPhase,
} from "./types";
import { AppSidebar } from "./components/AppSidebar";
import { ChatWorkspace } from "./components/ChatWorkspace";
import { EvidencePanel } from "./components/EvidencePanel";
import { OperationsDrawer } from "./components/OperationsDrawer";

const MESSAGE_STORAGE_KEY = "l2c.workspace.messages.v1";
const CONVERSATION_STORAGE_KEY = "l2c.workspace.conversation.v1";

interface WorkspaceState {
  messages: ChatMessage[];
  isRunning: boolean;
  activeRunId: string | null;
  activePhase: RunPhase | null;
  pendingResume: PendingResume | null;
  metrics: RunMetrics | null;
  citationIds: number[];
  selectedJobId: number | null;
  cancelRequested: boolean;
}

type WorkspaceAction =
  | {
      type: "start_turn";
      user: ChatMessage;
      assistant: ChatMessage;
    }
  | { type: "processing"; assistantId: string; runId: string }
  | { type: "event"; assistantId: string; event: RunEvent }
  | {
      type: "final";
      assistantId: string;
      payload: ChatFinalPayload;
    }
  | { type: "error"; assistantId: string; message: string }
  | { type: "finish" }
  | { type: "cancel_requested" }
  | { type: "select_job"; jobId: number | null }
  | { type: "reset" };

function loadStoredMessages(): ChatMessage[] {
  try {
    const stored = JSON.parse(
      localStorage.getItem(MESSAGE_STORAGE_KEY) || "[]",
    ) as ChatMessage[];
    if (!Array.isArray(stored)) {
      return [];
    }
    return stored
      .filter(
        (item) =>
          item &&
          (item.role === "user" || item.role === "assistant") &&
          typeof item.text === "string",
      )
      .slice(-40)
      .map((item) => ({
        ...item,
        state: item.state === "pending" ? "error" : item.state,
        text:
          item.state === "pending"
            ? "이전 실행은 앱이 종료되어 중단됐습니다."
            : item.text,
        events: (item.events || []).slice(-12),
      }));
  } catch {
    return [];
  }
}

function initialState(): WorkspaceState {
  const messages = loadStoredMessages();
  const latestAssistant = [...messages]
    .reverse()
    .find((message) => message.role === "assistant");
  const citationIds = latestAssistant
    ? extractCitationIds(latestAssistant.text)
    : [];
  const pendingResume =
    latestAssistant?.state === "waiting_input" &&
    latestAssistant.runId &&
    latestAssistant.clarification
      ? {
          runId: latestAssistant.runId,
          investigationId: latestAssistant.investigationId || "",
          clarification: latestAssistant.clarification,
          status: "waiting_input" as const,
        }
      : null;
  return {
    messages,
    isRunning: false,
    activeRunId: null,
    activePhase: null,
    pendingResume,
    metrics: latestAssistant?.metrics || null,
    citationIds,
    selectedJobId: citationIds[0] || null,
    cancelRequested: false,
  };
}

function updateMessage(
  messages: ChatMessage[],
  id: string,
  updater: (message: ChatMessage) => ChatMessage,
): ChatMessage[] {
  return messages.map((message) =>
    message.id === id ? updater(message) : message,
  );
}

function workspaceReducer(
  state: WorkspaceState,
  action: WorkspaceAction,
): WorkspaceState {
  switch (action.type) {
    case "start_turn":
      return {
        ...state,
        messages: [...state.messages, action.user, action.assistant],
        isRunning: true,
        activeRunId: null,
        activePhase: "received",
        cancelRequested: false,
      };
    case "processing":
      return {
        ...state,
        activeRunId: action.runId,
        messages: updateMessage(
          state.messages,
          action.assistantId,
          (message) => ({ ...message, runId: action.runId }),
        ),
      };
    case "event":
      return {
        ...state,
        activePhase: action.event.phase,
        messages: updateMessage(
          state.messages,
          action.assistantId,
          (message) => ({
            ...message,
            runId: action.event.run_id || message.runId,
            events: [...(message.events || []), action.event].slice(-24),
          }),
        ),
      };
    case "final": {
      const citationIds = extractCitationIds(action.payload.text);
      const waitingForInput = action.payload.status === "waiting_input";
      return {
        ...state,
        activeRunId: action.payload.run_id,
        activePhase:
          action.payload.status === "completed"
            ? "completed"
            : action.payload.status === "cancelled"
              ? "cancelled"
              : state.activePhase,
        metrics: action.payload.metrics || null,
        citationIds,
        selectedJobId: citationIds[0] || null,
        pendingResume: waitingForInput
          ? {
              runId: action.payload.run_id,
              investigationId: action.payload.investigation_id || "",
              clarification: action.payload.clarification || null,
              status: action.payload.status,
            }
          : null,
        messages: updateMessage(
          state.messages,
          action.assistantId,
          (message) => ({
            ...message,
            text: action.payload.text,
            runId: action.payload.run_id,
            investigationId: action.payload.investigation_id || "",
            state:
              action.payload.status === "waiting_input"
                ? "waiting_input"
                : action.payload.status === "cancelled"
                  ? "cancelled"
                  : action.payload.status === "failed"
                    ? "error"
                    : "complete",
            clarification: action.payload.clarification || null,
            metrics: action.payload.metrics,
          }),
        ),
      };
    }
    case "error":
      return {
        ...state,
        activePhase: "failed",
        pendingResume: null,
        messages: updateMessage(
          state.messages,
          action.assistantId,
          (message) => ({
            ...message,
            text: action.message,
            state: "error",
          }),
        ),
      };
    case "finish":
      return {
        ...state,
        isRunning: false,
        activeRunId: null,
        cancelRequested: false,
      };
    case "cancel_requested":
      return { ...state, cancelRequested: true };
    case "select_job":
      return { ...state, selectedJobId: action.jobId };
    case "reset":
      return {
        ...initialState(),
        messages: [],
        citationIds: [],
        selectedJobId: null,
        metrics: null,
      };
  }
}

function createId(prefix: string): string {
  const random =
    typeof crypto !== "undefined" && crypto.randomUUID
      ? crypto.randomUUID()
      : Math.random().toString(36).slice(2);
  return `${prefix}-${random}`;
}

function getConversationId(): string {
  const existing = localStorage.getItem(CONVERSATION_STORAGE_KEY);
  if (existing) {
    return existing;
  }
  const created = createId("conversation");
  localStorage.setItem(CONVERSATION_STORAGE_KEY, created);
  return created;
}

export default function App() {
  const [state, dispatch] = useReducer(workspaceReducer, undefined, initialState);
  const [conversationId, setConversationId] = useState(getConversationId);
  const [operations, setOperations] = useState<OperationsResponse | null>(null);
  const [operationsOpen, setOperationsOpen] = useState(false);
  const [leftOpen, setLeftOpen] = useState(false);
  const [evidenceOpen, setEvidenceOpen] = useState(false);
  const [connectionState, setConnectionState] = useState<
    "checking" | "online" | "offline"
  >("checking");

  const refreshOperations = useCallback(async () => {
    try {
      const data = await getOperations();
      setOperations(data);
      setConnectionState("online");
    } catch {
      setConnectionState("offline");
    }
  }, []);

  useEffect(() => {
    void refreshOperations();
  }, [refreshOperations]);

  useEffect(() => {
    if (state.isRunning) {
      return;
    }
    const persisted = state.messages.slice(-40).map((message) => ({
      ...message,
      events: (message.events || []).slice(-12),
    }));
    localStorage.setItem(MESSAGE_STORAGE_KEY, JSON.stringify(persisted));
  }, [state.isRunning, state.messages]);

  const sendQuery = useCallback(
    async (
      query: string,
      clarificationAnswer: ClarificationAnswer | null = null,
    ) => {
      const trimmed = query.trim();
      if (!trimmed || state.isRunning) {
        return;
      }

      const now = new Date().toISOString();
      const userId = createId("user");
      const assistantId = createId("assistant");
      const user: ChatMessage = {
        id: userId,
        role: "user",
        text: trimmed,
        createdAt: now,
        state: "complete",
      };
      const assistant: ChatMessage = {
        id: assistantId,
        role: "assistant",
        text: "",
        createdAt: now,
        state: "pending",
        events: [],
      };
      dispatch({ type: "start_turn", user, assistant });

      const resume = state.pendingResume;
      const request: ChatRequest = {
        query: trimmed,
        resume_run_id: resume?.runId || null,
        conversation_id: conversationId,
        investigation_id: resume?.investigationId || "",
        clarification_answer: clarificationAnswer,
      };

      try {
        await streamChat(request, {
          onProcessing: (payload) => {
            dispatch({
              type: "processing",
              assistantId,
              runId: payload.run_id,
            });
          },
          onEvent: (event) => {
            dispatch({ type: "event", assistantId, event });
          },
          onFinal: (payload) => {
            dispatch({ type: "final", assistantId, payload });
          },
          onError: (payload) => {
            dispatch({
              type: "error",
              assistantId,
              message: payload.message || "조사 실행에 실패했습니다.",
            });
          },
        });
      } catch (error) {
        dispatch({
          type: "error",
          assistantId,
          message:
            error instanceof Error
              ? error.message
              : "백엔드와 통신하지 못했습니다.",
        });
        setConnectionState("offline");
      } finally {
        dispatch({ type: "finish" });
        void refreshOperations();
      }
    },
    [
      conversationId,
      refreshOperations,
      state.isRunning,
      state.pendingResume,
    ],
  );

  const submitQuery = useCallback(
    (query: string) => {
      const clarification = state.pendingResume?.clarification;
      const answer = clarification
        ? {
            question_id: clarification.question_id,
            custom_value: query.trim(),
          }
        : null;
      void sendQuery(query, answer);
    },
    [sendQuery, state.pendingResume],
  );

  const submitClarification = useCallback(
    (answer: ClarificationAnswer, label: string) => {
      void sendQuery(label, answer);
    },
    [sendQuery],
  );

  const requestCancel = useCallback(async () => {
    if (!state.activeRunId || state.cancelRequested) {
      return;
    }
    dispatch({ type: "cancel_requested" });
    try {
      await cancelRun(state.activeRunId);
    } catch {
      dispatch({ type: "finish" });
    }
  }, [state.activeRunId, state.cancelRequested]);

  const startNewConversation = useCallback(() => {
    if (state.isRunning) {
      return;
    }
    const nextId = createId("conversation");
    localStorage.setItem(CONVERSATION_STORAGE_KEY, nextId);
    localStorage.removeItem(MESSAGE_STORAGE_KEY);
    setConversationId(nextId);
    dispatch({ type: "reset" });
    setEvidenceOpen(false);
    setLeftOpen(false);
  }, [state.isRunning]);

  const activeStatus = useMemo(() => {
    if (state.cancelRequested) {
      return "취소 처리 중";
    }
    if (state.pendingResume) {
      return "사용자 선택 필요";
    }
    if (state.isRunning) {
      return "조사 실행 중";
    }
    if (connectionState === "offline") {
      return "백엔드 연결 끊김";
    }
    return "준비됨";
  }, [
    connectionState,
    state.cancelRequested,
    state.isRunning,
    state.pendingResume,
  ]);

  return (
    <div className="app-frame">
      <AppSidebar
        open={leftOpen}
        connectionState={connectionState}
        recentRuns={operations?.runs || []}
        activeRunId={state.activeRunId}
        disabled={state.isRunning}
        onClose={() => setLeftOpen(false)}
        onNewConversation={startNewConversation}
        onOpenOperations={() => {
          setOperationsOpen(true);
          setLeftOpen(false);
          void refreshOperations();
        }}
      />

      <ChatWorkspace
        messages={state.messages}
        isRunning={state.isRunning}
        cancelRequested={state.cancelRequested}
        activeStatus={activeStatus}
        activePhase={state.activePhase}
        pendingResume={state.pendingResume}
        onOpenSidebar={() => setLeftOpen(true)}
        onOpenEvidence={() => setEvidenceOpen(true)}
        onSelectCitation={(jobId) => {
          dispatch({ type: "select_job", jobId });
          setEvidenceOpen(true);
        }}
        onSubmit={submitQuery}
        onSubmitClarification={submitClarification}
        onCancel={() => void requestCancel()}
      />

      <EvidencePanel
        open={evidenceOpen}
        selectedJobId={state.selectedJobId}
        citationIds={state.citationIds}
        metrics={state.metrics}
        messages={state.messages}
        onClose={() => setEvidenceOpen(false)}
        onSelectJob={(jobId) =>
          dispatch({ type: "select_job", jobId })
        }
      />

      <OperationsDrawer
        open={operationsOpen}
        data={operations}
        onClose={() => setOperationsOpen(false)}
        onRefresh={refreshOperations}
      />
    </div>
  );
}
