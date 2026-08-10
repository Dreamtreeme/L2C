export type RunStatus =
  | "queued"
  | "running"
  | "waiting_input"
  | "completed"
  | "partial"
  | "failed"
  | "cancelled";

export type RunPhase =
  | "received"
  | "database"
  | "planning"
  | "clarification"
  | "collection"
  | "validation"
  | "persistence"
  | "answering"
  | "completed"
  | "partial"
  | "failed"
  | "cancelled";

export interface ClarificationOption {
  option_id: string;
  label: string;
  value: string;
  collection_search_term?: string;
  matching_count?: number;
  concept_count?: number;
  description?: string;
}

export interface ClarificationQuestion {
  question_id: string;
  field: string;
  question: string;
  options: ClarificationOption[];
  allow_custom: boolean;
  reason?: string;
  candidate_count?: number;
  concept_count?: number;
  facet_type?: string;
}

export interface ClarificationAnswer {
  question_id: string;
  selected_option_id?: string;
  value?: string;
  custom_value?: string;
}

export interface ChatRequest {
  query: string;
  conversation_id: string;
  investigation_id?: string;
  clarification_answer?: ClarificationAnswer | null;
}

export interface RunEvent {
  run_id: string;
  event: string;
  phase: RunPhase;
  status: RunStatus;
  message: string;
  data: Record<string, unknown>;
  timestamp: string;
}

export interface TokenUsage {
  input_tokens?: number;
  output_tokens?: number;
  total_tokens?: number;
}

export interface RunStepMetric {
  stage?: string;
  phase?: string;
  component?: string;
  duration_sec?: number;
  action_source?: string;
  success?: boolean;
  [key: string]: unknown;
}

export interface RunMetrics {
  run_id?: string;
  duration_sec?: number;
  steps?: RunStepMetric[];
  llm?: {
    totals?: TokenUsage;
    cost?: {
      currency?: string;
      estimated_total?: number | null;
      unpriced_models?: string[];
    };
    calls?: Array<Record<string, unknown>>;
  };
  outcome?: {
    status?: string;
    last_phase?: string;
    failure_stage?: string;
    failure_code?: string;
  };
  langsmith?: {
    enabled?: boolean;
    project?: string;
    trace_id?: string;
  };
}

export interface AnswerEvidenceRef {
  citation_id: number;
  document_id: number;
  field: string;
  item_index?: number | null;
  evidence_text: string;
}

export interface GroundedAnswerLine {
  kind: "overview" | "detail" | "caveat";
  document_id?: number | null;
  title: string;
  text: string;
  citation_ids: number[];
}

export interface GroundedAnswer {
  lines: GroundedAnswerLine[];
  citations: AnswerEvidenceRef[];
}

export interface ChatFinalPayload {
  run_id: string;
  text: string;
  status: RunStatus;
  grounded_answer?: GroundedAnswer | null;
  clarification?: ClarificationQuestion | null;
  investigation_id?: string;
  resume_mode?: string;
  conversation_id?: string;
  metrics?: RunMetrics;
}

export interface ChatErrorPayload {
  run_id: string;
  message: string;
}

export interface ProcessingPayload {
  run_id: string;
}

export interface JobDetail {
  id: number;
  company_name: string;
  position: string;
  url: string;
  posted_at?: string | null;
  posted_at_text?: string | null;
  evidence_hash?: string | null;
  collected_at?: string | null;
  raw_text: string;
  source_platform?: string | null;
}

export interface RunRecord {
  run_id: string;
  query?: string;
  conversation_id?: string;
  status: RunStatus;
  phase: RunPhase;
  message?: string;
  created_at?: string;
  updated_at?: string;
  result?: Record<string, unknown>;
  error?: string;
  cancel_requested?: boolean;
}

export interface RetentionPreview {
  inventory?: Record<string, number>;
  files?: {
    log_count?: number;
    artifact_count?: number;
    reclaimable_bytes?: number;
  };
  database?: Record<string, number>;
  [key: string]: unknown;
}

export interface OperationsResponse {
  runs: RunRecord[];
  retention: RetentionPreview;
}

export type ChatMessageState =
  | "pending"
  | "complete"
  | "waiting_input"
  | "cancelled"
  | "error";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  createdAt: string;
  state: ChatMessageState;
  runId?: string;
  investigationId?: string;
  events?: RunEvent[];
  clarification?: ClarificationQuestion | null;
  metrics?: RunMetrics;
}

export interface PendingResume {
  runId: string;
  investigationId: string;
  clarification: ClarificationQuestion | null;
  status: RunStatus;
}
