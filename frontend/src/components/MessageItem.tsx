import DOMPurify from "dompurify";
import { AlertCircle, Bot, Clock3, FileText, UserRound } from "lucide-react";
import { marked } from "marked";
import { useMemo } from "react";

import { Button } from "@/components/ui/button";

import { formatDuration } from "../lib/format";
import type {
  ChatMessage,
  ClarificationAnswer,
  PendingResume,
} from "../types";
import { ClarificationPanel } from "./ClarificationPanel";
import { ProgressTimeline } from "./ProgressTimeline";

interface MessageItemProps {
  message: ChatMessage;
  isRunning: boolean;
  pendingResume: PendingResume | null;
  onSelectCitation: (jobId: number, messageId: string) => void;
  onSubmitClarification: (
    answer: ClarificationAnswer,
    label: string,
  ) => void;
}

function MarkdownContent({
  text,
  onSelectCitation,
}: {
  text: string;
  onSelectCitation: (jobId: number) => void;
}) {
  const html = useMemo(() => {
    const withCitations = text
      .replace(
        /\[job_id:(\d+)\]/g,
        '<button type="button" class="citation-token" data-job-id="$1">출처 $1</button>',
      )
      .replace(
        /\[출처 확인 불가\]/g,
        '<span class="citation-missing">출처 확인 불가</span>',
      );
    return DOMPurify.sanitize(marked.parse(withCitations) as string, {
      ADD_ATTR: ["data-job-id", "target", "rel"],
    });
  }, [text]);

  return (
    <div
      className="markdown-body"
      onClick={(event) => {
        const target = event.target as HTMLElement;
        const citation = target.closest<HTMLElement>("[data-job-id]");
        if (!citation) {
          return;
        }
        event.preventDefault();
        const jobId = Number(citation.dataset.jobId);
        if (Number.isFinite(jobId)) {
          onSelectCitation(jobId);
        }
      }}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

export function MessageItem({
  message,
  isRunning,
  pendingResume,
  onSelectCitation,
  onSubmitClarification,
}: MessageItemProps) {
  if (message.role === "user") {
    return (
      <article className="message-row is-user">
        <div className="message-avatar" aria-hidden="true">
          <UserRound size={16} />
        </div>
        <div className="user-message">{message.text}</div>
      </article>
    );
  }

  const isPending = message.state === "pending";
  const hasInlineCitations = /\[job_id:\d+\]/.test(message.text);
  const clarificationActive =
    message.state === "waiting_input" &&
    pendingResume?.clarification?.question_id ===
      message.clarification?.question_id;

  return (
    <article className={`message-row is-assistant is-${message.state}`}>
      <div className="message-avatar" aria-hidden="true">
        {message.state === "error" ? (
          <AlertCircle size={17} />
        ) : (
          <Bot size={17} />
        )}
      </div>
      <div className="assistant-message">
        {isPending ? (
          <ProgressTimeline
            events={message.events || []}
            pending={isPending}
          />
        ) : (
          <>
            <MarkdownContent
              text={message.text || "응답 내용이 없습니다."}
              onSelectCitation={(jobId) =>
                onSelectCitation(jobId, message.id)
              }
            />
            {!hasInlineCitations && message.citationIds?.length ? (
              <div className="message-source-list" aria-label="답변 출처">
                {message.citationIds.map((jobId) => (
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    key={jobId}
                    onClick={() => onSelectCitation(jobId, message.id)}
                  >
                    <FileText size={13} />
                    출처 {jobId}
                  </Button>
                ))}
              </div>
            ) : null}
            {message.metrics?.duration_sec ? (
              <div className="message-metrics">
                <Clock3 size={13} />
                {formatDuration(message.metrics.duration_sec)}
              </div>
            ) : null}
          </>
        )}
        {message.clarification ? (
          <ClarificationPanel
            clarification={message.clarification}
            disabled={isRunning || !clarificationActive}
            onSubmit={onSubmitClarification}
          />
        ) : null}
      </div>
    </article>
  );
}
