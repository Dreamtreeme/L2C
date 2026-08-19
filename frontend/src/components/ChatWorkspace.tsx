import {
  Menu,
  PanelRight,
  Search,
  Send,
  Square,
} from "lucide-react";
import {
  FormEvent,
  KeyboardEvent,
  useEffect,
  useRef,
  useState,
} from "react";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

import { PHASE_LABELS } from "../lib/format";
import type {
  ChatMessage,
  ClarificationAnswer,
  PendingResume,
  RunPhase,
} from "../types";
import { MessageItem } from "./MessageItem";

const EXAMPLE_QUERIES = [
  "최근 AI 엔지니어 공고 3개 찾아줘",
  "신입 백엔드 공고의 주요 기술을 정리해줘",
  "서울 iOS 개발자 공고 2개 비교해줘",
];

interface ChatWorkspaceProps {
  messages: ChatMessage[];
  isRunning: boolean;
  cancelRequested: boolean;
  activeStatus: string;
  statusError: boolean;
  activePhase: RunPhase | null;
  pendingResume: PendingResume | null;
  onOpenSidebar: () => void;
  onOpenEvidence: () => void;
  onSelectCitation: (jobId: number, messageId: string) => void;
  onSubmit: (query: string) => void;
  onSubmitClarification: (
    answer: ClarificationAnswer,
    label: string,
  ) => void;
  onCancel: () => void;
}

export function ChatWorkspace({
  messages,
  isRunning,
  cancelRequested,
  activeStatus,
  statusError,
  activePhase,
  pendingResume,
  onOpenSidebar,
  onOpenEvidence,
  onSelectCitation,
  onSubmit,
  onSubmitClarification,
  onCancel,
}: ChatWorkspaceProps) {
  const [query, setQuery] = useState("");
  const scrollerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const stickToBottomRef = useRef(true);
  const previousMessageCountRef = useRef(messages.length);

  useEffect(() => {
    const scroller = scrollerRef.current;
    const hasNewMessage = messages.length > previousMessageCountRef.current;
    if (scroller && (stickToBottomRef.current || hasNewMessage)) {
      scroller.scrollTop = scroller.scrollHeight;
    }
    previousMessageCountRef.current = messages.length;
  }, [messages]);

  const submit = (event?: FormEvent) => {
    event?.preventDefault();
    const value = query.trim();
    if (!value || isRunning) {
      return;
    }
    setQuery("");
    onSubmit(value);
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.nativeEvent.isComposing) {
      return;
    }
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  };

  const fillExampleQuery = (example: string) => {
    setQuery(example);
    inputRef.current?.focus();
  };

  return (
    <main className="chat-workspace">
      <header className="workspace-header">
        <Button
          type="button"
          variant="ghost"
          size="icon-lg"
          className="icon-button mobile-only"
          title="탐색 메뉴"
          aria-label="탐색 메뉴 열기"
          onClick={onOpenSidebar}
        >
          <Menu size={19} />
        </Button>
        <div className="workspace-title">
          <strong>현재 조사</strong>
          <span
            className={`run-status ${
              statusError
                ? "is-error"
                : isRunning
                  ? "is-running"
                  : pendingResume
                    ? "is-waiting"
                    : ""
            }`}
          >
            <i aria-hidden="true" />
            {activeStatus}
          </span>
        </div>
        {isRunning && activePhase ? (
          <span className="current-phase">
            {PHASE_LABELS[activePhase]}
          </span>
        ) : null}
        <Button
          type="button"
          variant="ghost"
          size="icon-lg"
          className="icon-button evidence-toggle"
          title="근거 및 실행 정보"
          aria-label="근거 및 실행 정보 열기"
          onClick={onOpenEvidence}
        >
          <PanelRight size={19} />
        </Button>
      </header>

      <div
        className="message-scroller"
        ref={scrollerRef}
        onScroll={(event) => {
          const scroller = event.currentTarget;
          stickToBottomRef.current =
            scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight <
            80;
        }}
      >
        {messages.length === 0 ? (
          <section className="empty-workspace">
            <span className="empty-icon" aria-hidden="true">
              <Search size={24} />
            </span>
            <h1>어떤 채용 정보를 조사할까요?</h1>
            <div className="example-query-list">
              {EXAMPLE_QUERIES.map((example) => (
                <Button
                  type="button"
                  variant="ghost"
                  className="w-full justify-start"
                  key={example}
                  onClick={() => fillExampleQuery(example)}
                >
                  {example}
                </Button>
              ))}
            </div>
          </section>
        ) : (
          <div className="message-list">
            {messages.map((message) => (
              <MessageItem
                key={message.id}
                message={message}
                isRunning={isRunning}
                pendingResume={pendingResume}
                onSelectCitation={onSelectCitation}
                onSubmitClarification={onSubmitClarification}
              />
            ))}
          </div>
        )}
      </div>

      <footer className="composer-area">
        <form className="composer" onSubmit={submit}>
          <Textarea
            ref={inputRef}
            rows={1}
            value={query}
            disabled={isRunning}
            placeholder={
              pendingResume
                ? "선택지를 고르거나 직접 답변하세요"
                : "채용시장에 대해 질문하세요"
            }
            aria-label="질문 입력"
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={handleKeyDown}
          />
          {isRunning ? (
            <Button
              type="button"
              variant="destructive"
              size="icon"
              className="composer-action is-stop"
              title="실행 취소"
              aria-label="실행 취소"
              disabled={cancelRequested}
              onClick={onCancel}
            >
              <Square size={15} fill="currentColor" />
            </Button>
          ) : (
            <Button
              type="submit"
              size="icon"
              className="composer-action"
              title="질문 보내기"
              aria-label="질문 보내기"
              disabled={!query.trim()}
            >
              <Send size={18} />
            </Button>
          )}
        </form>
      </footer>
    </main>
  );
}
