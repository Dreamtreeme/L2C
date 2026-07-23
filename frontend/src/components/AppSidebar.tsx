import {
  Activity,
  CircleDot,
  Clock3,
  PanelLeftClose,
  Plus,
  ScanSearch,
} from "lucide-react";

import { PHASE_LABELS, relativeDate, STATUS_LABELS } from "../lib/format";
import type { RunRecord } from "../types";

interface AppSidebarProps {
  open: boolean;
  connectionState: "checking" | "online" | "offline";
  recentRuns: RunRecord[];
  activeRunId: string | null;
  disabled: boolean;
  onClose: () => void;
  onNewConversation: () => void;
  onOpenOperations: () => void;
}

export function AppSidebar({
  open,
  connectionState,
  recentRuns,
  activeRunId,
  disabled,
  onClose,
  onNewConversation,
  onOpenOperations,
}: AppSidebarProps) {
  return (
    <>
      <button
        type="button"
        className={`mobile-scrim ${open ? "is-visible" : ""}`}
        aria-label="탐색 메뉴 닫기"
        onClick={onClose}
      />
      <aside className={`app-sidebar ${open ? "is-open" : ""}`}>
        <div className="brand-row">
          <span className="brand-mark" aria-hidden="true">
            <ScanSearch size={19} strokeWidth={2.1} />
          </span>
          <div className="brand-copy">
            <strong>L2C</strong>
            <span>채용 조사</span>
          </div>
          <button
            type="button"
            className="icon-button sidebar-close"
            title="메뉴 닫기"
            aria-label="메뉴 닫기"
            onClick={onClose}
          >
            <PanelLeftClose size={18} />
          </button>
        </div>

        <button
          type="button"
          className="new-investigation-button"
          disabled={disabled}
          onClick={onNewConversation}
        >
          <Plus size={17} />
          새 조사
        </button>

        <nav className="sidebar-nav" aria-label="주요 메뉴">
          <button type="button" className="sidebar-nav-item is-active">
            <CircleDot size={16} />
            현재 조사
          </button>
          <button
            type="button"
            className="sidebar-nav-item"
            onClick={onOpenOperations}
          >
            <Activity size={16} />
            실행 현황
          </button>
        </nav>

        <section className="recent-section">
          <div className="section-label">
            <Clock3 size={14} />
            최근 실행
          </div>
          <div className="recent-run-list">
            {recentRuns.length === 0 ? (
              <p className="sidebar-empty">실행 기록이 없습니다.</p>
            ) : (
              recentRuns.slice(0, 8).map((run) => (
                <div
                  className={`recent-run ${
                    run.run_id === activeRunId ? "is-active" : ""
                  }`}
                  key={run.run_id}
                >
                  <p>{run.user_query || run.query || "제목 없는 조사"}</p>
                  <span>
                    {PHASE_LABELS[run.phase] || STATUS_LABELS[run.status]}
                    {run.updated_at ? ` · ${relativeDate(run.updated_at)}` : ""}
                  </span>
                </div>
              ))
            )}
          </div>
        </section>

        <div className="connection-row">
          <span
            className={`connection-dot is-${connectionState}`}
            aria-hidden="true"
          />
          <span>
            {connectionState === "online"
              ? "로컬 백엔드 연결됨"
              : connectionState === "offline"
                ? "백엔드 연결 끊김"
                : "연결 확인 중"}
          </span>
        </div>
      </aside>
    </>
  );
}
