import { Activity, LoaderCircle, RefreshCw, X } from "lucide-react";
import { useEffect } from "react";

import { PHASE_LABELS, relativeDate, STATUS_LABELS } from "../lib/format";
import type { OperationsResponse } from "../types";

interface OperationsDrawerProps {
  open: boolean;
  data: OperationsResponse | null;
  onClose: () => void;
  onRefresh: () => Promise<void>;
}

export function OperationsDrawer({
  open,
  data,
  onClose,
  onRefresh,
}: OperationsDrawerProps) {
  useEffect(() => {
    if (open) {
      void onRefresh();
    }
  }, [onRefresh, open]);

  if (!open) {
    return null;
  }

  return (
    <div className="drawer-layer" role="dialog" aria-modal="true">
      <button
        type="button"
        className="drawer-scrim"
        aria-label="운영 현황 닫기"
        onClick={onClose}
      />
      <aside className="operations-drawer">
        <header className="panel-header">
          <div>
            <Activity size={18} />
            <strong>실행 현황</strong>
          </div>
          <div className="header-actions">
            <button
              type="button"
              className="icon-button"
              title="새로고침"
              aria-label="운영 현황 새로고침"
              onClick={() => void onRefresh()}
            >
              <RefreshCw size={17} />
            </button>
            <button
              type="button"
              className="icon-button"
              title="닫기"
              aria-label="운영 현황 닫기"
              onClick={onClose}
            >
              <X size={18} />
            </button>
          </div>
        </header>

        <div className="operations-scroll">
          {!data ? (
            <div className="panel-empty">
              <LoaderCircle size={21} className="spin" />
              <span>운영 정보를 불러오는 중</span>
            </div>
          ) : (
            <section className="operations-section">
              <div className="operations-heading">
                <h2>
                  <Activity size={16} />
                  최근 실행
                </h2>
              </div>
              <div className="operations-run-list">
                {data.runs.length === 0 ? (
                  <p className="muted-text">실행 기록이 없습니다.</p>
                ) : (
                  data.runs.map((run) => (
                    <div className="operations-run" key={run.run_id}>
                      <i className={`is-${run.status}`} aria-hidden="true" />
                      <div>
                        <strong>{run.query || "제목 없는 조사"}</strong>
                        <span>
                          {run.message ||
                            PHASE_LABELS[run.phase] ||
                            STATUS_LABELS[run.status]}
                        </span>
                      </div>
                      <time>{relativeDate(run.updated_at)}</time>
                    </div>
                  ))
                )}
              </div>
            </section>
          )}
        </div>
      </aside>
    </div>
  );
}
