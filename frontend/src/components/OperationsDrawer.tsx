import {
  Activity,
  Database,
  HardDrive,
  LoaderCircle,
  RefreshCw,
  Trash2,
  X,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { applyRetention } from "../lib/api";
import {
  formatBytes,
  PHASE_LABELS,
  relativeDate,
  STATUS_LABELS,
} from "../lib/format";
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
  const [working, setWorking] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (open) {
      setError("");
      void onRefresh();
    }
  }, [onRefresh, open]);

  const cleanupSummary = useMemo(() => {
    const files = data?.retention.files || {};
    const database = data?.retention.database || {};
    return {
      files:
        Number(files.log_count || 0) + Number(files.artifact_count || 0),
      rows: Object.values(database).reduce(
        (sum, value) => sum + Number(value || 0),
        0,
      ),
      bytes: Number(files.reclaimable_bytes || 0),
    };
  }, [data]);

  const runCleanup = async () => {
    if (
      !window.confirm(
        `만료 파일 ${cleanupSummary.files}개와 DB 이력 ${cleanupSummary.rows}건을 정리할까요?`,
      )
    ) {
      return;
    }
    setWorking(true);
    setError("");
    try {
      await applyRetention();
      await onRefresh();
    } catch (reason) {
      setError(
        reason instanceof Error
          ? reason.message
          : "만료 항목을 정리하지 못했습니다.",
      );
    } finally {
      setWorking(false);
    }
  };

  if (!open) {
    return null;
  }

  const inventory = Object.entries(data?.retention.inventory || {});

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
              disabled={working}
              onClick={() => void onRefresh()}
            >
              <RefreshCw size={17} className={working ? "spin" : ""} />
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
            <>
              <section className="operations-section">
                <div className="operations-heading">
                  <h2>
                    <Database size={16} />
                    저장 현황
                  </h2>
                  <button
                    type="button"
                    className="cleanup-button"
                    disabled={
                      working ||
                      (cleanupSummary.files === 0 &&
                        cleanupSummary.rows === 0)
                    }
                    onClick={() => void runCleanup()}
                  >
                    <Trash2 size={14} />
                    만료 항목 정리
                  </button>
                </div>
                <div className="inventory-grid">
                  {inventory.map(([name, count]) => (
                    <div key={name}>
                      <span>{name.replaceAll("_", " ")}</span>
                      <strong>{Number(count || 0).toLocaleString()}</strong>
                    </div>
                  ))}
                </div>
                <div className="cleanup-preview">
                  <HardDrive size={15} />
                  <span>
                    만료 파일 {cleanupSummary.files}개 · DB 이력{" "}
                    {cleanupSummary.rows}건 ·{" "}
                    {formatBytes(cleanupSummary.bytes)}
                  </span>
                </div>
                {error ? <p className="inline-error">{error}</p> : null}
              </section>

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
                          <strong>
                            {run.query || "제목 없는 조사"}
                          </strong>
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
            </>
          )}
        </div>
      </aside>
    </div>
  );
}
