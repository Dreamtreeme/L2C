import {
  Activity,
  Building2,
  CalendarDays,
  Clock3,
  Coins,
  Database,
  ExternalLink,
  FileText,
  Gauge,
  Hash,
  Link2,
  LoaderCircle,
  MapPin,
  X,
} from "lucide-react";
import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";

import { getJob } from "../lib/api";
import {
  formatDate,
  formatDuration,
  formatNumber,
  hostLabel,
  PHASE_LABELS,
} from "../lib/format";
import type { JobDetail, RunEvent, RunMetrics } from "../types";

type EvidenceTab = "sources" | "run";

interface EvidencePanelProps {
  open: boolean;
  selectedJobId: number | null;
  citationIds: number[];
  metrics: RunMetrics | null;
  events: RunEvent[];
  onClose: () => void;
  onSelectJob: (jobId: number) => void;
}

function metricValue(
  label: string,
  value: string,
  icon: React.ReactNode,
) {
  return (
    <div className="metric-cell">
      <span>{icon}</span>
      <div>
        <small>{label}</small>
        <strong>{value}</strong>
      </div>
    </div>
  );
}

export function EvidencePanel({
  open,
  selectedJobId,
  citationIds,
  metrics,
  events,
  onClose,
  onSelectJob,
}: EvidencePanelProps) {
  const [tab, setTab] = useState<EvidenceTab>("sources");
  const [job, setJob] = useState<JobDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!selectedJobId) {
      setJob(null);
      return;
    }
    let active = true;
    setLoading(true);
    setError("");
    void getJob(selectedJobId)
      .then((data) => {
        if (active) {
          setJob(data);
        }
      })
      .catch((reason) => {
        if (active) {
          setJob(null);
          setError(
            reason instanceof Error
              ? reason.message
              : "공고 정보를 불러오지 못했습니다.",
          );
        }
      })
      .finally(() => {
        if (active) {
          setLoading(false);
        }
      });
    return () => {
      active = false;
    };
  }, [selectedJobId]);

  useEffect(() => {
    if (selectedJobId) {
      setTab("sources");
    }
  }, [selectedJobId]);

  const latestEvents = events.slice(-12);
  const totalTokens = metrics?.llm?.totals?.total_tokens || 0;
  const estimatedCost = metrics?.llm?.cost?.estimated_total;
  const steps = metrics?.steps || [];

  return (
    <>
      <button
        type="button"
        className={`evidence-scrim ${open ? "is-visible" : ""}`}
        aria-label="근거 패널 닫기"
        onClick={onClose}
      />
      <aside className={`evidence-panel ${open ? "is-open" : ""}`}>
        <header className="panel-header">
          <strong>근거 및 실행 정보</strong>
          <Button
            type="button"
            variant="ghost"
            size="icon-lg"
            className="icon-button evidence-close"
            title="패널 닫기"
            aria-label="근거 패널 닫기"
            onClick={onClose}
          >
            <X size={18} />
          </Button>
        </header>

        <Tabs
          className="evidence-tabs"
          value={tab}
          onValueChange={(value) => setTab(value as EvidenceTab)}
        >
          <TabsList className="panel-tabs" variant="line">
            <TabsTrigger
              value="sources"
              className={tab === "sources" ? "is-active" : ""}
            >
              근거 {citationIds.length > 0 ? citationIds.length : ""}
            </TabsTrigger>
            <TabsTrigger
              value="run"
              className={tab === "run" ? "is-active" : ""}
            >
              실행
            </TabsTrigger>
          </TabsList>

          <TabsContent value="sources" className="panel-scroll">
            {citationIds.length > 0 ? (
              <div className="source-index">
                {citationIds.map((jobId) => (
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    className={jobId === selectedJobId ? "is-active" : ""}
                    key={jobId}
                    onClick={() => onSelectJob(jobId)}
                  >
                    <FileText size={14} />
                    출처 {jobId}
                  </Button>
                ))}
              </div>
            ) : null}

            {loading ? (
              <div className="panel-empty">
                <LoaderCircle size={21} className="spin" />
                <span>공고 근거를 불러오는 중</span>
              </div>
            ) : error ? (
              <div className="panel-error">
                <FileText size={20} />
                <span>{error}</span>
              </div>
            ) : job ? (
              <article className="job-evidence">
                <div className="job-heading">
                  <span className="source-domain">
                    {job.source_platform || hostLabel(job.url)}
                  </span>
                  <h2>{job.position || "직무명 없음"}</h2>
                  <p>
                    <Building2 size={15} />
                    {job.company_name || "회사명 없음"}
                  </p>
                </div>

                <dl className="job-metadata">
                  <div>
                    <dt>
                      <CalendarDays size={14} />
                      게시일
                    </dt>
                    <dd>{job.posted_at_text || job.posted_at || "-"}</dd>
                  </div>
                  <div>
                    <dt>
                      <Clock3 size={14} />
                      수집 시각
                    </dt>
                    <dd>{formatDate(job.collected_at)}</dd>
                  </div>
                  <div>
                    <dt>
                      <Hash size={14} />
                      근거 해시
                    </dt>
                    <dd className="hash-value">
                      {job.evidence_hash
                        ? job.evidence_hash.slice(0, 12)
                        : "-"}
                    </dd>
                  </div>
                  <div>
                    <dt>
                      <Link2 size={14} />
                      원본
                    </dt>
                    <dd>
                      <a href={job.url} target="_blank" rel="noreferrer">
                        {hostLabel(job.url)}
                        <ExternalLink size={13} />
                      </a>
                    </dd>
                  </div>
                </dl>

                <section className="evidence-text">
                  <h3>
                    <FileText size={15} />
                    수집 원문
                  </h3>
                  <pre>{job.raw_text || "저장된 원문이 없습니다."}</pre>
                </section>
              </article>
            ) : (
              <div className="panel-empty">
                <MapPin size={21} />
                <span>
                  답변의 출처를 선택하면
                  <br />
                  저장된 공고 근거가 표시됩니다.
                </span>
              </div>
            )}
          </TabsContent>

          <TabsContent value="run" className="panel-scroll run-panel">
            {metrics ? (
              <>
                <div className="metrics-grid">
                  {metricValue(
                    "실행시간",
                    formatDuration(metrics.duration_sec),
                    <Gauge size={16} />,
                  )}
                  {metricValue(
                    "LLM 토큰",
                    formatNumber(totalTokens),
                    <Database size={16} />,
                  )}
                  {metricValue(
                    "LLM 호출",
                    formatNumber(
                      metrics.llm?.call_count ??
                        metrics.llm?.calls?.length ??
                        0,
                    ),
                    <Activity size={16} />,
                  )}
                  {metricValue(
                    "예상 비용",
                    typeof estimatedCost === "number"
                      ? `$${estimatedCost.toFixed(4)}`
                      : "가격 미등록",
                    <Coins size={16} />,
                  )}
                </div>

                <section className="run-section">
                  <h3>단계별 실행</h3>
                  {steps.length > 0 ? (
                    <div className="step-list">
                      {steps.slice(-16).map((step, index) => {
                        const phase = String(step.phase || step.stage || "");
                        return (
                          <div
                            className="step-row"
                            key={`${phase}-${index}`}
                          >
                            <div>
                              <strong>
                                {PHASE_LABELS[
                                  phase as keyof typeof PHASE_LABELS
                                ] ||
                                  step.component ||
                                  "실행 단계"}
                              </strong>
                              {step.action_source ? (
                                <span>{step.action_source}</span>
                              ) : null}
                            </div>
                            <time>
                              {formatDuration(step.duration_sec)}
                            </time>
                          </div>
                        );
                      })}
                    </div>
                  ) : (
                    <p className="muted-text">저장된 단계 지표가 없습니다.</p>
                  )}
                </section>
              </>
            ) : latestEvents.length > 0 ? (
              <section className="run-section">
                <h3>현재 진행</h3>
                <div className="event-list">
                  {latestEvents.map((event) => (
                    <div
                      className="event-row"
                      key={`${event.phase}-${event.timestamp}`}
                    >
                      <i aria-hidden="true" />
                      <div>
                        <strong>{PHASE_LABELS[event.phase]}</strong>
                        <span>{event.message}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            ) : (
              <div className="panel-empty">
                <Activity size={21} />
                <span>조사를 실행하면 단계별 지표가 표시됩니다.</span>
              </div>
            )}
          </TabsContent>
        </Tabs>
      </aside>
    </>
  );
}
