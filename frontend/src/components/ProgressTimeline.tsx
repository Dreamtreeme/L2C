import {
  Circle,
  CircleAlert,
  CircleCheck,
  CirclePause,
  CircleX,
  LoaderCircle,
} from "lucide-react";

import { PHASE_LABELS } from "../lib/format";
import type { RunEvent, RunPhase } from "../types";

interface ProgressTimelineProps {
  events: RunEvent[];
  pending: boolean;
}

function latestEventsByPhase(events: RunEvent[]): RunEvent[] {
  const order: RunPhase[] = [];
  const latest = new Map<RunPhase, RunEvent>();
  events.forEach((event) => {
    if (!latest.has(event.phase)) {
      order.push(event.phase);
    }
    latest.set(event.phase, event);
  });
  return order.map((phase) => latest.get(phase)!);
}

function StageIcon({
  event,
  active,
}: {
  event: RunEvent;
  active: boolean;
}) {
  if (event.status === "failed") {
    return <CircleAlert size={17} />;
  }
  if (event.status === "cancelled") {
    return <CircleX size={17} />;
  }
  if (event.status === "waiting_input") {
    return <CirclePause size={17} />;
  }
  if (active) {
    return <LoaderCircle size={17} className="spin" />;
  }
  if (event.status === "completed") {
    return <CircleCheck size={17} />;
  }
  return <Circle size={13} />;
}

export function ProgressTimeline({
  events,
  pending,
}: ProgressTimelineProps) {
  const stages = latestEventsByPhase(events);
  if (stages.length === 0) {
    return (
      <div className="progress-placeholder">
        <LoaderCircle size={15} className="spin" />
        요청을 준비하고 있습니다.
      </div>
    );
  }

  return (
    <div className="progress-timeline" aria-label="조사 진행 단계">
      {stages.map((event, index) => {
        const active =
          pending &&
          index === stages.length - 1 &&
          (event.status === "queued" || event.status === "running");
        return (
          <div
            className={`progress-stage ${
              active ? "is-active" : "is-complete"
            } is-${event.status}`}
            key={`${event.phase}-${event.timestamp}`}
          >
            <span className="progress-icon">
              <StageIcon event={event} active={active} />
            </span>
            <div>
              <strong>{PHASE_LABELS[event.phase]}</strong>
              <span>{event.message || PHASE_LABELS[event.phase]}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}
