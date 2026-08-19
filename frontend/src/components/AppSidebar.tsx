import {
  Activity,
  PanelLeftClose,
  Plus,
  ScanSearch,
} from "lucide-react";

import { Button } from "@/components/ui/button";

interface AppSidebarProps {
  open: boolean;
  connectionState: "checking" | "online" | "offline";
  disabled: boolean;
  onClose: () => void;
  onNewConversation: () => void;
  onOpenOperations: () => void;
}

export function AppSidebar({
  open,
  connectionState,
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
          <Button
            type="button"
            variant="ghost"
            size="icon-lg"
            className="icon-button sidebar-close"
            title="메뉴 닫기"
            aria-label="메뉴 닫기"
            onClick={onClose}
          >
            <PanelLeftClose size={18} />
          </Button>
        </div>

        <Button
          type="button"
          size="lg"
          className="new-investigation-button"
          disabled={disabled}
          onClick={onNewConversation}
        >
          <Plus size={17} />
          새 조사
        </Button>

        <nav className="sidebar-nav" aria-label="주요 메뉴">
          <Button
            type="button"
            variant="ghost"
            size="lg"
            className="sidebar-nav-item w-full justify-start"
            onClick={onOpenOperations}
          >
            <Activity size={16} />
            실행 현황
          </Button>
        </nav>

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
