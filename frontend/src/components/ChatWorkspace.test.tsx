import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ChatWorkspace } from "./ChatWorkspace";

describe("ChatWorkspace", () => {
  it("한글 조합 중 Enter 입력은 질문을 전송하지 않는다", () => {
    const onSubmit = vi.fn();
    render(
      <ChatWorkspace
        messages={[]}
        isRunning={false}
        cancelRequested={false}
        activeStatus="준비됨"
        statusError={false}
        activePhase={null}
        pendingResume={null}
        onOpenSidebar={vi.fn()}
        onOpenEvidence={vi.fn()}
        onSelectCitation={vi.fn()}
        onSubmit={onSubmit}
        onSubmitClarification={vi.fn()}
        onCancel={vi.fn()}
      />,
    );

    const input = screen.getByLabelText("질문 입력");
    fireEvent.change(input, { target: { value: "AI 엔지니어" } });
    fireEvent.keyDown(input, { key: "Enter", isComposing: true });
    expect(onSubmit).not.toHaveBeenCalled();

    fireEvent.keyDown(input, { key: "Enter", isComposing: false });
    expect(onSubmit).toHaveBeenCalledWith("AI 엔지니어");
  });
});
