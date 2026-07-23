import { Check, CornerDownLeft } from "lucide-react";
import { useMemo, useState } from "react";

import type {
  ClarificationAnswer,
  ClarificationQuestion,
} from "../types";

interface ClarificationPanelProps {
  clarification: ClarificationQuestion;
  disabled: boolean;
  onSubmit: (answer: ClarificationAnswer, label: string) => void;
}

export function ClarificationPanel({
  clarification,
  disabled,
  onSubmit,
}: ClarificationPanelProps) {
  const initialOption = clarification.options[0]?.option_id || "";
  const [selected, setSelected] = useState(initialOption);
  const [customValue, setCustomValue] = useState("");
  const selectedOption = useMemo(
    () =>
      clarification.options.find((option) => option.option_id === selected),
    [clarification.options, selected],
  );

  const submit = () => {
    if (selected === "__custom__") {
      const value = customValue.trim();
      if (!value) {
        return;
      }
      onSubmit(
        {
          question_id: clarification.question_id,
          custom_value: value,
        },
        value,
      );
      return;
    }
    if (!selectedOption) {
      return;
    }
    onSubmit(
      {
        question_id: clarification.question_id,
        selected_option_id: selectedOption.option_id,
      },
      selectedOption.label,
    );
  };

  return (
    <div className="clarification-panel">
      <p className="clarification-question">{clarification.question}</p>
      {clarification.reason ? (
        <p className="clarification-reason">{clarification.reason}</p>
      ) : null}
      <div className="clarification-options">
        {clarification.options.map((option) => (
          <label
            className={`clarification-option ${
              selected === option.option_id ? "is-selected" : ""
            }`}
            key={option.option_id}
          >
            <input
              type="radio"
              name={clarification.question_id}
              value={option.option_id}
              checked={selected === option.option_id}
              disabled={disabled}
              onChange={() => setSelected(option.option_id)}
            />
            <span className="option-indicator">
              {selected === option.option_id ? <Check size={13} /> : null}
            </span>
            <span className="option-copy">
              <strong>{option.label}</strong>
              {option.description ? <span>{option.description}</span> : null}
            </span>
            {typeof option.matching_count === "number" &&
            option.matching_count > 0 ? (
              <small>{option.matching_count}건</small>
            ) : null}
          </label>
        ))}
        {clarification.allow_custom ? (
          <label
            className={`clarification-option custom-option ${
              selected === "__custom__" ? "is-selected" : ""
            }`}
          >
            <input
              type="radio"
              name={clarification.question_id}
              value="__custom__"
              checked={selected === "__custom__"}
              disabled={disabled}
              onChange={() => setSelected("__custom__")}
            />
            <span className="option-indicator">
              {selected === "__custom__" ? <Check size={13} /> : null}
            </span>
            <input
              type="text"
              value={customValue}
              disabled={disabled}
              placeholder="직접 입력"
              aria-label="직접 답변 입력"
              onFocus={() => setSelected("__custom__")}
              onChange={(event) => setCustomValue(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  event.preventDefault();
                  submit();
                }
              }}
            />
          </label>
        ) : null}
      </div>
      <button
        type="button"
        className="clarification-submit"
        disabled={
          disabled ||
          !selected ||
          (selected === "__custom__" && !customValue.trim())
        }
        onClick={submit}
      >
        선택
        <CornerDownLeft size={15} />
      </button>
    </div>
  );
}
