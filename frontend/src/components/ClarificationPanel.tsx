import { CornerDownLeft } from "lucide-react";
import { useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  RadioGroup,
  RadioGroupItem,
} from "@/components/ui/radio-group";

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
      <RadioGroup
        className="clarification-options"
        value={selected}
        disabled={disabled}
        onValueChange={setSelected}
      >
        {clarification.options.map((option) => (
          <label
            className={`clarification-option ${
              selected === option.option_id ? "is-selected" : ""
            }`}
            key={option.option_id}
            htmlFor={`${clarification.question_id}-${option.option_id}`}
          >
            <RadioGroupItem
              id={`${clarification.question_id}-${option.option_id}`}
              value={option.option_id}
              className="option-indicator"
            />
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
            htmlFor={`${clarification.question_id}-custom`}
          >
            <RadioGroupItem
              id={`${clarification.question_id}-custom`}
              value="__custom__"
              className="option-indicator"
            />
            <Input
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
      </RadioGroup>
      <Button
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
      </Button>
    </div>
  );
}
