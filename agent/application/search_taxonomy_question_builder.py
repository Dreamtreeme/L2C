"""검색 의미 사전의 직무 범위를 객관식 질문으로 만든다."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, Protocol

from agent.application.search_taxonomy_utils import OCCUPATION_DOMAIN_ROOT_KEY
from shared.schema.investigation_schema import (
    ClarificationOption,
    ClarificationQuestion,
    InvestigationConstraints,
)


class TaxonomyQuestionSource(Protocol):
    """질문 생성에 필요한 검색 의미 사전 조회 계약."""

    def list_direct_children(self, concept_key: str) -> list[dict[str, str]]: ...

    def matching_occupation_job_ids(
        self,
        concept_keys: Iterable[str],
        constraints: InvestigationConstraints | None = None,
    ) -> set[int]: ...

    def occupation_descendant_count(self, concept_key: str) -> int: ...

    def occupation_resolution_candidates(
        self,
        domain_concept_keys: Iterable[str],
    ) -> list[dict[str, Any]]: ...

    def concept_label(self, concept_key: str) -> str: ...


class TaxonomyQuestionBuilder:
    """현재 확정된 직무 범위에 맞는 다음 질문 하나를 만든다."""

    def __init__(self, source: TaxonomyQuestionSource):
        self.source = source

    @staticmethod
    def _concept_option_id(concept_key: str) -> str:
        digest = hashlib.sha1(concept_key.encode("utf-8")).hexdigest()[:10]
        return f"concept-{digest}"

    def _counted_scope_item(
        self,
        item: dict[str, str],
        constraints: InvestigationConstraints,
    ) -> dict[str, Any]:
        concept_key = str(item["concept_key"])
        return {
            **item,
            "matching_count": len(
                self.source.matching_occupation_job_ids(
                    [concept_key],
                    constraints,
                )
            ),
            "concept_count": self.source.occupation_descendant_count(concept_key),
        }

    def build_domain_question(
        self,
        constraints: InvestigationConstraints,
        *,
        answered_question_ids: Iterable[str] = (),
    ) -> ClarificationQuestion | None:
        """일반 공고 요청에 업무 기능 기준 최상위 영역을 제시한다."""

        question_id = "occupation_domain"
        if question_id in set(answered_question_ids):
            return None
        domains = [
            item
            for item in self.source.list_direct_children(
                OCCUPATION_DOMAIN_ROOT_KEY
            )
            if item["concept_type"] == "domain"
        ]
        if not domains:
            return None
        counted = [
            self._counted_scope_item(item, constraints)
            for item in domains
        ]
        options = [
            ClarificationOption(
                option_id=self._concept_option_id(str(item["concept_key"])),
                label=str(item["label"]),
                value=str(item["concept_key"]),
                collection_search_term=str(item["label"]),
                matching_count=int(item["matching_count"]),
                concept_count=int(item["concept_count"]),
                description=(
                    f"저장 공고 {item['matching_count']}건 · "
                    f"사전 직무 {item['concept_count']}개"
                ),
            )
            for item in counted
        ]
        return ClarificationQuestion(
            question_id=question_id,
            field="occupation_domain_concept_keys",
            question="어떤 업무 영역의 채용공고를 찾을까요?",
            options=options,
            allow_custom=True,
            reason=(
                "회사의 업종이 아니라 실제 수행할 업무를 기준으로 선택합니다. "
                "원하는 직무가 명확하면 직접 입력할 수 있습니다."
            ),
            candidate_count=len(
                self.source.matching_occupation_job_ids(
                    [OCCUPATION_DOMAIN_ROOT_KEY],
                    constraints,
                )
            ),
            concept_count=self.source.occupation_descendant_count(
                OCCUPATION_DOMAIN_ROOT_KEY
            ),
            facet_type="occupation_domain",
        )

    def _family_children(self, domain_key: str) -> list[dict[str, str]]:
        children = [
            item
            for item in self.source.list_direct_children(domain_key)
            if item["concept_type"] == "occupation"
        ]
        if len(children) != 1:
            return children
        nested = [
            item
            for item in self.source.list_direct_children(
                str(children[0]["concept_key"])
            )
            if item["concept_type"] == "occupation"
        ]
        return [children[0], *nested] if len(nested) >= 2 else children

    def build_family_question(
        self,
        constraints: InvestigationConstraints,
        *,
        answered_question_ids: Iterable[str] = (),
    ) -> ClarificationQuestion | None:
        """선택된 업무 영역 아래의 검토된 직무군을 모두 제시한다."""

        domain_keys = list(constraints.occupation_domain_concept_keys)
        if not domain_keys:
            return None
        fingerprint = hashlib.sha1(
            "|".join(sorted(domain_keys)).encode("utf-8")
        ).hexdigest()[:10]
        question_id = f"occupation_family:{fingerprint}"
        if question_id in set(answered_question_ids):
            return None
        families: dict[str, dict[str, str]] = {}
        for domain_key in domain_keys:
            for item in self._family_children(domain_key):
                families[str(item["concept_key"])] = item
        if not families:
            return None
        counted = [
            self._counted_scope_item(item, constraints)
            for item in families.values()
        ]
        counted.sort(
            key=lambda item: (
                -int(item["matching_count"]),
                str(item["label"]),
            )
        )
        options = [
            ClarificationOption(
                option_id=self._concept_option_id(str(item["concept_key"])),
                label=str(item["label"]),
                value=str(item["concept_key"]),
                collection_search_term=str(item["label"]),
                matching_count=int(item["matching_count"]),
                concept_count=int(item["concept_count"]),
                description=(
                    f"저장 공고 {item['matching_count']}건 · "
                    f"사전 직무 {item['concept_count']}개"
                ),
            )
            for item in counted
        ]
        domain_labels = ", ".join(
            self.source.concept_label(key)
            for key in domain_keys
        )
        return ClarificationQuestion(
            question_id=question_id,
            field="occupation_concept_keys",
            question=f"{domain_labels} 중 어떤 직무군을 찾을까요?",
            options=options,
            allow_custom=True,
            reason=(
                "상위 직무군은 모든 하위 직무를 포함합니다. 더 좁은 직무군을 "
                "선택하거나 원하는 직무명을 직접 입력할 수 있습니다."
            ),
            candidate_count=len(
                self.source.matching_occupation_job_ids(
                    domain_keys,
                    constraints,
                )
            ),
            concept_count=len(
                self.source.occupation_resolution_candidates(domain_keys)
            ),
            facet_type="occupation_family",
        )

    def build_next_scope_question(
        self,
        constraints: InvestigationConstraints,
        *,
        answered_question_ids: Iterable[str] = (),
    ) -> ClarificationQuestion | None:
        """현재 확정 수준에 맞는 다음 직무 범위 질문 하나를 만든다."""

        answered = tuple(answered_question_ids)
        if not constraints.occupation_concept_keys:
            if (
                not constraints.occupation_domain_concept_keys
                and constraints.occupation_scope_required
                and not constraints.occupation_query
            ):
                return self.build_domain_question(
                    constraints,
                    answered_question_ids=answered,
                )
            if (
                constraints.occupation_domain_concept_keys
                and not constraints.occupation_query
            ):
                if constraints.occupation_scope_mode == "all":
                    return None
                return self.build_family_question(
                    constraints,
                    answered_question_ids=answered,
                )
            return None
        return self.build_scope_question(
            constraints,
            answered_question_ids=answered,
        )

    def build_scope_question(
        self,
        constraints: InvestigationConstraints,
        *,
        answered_question_ids: Iterable[str] = (),
    ) -> ClarificationQuestion | None:
        """실제 공고가 있는 하위 직무만 카디널리티와 함께 제시한다."""

        if constraints.occupation_scope_mode == "all":
            return None
        answered = set(answered_question_ids)
        for concept_key in constraints.occupation_concept_keys:
            question_id = f"occupation_scope:{concept_key}"
            if question_id in answered:
                continue
            children = self.source.list_direct_children(concept_key)
            counted = [
                {
                    **child,
                    "count": len(
                        self.source.matching_occupation_job_ids(
                            [child["concept_key"]],
                            constraints,
                        )
                    ),
                }
                for child in children
            ]
            counted = [item for item in counted if item["count"] > 0]
            if len(counted) < 2:
                continue
            counted.sort(
                key=lambda item: (
                    -int(item["count"]),
                    str(item["label"]),
                )
            )
            total_count = len(
                self.source.matching_occupation_job_ids(
                    [concept_key],
                    constraints,
                )
            )
            options = [
                ClarificationOption(
                    option_id=self._concept_option_id(
                        str(item["concept_key"])
                    ),
                    label=f"{item['label']} ({item['count']}건)",
                    value=str(item["concept_key"]),
                    collection_search_term=str(item["label"]),
                    matching_count=int(item["count"]),
                    concept_count=self.source.occupation_descendant_count(
                        str(item["concept_key"])
                    ),
                    description=(
                        f"현재 조건에서 이 직무로 연결된 공고 "
                        f"{item['count']}건"
                    ),
                )
                for item in counted
            ]
            options.append(
                ClarificationOption(
                    option_id="all-descendants",
                    label=f"전체 범위 ({total_count}건)",
                    value=concept_key,
                    matching_count=total_count,
                    concept_count=self.source.occupation_descendant_count(
                        concept_key
                    ),
                    description="현재 검색어를 유지하고 모든 하위 직무를 포함",
                )
            )
            return ClarificationQuestion(
                question_id=question_id,
                field="occupation_concept_keys",
                question=(
                    f"{self.source.concept_label(concept_key)} 공고 "
                    f"{total_count}건 중 어떤 범위로 좁힐까요?"
                ),
                options=options,
                allow_custom=False,
                reason=(
                    "각 수치는 해당 필터를 적용했을 때의 결과 수이며, "
                    "복합 직무 공고는 여러 선택지에 포함될 수 있습니다."
                ),
                candidate_count=total_count,
                concept_count=self.source.occupation_descendant_count(
                    concept_key
                ),
                facet_type="occupation",
            )
        return None


__all__ = ["TaxonomyQuestionBuilder", "TaxonomyQuestionSource"]
