"""Rule-based Russian text presentation derived only from structured facts."""

from __future__ import annotations

from collections import Counter, defaultdict

from ..contracts import EventKind, StreamAnalysisResult


_EVENT_KIND_ORDER: tuple[EventKind, ...] = (
    EventKind.APPEARED,
    EventKind.DISAPPEARED,
    EventKind.COUNT_CHANGED,
    EventKind.POSITION_CHANGED,
    EventKind.PERSISTED,
)

_EVENT_GROUP_LABELS: dict[EventKind, str] = {
    EventKind.APPEARED: "Появились в следующем кадре",
    EventKind.DISAPPEARED: "Исчезли в следующем кадре",
    EventKind.COUNT_CHANGED: "Изменилось количество",
    EventKind.POSITION_CHANGED: "Изменилось положение",
    EventKind.PERSISTED: "Продолжили присутствовать",
}

_EVENT_COUNT_LABELS: dict[EventKind, str] = {
    EventKind.APPEARED: "появились в следующем кадре",
    EventKind.DISAPPEARED: "исчезли в следующем кадре",
    EventKind.COUNT_CHANGED: "изменение количества",
    EventKind.POSITION_CHANGED: "изменение положения",
    EventKind.PERSISTED: "продолжили присутствовать без отдельного изменения",
}

_RUN_STATUS_LABELS: dict[str, str] = {
    "completed": "анализ завершён",
    "completed_with_warnings": "анализ завершён с предупреждениями",
    "partial": "получен частичный результат",
    "failed": "анализ завершился ошибкой",
}

_EVENT_STATUS_LABELS: dict[str, str] = {
    "certain": "уверенно",
    "uncertain": "неопределённо",
}


def _event_status(value: str) -> str:
    return _EVENT_STATUS_LABELS.get(value, value)


def _event_line(event) -> str:
    before = event.evidence.from_count
    after = event.evidence.to_count
    status = _event_status(event.status.value)

    if event.kind is EventKind.APPEARED:
        return f"{event.predicted_type_id}: было {before}, стало {after}; статус: {status}."
    if event.kind is EventKind.DISAPPEARED:
        return f"{event.predicted_type_id}: было {before}, стало {after}; статус: {status}."
    if event.kind is EventKind.COUNT_CHANGED:
        return (
            f"{event.predicted_type_id}: количество изменилось с {before} до {after}; "
            f"статус: {status}."
        )
    if event.kind is EventKind.POSITION_CHANGED:
        shift = (
            "не указан"
            if event.evidence.position_shift_norm is None
            else str(event.evidence.position_shift_norm)
        )
        return (
            f"{event.predicted_type_id}: количество {before} → {after}, "
            f"нормализованный сдвиг: {shift}; статус: {status}."
        )
    return f"{event.predicted_type_id}: {before} → {after}; статус: {status}."


def _plural_type(count: int) -> str:
    if count % 10 == 1 and count % 100 != 11:
        return "тип"
    if count % 10 in (2, 3, 4) and count % 100 not in (12, 13, 14):
        return "типа"
    return "типов"


def _display_events_for_transition(events: tuple | list) -> tuple:
    """Hide redundant persisted rows when the same type has a stronger event."""

    changed_type_ids = {
        event.predicted_type_id
        for event in events
        if event.kind is not EventKind.PERSISTED
    }
    return tuple(
        event
        for event in events
        if not (
            event.kind is EventKind.PERSISTED
            and event.predicted_type_id in changed_type_ids
        )
    )


def build_text_report(result: StreamAnalysisResult) -> str:
    """Render deterministic human-readable facts without inference or GT data."""

    if not isinstance(result, StreamAnalysisResult):
        raise TypeError("result must be StreamAnalysisResult.")

    events_by_transition: dict[str, list] = defaultdict(list)
    for event in result.change_events:
        events_by_transition[event.comparison_id].append(event)
    display_events_by_transition = {
        comparison_id: _display_events_for_transition(events)
        for comparison_id, events in events_by_transition.items()
    }
    display_event_counts = Counter(
        event.kind.value
        for events in display_events_by_transition.values()
        for event in events
    )
    lines = [
        "Отчёт по анализу потока изображений",
        f"Запуск: {result.run_id}",
        f"Итог: {_RUN_STATUS_LABELS.get(result.run_status.value, result.run_status.value)}.",
        "",
        "Краткая сводка",
        f"- Обработано кадров: {len(result.frame_ids)}.",
        f"- Найдено областей-кандидатов: {len(result.candidate_record_ids)}.",
        (
            f"- Выделено повторяющихся визуальных типов: "
            f"{len(result.recurring_type_ids)}."
        ),
        f"- Сравнено соседних пар кадров: {len(result.frame_comparisons)}.",
        f"- Сформировано событий изменений: {len(result.change_events)}.",
        (
            "- В текстовом отчёте ниже повторное присутствие не дублируется, "
            "если для того же type_id в переходе уже указано появление, "
            "исчезновение, изменение количества или изменение положения."
        ),
        "",
        "События по всему потоку",
    ]
    for kind in _EVENT_KIND_ORDER:
        count = display_event_counts[kind.value]
        lines.append(f"- {_EVENT_COUNT_LABELS[kind]}: {count}.")

    if result.frame_comparisons:
        lines.extend(("", "Переходы между кадрами"))
        for comparison in result.frame_comparisons:
            pair = comparison.frame_pair
            lines.extend(("", f"{pair.from_frame_id} → {pair.to_frame_id}"))
            transition_events = display_events_by_transition.get(comparison.comparison_id, ())
            wrote_group = False
            for kind in _EVENT_KIND_ORDER:
                grouped = sorted(
                    (event for event in transition_events if event.kind is kind),
                    key=lambda item: (item.predicted_type_id, item.event_id),
                )
                if not grouped:
                    continue
                if wrote_group:
                    lines.append("")
                lines.append(f"  {_EVENT_GROUP_LABELS[kind]}:")
                for event in grouped:
                    lines.append(f"  - {_event_line(event)}")
                wrote_group = True
            if not wrote_group:
                if comparison.emission_status.value == "withheld":
                    diagnostics = ", ".join(comparison.withheld_diagnostic_ids) or "нет ссылок"
                    lines.append(
                        "  События для этого перехода не сформированы; "
                        f"диагностика: {diagnostics}."
                    )
                else:
                    lines.append("  По этому переходу не сформировано событий изменений.")

    warning_count = int(result.status_summary.get("warning_count", 0))
    error_count = int(result.status_summary.get("error_count", 0))
    withheld_count = sum(
        comparison.emission_status.value == "withheld"
        for comparison in result.frame_comparisons
    )
    lines.extend(
        (
            "",
            "Техническая диагностика",
            f"- Предупреждения: {warning_count}.",
            f"- Ошибки: {error_count}.",
            f"- Переходы без сформированных событий: {withheld_count}.",
        )
    )
    withheld = [
        comparison for comparison in result.frame_comparisons
        if comparison.emission_status.value == "withheld"
    ]
    for comparison in withheld:
        diagnostics = ", ".join(comparison.withheld_diagnostic_ids) or "нет ссылок"
        lines.append(
            f"- {comparison.frame_pair.from_frame_id} → "
            f"{comparison.frame_pair.to_frame_id}: события не сформированы; "
            f"диагностика: {diagnostics}."
        )

    if warning_count or error_count or withheld_count:
        lines.append(
            "Подробные структурированные предупреждения и ошибки находятся в "
            "run_manifest.json и stream_analysis.json."
        )

    lines.extend(
        (
            "",
            "Как читать этот отчёт",
            (
                "- type_001, type_002 и похожие обозначения — это внутренние "
                "идентификаторы повторяющихся визуальных типов, а не названия "
                "реальных объектов."
            ),
            (
                "- Отчёт описывает только результат анализа изображений. Он не "
                "использует разметку (ground truth) и не содержит метрики оценки "
                "качества."
            ),
            (
                f"- Всего в потоке найдено {len(result.recurring_type_ids)} "
                f"{_plural_type(len(result.recurring_type_ids))} таких визуальных "
                "повторений."
            ),
        )
    )
    return "\n".join(lines) + "\n"


__all__ = ["build_text_report"]
