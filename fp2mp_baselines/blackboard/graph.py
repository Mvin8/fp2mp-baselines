from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from ..graph_utils import build_llm_log_entry, message_content_to_text, message_to_log_dict
from ..state import BlackboardState
from .board import BaseNote, Board, get_id
from .prompts import (
    CLEANER_PROMPT,
    CONTROLLER_PROMPT,
    CRITIC_PROMPT,
    DECIDER_PROMPT,
    PLANNER_PROMPT,
    SUMMARIZER_PROMPT,
    WORKER_PROMPT,
)


BLACKBOARD_NODE = "blackboard"


class ControllerResponse(BaseModel):
    agents_ids: list[str] = Field(min_length=1, description="Упорядоченный список ID агентов")


class CleanerResponse(BaseModel):
    notes_ids: list[str] = Field(default_factory=list, description="Список ID записей к удалению")


class DeciderResponse(BaseModel):
    note: BaseNote = Field(description="Запись для добавления на доску")
    is_final: bool = Field(default=False, description="Сигнал о завершении процесса работы над задачей")


class Worker(BaseModel):
    id: str = Field(default_factory=get_id)
    role_name: str
    role_description: str
    prompt: str
    response_format: type[BaseModel] | None = None


def _worker_prompt(worker: Worker, question: str) -> str:
    return (WORKER_PROMPT + "\n" + worker.prompt).format(
        id=worker.id,
        role_name=worker.role_name,
        role_description=worker.role_description,
        question=question,
    )


def _invoke_llm(
    llm: BaseChatModel,
    *,
    agent_id: str,
    prompt: str,
    board: Board,
    response_format: type[BaseModel] | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    messages = [
        HumanMessage(content=prompt),
        HumanMessage(content=board.to_str()),
    ]
    model = llm.with_structured_output(response_format) if response_format is not None else llm
    response = model.invoke(messages)
    if response_format is not None:
        result = response
        log_entry = {
            "agent": agent_id,
            "event": "llm_call",
            "messages": [message_to_log_dict(message) for message in messages],
            "response": {
                "type": "structured",
                "data": response.model_dump(),
            },
            "metadata": metadata or {},
        }
    else:
        result = message_content_to_text(response.content)
        log_entry = {
            **build_llm_log_entry(agent_id, messages, response),
            "metadata": metadata or {},
        }
    return result, log_entry


def _run_blackboard(
    question: str,
    llm: BaseChatModel,
    *,
    iterations: int,
) -> BlackboardState:
    board = Board()
    log: list[dict[str, Any]] = []

    planner = Worker(
        prompt=PLANNER_PROMPT,
        role_name="Планировщик",
        role_description="Разрабатывает пошаговый план решения задачи на основе содержимого доски",
        response_format=BaseNote,
    )
    critic = Worker(
        prompt=CRITIC_PROMPT,
        role_name="Критик",
        role_description="Выявляет ошибочные или вводящие в заблуждение записи на доске",
        response_format=BaseNote,
    )
    cleaner = Worker(
        prompt=CLEANER_PROMPT,
        role_name="Уборщик",
        role_description="Анализирует доску, выявляет и удаляет бесполезные или избыточные записи",
        response_format=CleanerResponse,
    )
    decider = Worker(
        prompt=DECIDER_PROMPT,
        role_name="Арбитр",
        role_description="Оценивает полноту информации. Останавливает либо инициирует продолжение обсуждения",
        response_format=DeciderResponse,
    )

    workers = {worker.id: worker for worker in [planner, critic, cleaner, decider]}
    controller_id = get_id()
    controller_metadata = {
        "id": controller_id,
        "role_name": "Контроллер",
        "workers": [
            {
                "id": worker.id,
                "role_name": worker.role_name,
                "role_description": worker.role_description,
            }
            for worker in workers.values()
        ],
        "question": question,
    }
    controller_prompt = CONTROLLER_PROMPT.format(
        workers=controller_metadata["workers"],
        question=question,
    )

    is_final = False
    for _ in range(iterations):
        controller_response, controller_log = _invoke_llm(
            llm,
            agent_id=controller_id,
            prompt=controller_prompt,
            board=board,
            response_format=ControllerResponse,
            metadata=controller_metadata,
        )
        log.append(controller_log)

        for worker_id in controller_response.agents_ids:
            worker = workers.get(worker_id)
            if worker is None:
                continue

            worker_metadata = {
                "id": worker.id,
                "role_name": worker.role_name,
                "role_description": worker.role_description,
                "question": question,
            }
            response, worker_log = _invoke_llm(
                llm,
                agent_id=worker.id,
                prompt=_worker_prompt(worker, question),
                board=board,
                response_format=worker.response_format,
                metadata=worker_metadata,
            )
            log.append(worker_log)

            if worker == cleaner:
                board.remove_notes(response.notes_ids)
                continue

            if worker == decider:
                board.add_note(response.note, worker.id)
                is_final = response.is_final
            else:
                board.add_note(response, worker.id)

            if is_final:
                break

        if is_final:
            break

    summarizer_id = get_id()
    output, summarizer_log = _invoke_llm(
        llm,
        agent_id=summarizer_id,
        prompt=SUMMARIZER_PROMPT.format(question=question),
        board=board,
        metadata={
            "id": summarizer_id,
            "role_name": "Суммаризатор",
            "question": question,
        },
    )
    log.append(summarizer_log)

    return {
        "input": question,
        "output": str(output),
        "board": board.to_str(),
        "notes": [note.model_dump() for note in board.notes],
        "is_final": is_final,
        "log": log,
    }


def build_blackboard_graph(
    llm: BaseChatModel,
    *,
    iterations: int = 3,
) -> CompiledStateGraph:
    """Build a Blackboard LangGraph baseline."""

    def blackboard_node(state: BlackboardState) -> BlackboardState:
        return _run_blackboard(
            state["input"],
            llm,
            iterations=iterations,
        )

    graph = StateGraph(BlackboardState)
    graph.add_node(BLACKBOARD_NODE, blackboard_node)
    graph.add_edge(START, BLACKBOARD_NODE)
    graph.add_edge(BLACKBOARD_NODE, END)
    return graph.compile()


__all__ = ["build_blackboard_graph"]
