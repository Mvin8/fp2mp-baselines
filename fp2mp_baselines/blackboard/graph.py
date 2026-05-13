from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from ..graph_utils import build_message_log, message_content_to_text
from ..state import BlackboardState
from .board import BaseNote, Board, get_id
from .prompts import (
    CLEANER_PROMPT,
    CONTROLLER_PROMPT,
    CRITIC_PROMPT,
    DECIDER_PROMPT,
    EXPERT_PROMPT,
    GENERATOR_PROMPT,
    PLANNER_PROMPT,
    SUMMARIZER_PROMPT,
    WORKER_PROMPT,
)


BLACKBOARD_NODE = "blackboard"


class ControllerResponse(BaseModel):
    agents_ids: list[str] = Field(min_length=1, description="Ordered list of agent IDs")


class CleanerResponse(BaseModel):
    notes_ids: list[str] = Field(default_factory=list, description="List of note IDs to remove")


class GeneratorRole(BaseModel):
    name: str = Field(description="Expert role name")
    description: str = Field(description="Short description of the role's expertise")


class GeneratorResponse(BaseModel):
    roles: list[GeneratorRole] = Field(
        min_length=1,
        max_length=3,
        description="List of expert roles for solving the task",
    )


class DeciderResponse(BaseModel):
    note: BaseNote = Field(description="Note to add to the blackboard")
    is_final: bool = Field(default=False, description="Signal that the task-solving process is complete")


class Worker(BaseModel):
    id: str = Field(default_factory=get_id)
    role_type: str
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
    prompt: str,
    board: Board,
    response_format: type[BaseModel] | None = None,
) -> tuple[Any, list[BaseMessage]]:
    messages = [
        HumanMessage(content=prompt),
        *board.to_messages(),
    ]
    model = llm.with_structured_output(response_format) if response_format is not None else llm
    response = model.invoke(messages)
    if response_format is not None:
        result = response
        log = messages
    else:
        result = message_content_to_text(response.content)
        log = build_message_log(messages, response)
    return result, log


def _build_builtin_workers() -> list[Worker]:
    return [
        Worker(
            role_type="planner",
            prompt=PLANNER_PROMPT,
            role_name="Planner",
            role_description="Develops a step-by-step plan for solving the task based on the blackboard contents",
            response_format=BaseNote,
        ),
        Worker(
            role_type="critic",
            prompt=CRITIC_PROMPT,
            role_name="Critic",
            role_description="Identifies incorrect or misleading notes on the blackboard",
            response_format=BaseNote,
        ),
        Worker(
            role_type="cleaner",
            prompt=CLEANER_PROMPT,
            role_name="Cleaner",
            role_description="Analyzes the blackboard, identifies useless or redundant notes, and removes them",
            response_format=CleanerResponse,
        ),
        Worker(
            role_type="decider",
            prompt=DECIDER_PROMPT,
            role_name="Arbiter",
            role_description="Evaluates information completeness and either stops or continues the discussion",
            response_format=DeciderResponse,
        ),
    ]


def _build_expert_workers(question: str, llm: BaseChatModel, board: Board) -> tuple[list[Worker], list[BaseMessage]]:
    response, log = _invoke_llm(
        llm,
        prompt=GENERATOR_PROMPT.format(question=question),
        board=board,
        response_format=GeneratorResponse,
    )
    workers = [
        Worker(
            role_type="expert",
            role_name=role.name,
            role_description=role.description,
            prompt=EXPERT_PROMPT,
            response_format=BaseNote,
        )
        for role in response.roles
    ]
    return workers, log


def _format_workers(workers: dict[str, Worker]) -> str:
    return "\n".join(
        (
            f"- ID: {worker.id}; role: {worker.role_name}; "
            f"type: {worker.role_type}; description: {worker.role_description}"
        )
        for worker in workers.values()
    )


def _run_blackboard(
    question: str,
    llm: BaseChatModel,
    *,
    iterations: int,
) -> BlackboardState:
    board = Board()
    log: list[BaseMessage] = []

    expert_workers, generator_log = _build_expert_workers(question, llm, board)
    log.extend(generator_log)
    all_workers = [*_build_builtin_workers(), *expert_workers]
    workers = {worker.id: worker for worker in all_workers}
    controller_prompt = CONTROLLER_PROMPT.format(workers=_format_workers(workers), question=question)

    is_final = False
    for _ in range(iterations):
        controller_response, controller_log = _invoke_llm(
            llm,
            prompt=controller_prompt,
            board=board,
            response_format=ControllerResponse,
        )
        log.extend(controller_log)

        for worker_id in controller_response.agents_ids:
            worker = workers.get(worker_id)
            if worker is None:
                continue

            response, worker_log = _invoke_llm(
                llm,
                prompt=_worker_prompt(worker, question),
                board=board,
                response_format=worker.response_format,
            )
            log.extend(worker_log)

            if worker.role_type == "cleaner":
                board.remove_notes(response.notes_ids)
                continue

            if worker.role_type == "decider":
                board.add_note(response.note, worker.id, worker.role_name)
                is_final = response.is_final
            else:
                board.add_note(response, worker.id, worker.role_name)

            if is_final:
                break

        if is_final:
            break

    output, summarizer_log = _invoke_llm(
        llm,
        prompt=SUMMARIZER_PROMPT.format(question=question),
        board=board,
    )
    log.extend(summarizer_log)

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
