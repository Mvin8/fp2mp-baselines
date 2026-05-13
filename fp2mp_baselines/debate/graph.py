from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ..graph_utils import build_message_log, message_content_to_text
from ..state import DebateState
from ..voting import aggregate_majority_vote, format_agent_responses
from .prompts import DEBATE_ROUND_PROMPT


DEBATE_NODE = "debate"


def _build_peer_responses(responses: list[str], agent_index: int) -> str:
    peers = [response for index, response in enumerate(responses) if index != agent_index]
    return format_agent_responses(peers)


def build_debate_graph(
    llm: BaseChatModel,
    *,
    num_agents: int = 5,
    debate_rounds: int = 5,
    vote_llm: BaseChatModel | None = None,
) -> CompiledStateGraph:
    """Build a Multi-Agent Debate graph with simultaneous-talk rounds."""

    if num_agents < 1:
        raise ValueError("num_agents must be at least 1")
    if debate_rounds < 0:
        raise ValueError("debate_rounds must be non-negative")

    voter_model = vote_llm or llm

    def debate_node(state: DebateState) -> DebateState:
        log = list(state.get("log", []))
        responses: list[str] = []

        for _ in range(num_agents):
            messages = [HumanMessage(content=state["input"])]
            response = llm.invoke(messages)
            responses.append(message_content_to_text(response.content))
            log.extend(build_message_log(messages, response))

        rounds = [responses]

        for _ in range(debate_rounds):
            previous_responses = responses
            next_responses: list[str] = []

            for agent_index, own_response in enumerate(previous_responses):
                messages = [
                    HumanMessage(
                        content=DEBATE_ROUND_PROMPT.format(
                            input=state["input"],
                            peer_responses=_build_peer_responses(previous_responses, agent_index),
                            own_response=own_response,
                        )
                    )
                ]
                response = llm.invoke(messages)
                next_responses.append(message_content_to_text(response.content))
                log.extend(build_message_log(messages, response))

            responses = next_responses
            rounds.append(responses)

        vote = aggregate_majority_vote(
            input_text=state["input"],
            responses=responses,
            llm=voter_model,
        )
        log.extend(vote.log)

        return {
            "input": state["input"],
            "output": vote.answer,
            "agent_responses": responses,
            "rounds": rounds,
            "vote_counts": vote.counts,
            "log": log,
        }

    graph = StateGraph(DebateState)
    graph.add_node(DEBATE_NODE, debate_node)
    graph.add_edge(START, DEBATE_NODE)
    graph.add_edge(DEBATE_NODE, END)
    return graph.compile()


__all__ = ["build_debate_graph"]
