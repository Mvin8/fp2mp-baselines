from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ..graph_utils import build_message_log, message_content_to_text
from ..state import MajorVoteState
from ..voting import aggregate_majority_vote


MAJOR_VOTE_NODE = "major_vote"


def build_major_vote_graph(
    llm: BaseChatModel,
    *,
    num_agents: int = 5,
    vote_llm: BaseChatModel | None = None,
) -> CompiledStateGraph:
    """Build a Majority Voting graph.

    Agents independently answer the same task. The final output is the majority
    answer; for free-form ties, a voting LLM picks the strongest consensus answer.
    """

    if num_agents < 1:
        raise ValueError("num_agents must be at least 1")

    voter_model = vote_llm or llm

    def major_vote_node(state: MajorVoteState) -> MajorVoteState:
        responses: list[str] = []
        log = list(state.get("log", []))

        for _ in range(num_agents):
            messages = [HumanMessage(content=state["input"])]
            response = llm.invoke(messages)
            responses.append(message_content_to_text(response.content))
            log.extend(build_message_log(messages, response))

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
            "vote_counts": vote.counts,
            "log": log,
        }

    graph = StateGraph(MajorVoteState)
    graph.add_node(MAJOR_VOTE_NODE, major_vote_node)
    graph.add_edge(START, MAJOR_VOTE_NODE)
    graph.add_edge(MAJOR_VOTE_NODE, END)
    return graph.compile()


__all__ = ["build_major_vote_graph"]
