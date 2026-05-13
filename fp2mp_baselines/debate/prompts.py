DEBATE_ROUND_PROMPT = """These are the latest opinions from the other agents:

{peer_responses}

This is your latest opinion:
{own_response}

Use these opinions as additional input. Reconsider your answer and provide your final answer to the task:
{input}"""


__all__ = ["DEBATE_ROUND_PROMPT"]
