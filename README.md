# FP2MP Baselines

LangGraph-бейзлайны для text-to-text задач FP2MP.

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Переменные окружения для LLM:

```bash
export FP2MP_API_KEY="..."
export FP2MP_CHAT_URL="..."
```

## Архитектуры

- `SingleAgentBaseline`
- `CotBaseline`
- `ReactBaseline` с `ddgs_tool`
- `GeneratorCriticBaseline`
- `MajorVoteBaseline`
- `DebateBaseline`
- `BlackboardBaseline`

Все бейзлайны хранят скомпилированный LangGraph в `baseline.graph`. Полный результат доступен через `invoke_state(...)`, который вызывает `graph.invoke(...)`; в `state["log"]` сохраняются сообщения агентов и ответы модели.

## Пример

```python
from langchain_openai import ChatOpenAI
from fp2mp_baselines import SingleAgentBaseline
from fp2mp_baselines.config import config


llm = ChatOpenAI(
    model="deepseek/deepseek-v4-flash",
    base_url=config.base_url,
    api_key=config.api_key,
    temperature=0.5,
)

task = "Город Санкт-Петербург. Разработай план уплотнения городского центра."

baseline = SingleAgentBaseline(llm=llm)
state = baseline.invoke_state(task)

print(state["output"])
print(state["log"])
```

Замена архитектуры:

```python
from fp2mp_baselines import (
    BlackboardBaseline,
    CotBaseline,
    DebateBaseline,
    GeneratorCriticBaseline,
    MajorVoteBaseline,
    ReactBaseline,
)

cot = CotBaseline(llm=llm)
react = ReactBaseline(llm=llm)
generator_critic = GeneratorCriticBaseline(generator_llm=llm)
major_vote = MajorVoteBaseline(llm=llm, num_agents=5)
debate = DebateBaseline(llm=llm, num_agents=5, debate_rounds=5)
blackboard = BlackboardBaseline(llm=llm, iterations=3)
```

## Структура

- `fp2mp_baselines/single_agent/` - single-agent baseline
- `fp2mp_baselines/cot/` - Chain-of-Thought style baseline
- `fp2mp_baselines/react/` - ReAct baseline с поиском DuckDuckGo
- `fp2mp_baselines/generator_critic/` - Generator-Critic baseline
- `fp2mp_baselines/major_vote/` - Majority Voting baseline
- `fp2mp_baselines/debate/` - Multi-Agent Debate baseline
- `fp2mp_baselines/blackboard/` - Blackboard baseline
- `fp2mp_baselines/state.py` - общие state-типы
- `fp2mp_baselines/graph_utils.py` - общее логирование LLM-вызовов

## Примеры

Ноутбуки лежат в `examples/`:

- `1_single_agent_baseline.ipynb`
- `2_cot_baseline.ipynb`
- `3_react_baseline.ipynb`
- `4_generator_critic_baseline.ipynb`
- `5_blackboard_baseline.ipynb`
- `6_major_vote_baseline.ipynb`
- `7_debate_baseline.ipynb`
