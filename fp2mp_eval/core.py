
import pandas as pd
# from concurrent.futures import ThreadPoolExecutor
from langchain_openai import ChatOpenAI
from tqdm.contrib.concurrent import thread_map
from .models import Evaluation, Dimension
from ._prompt import EVAL_PROMPT
from ._config import config

class FP2MPEval():

    def __init__(self, model : str = 'deepseek/deepseek-v3.2', temperature : float = 0.5, n_judges : int = 10, **kwargs):
        self._llm = ChatOpenAI(
            model=model,
            base_url=config.base_url,
            api_key=config.api_key,
            temperature=temperature,
            **kwargs
        )
        self._n_judges = n_judges

    @property
    def llm(self) -> ChatOpenAI:
        return self._llm.with_structured_output(Evaluation)
    
    def evaluations_to_df(self, evaluations : list[Evaluation]) -> pd.DataFrame:
        data = []

        for evaluation in evaluations:
            d = {}
            dump = evaluation.model_dump()
            for k,v in dump.items():
                d[k] = v['score']
            data.append(d)
            
        return pd.DataFrame(data)
    
    def evaluations_to_long_df(self, evaluations : list[Evaluation]) -> pd.DataFrame:
        data = []

        for judge, evaluation in enumerate(evaluations):
            for indicator, v in evaluation.model_dump().items():
                data.append({
                    'judge': judge,
                    'indicator': indicator,
                    **v
                })

        return pd.DataFrame(data)
    
    def evaluate_case(self, case : tuple[str,str], max_workers : int = 2) -> list[Evaluation]:
        problem, solution = case
        message = EVAL_PROMPT.format(problem=problem, solution=solution)
        n_judges = self._n_judges

        def invoke(*args, **kwargs):
            return self.llm.invoke(message)

        results = thread_map(invoke, range(n_judges), max_workers=max_workers)

        return results
    
    # def evaluate_cases(self, cases : list[tuple[str,str]], max_workers : int = 10) -> list[list[Evaluation]]:

    #     with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #         results = list(executor.map(self.evaluate_case, cases))

    #     return results