from typing import Dict
from tqdm.auto import tqdm 

from pipelines.agents import (

    ProblemFinder, ProblemValidator,
    MethodValidator,
    ExperimentValidator
)
from utils.evaluation import get_avg_feedbacks_score, get_num_feedbacks_scores


class ReviewPipeline:
    def __init__(self, api_client=None, iterations: int = 3):

        self.iterations = max(1, iterations)
        self.problem_identifier = ProblemFinder(api_client)
        self.problem_validator = ProblemValidator(api_client)
        self.method_validator = MethodValidator(api_client)
        self.experiment_validator = ExperimentValidator(api_client)

        self._labels = [
            'Problem Identifier', 'Problem Validator',
            'Method Validator',
            'Experiment Validator'
        ]
        self._label_width = max(len(s) for s in self._labels)

    def _log(self, message: str) -> None:
        tqdm.write(str(message))

    def _fmt(self, label: str) -> str:
        return f'[{label.ljust(self._label_width)}]'

    def run(self, context: Dict) -> Dict:
        print("REVIEW PIPELINE | start")
        history = {'problems': [], 'methods': [], 'experiments': []}
        self._log(f"{self._fmt('Paper Title')} {next(iter(context.keys()))}")
        
        

        print("RESEARCH PIPELINE | start Problem")
        # Problem validation
        for i in range(self.iterations):
            
            
            self._log(f"{self._fmt('Problem Finder')} Iteration {i + 1}/{self.iterations} — individuating problem…")
            response = self.problem_finder.run(context)
            #print(response)
            context.update(response)
            print("PROBLEM FINDER | FINISHED")
            
            self._log(f"{self._fmt('Problem Validator')} Iteration {i + 1}/{self.iterations} — validating problem…")
            response = self.problem_validator.run(context)
            #print(context)
            context.update(response)

            print("PROBLEM VALIDATOR | FINISHED")
            
            feedback = context.get('problem_feedbacks')
            
            history['problems'].append(
                {
                    'problem': context.get('problem'),
                    'rationale': context.get('problem_rationale'),
                    'feedbacks': feedback,
                }
            )

        print("FEEDBACKS | PROBLEM FEEDBACKS")

        # Select the best-scored problem to use for context
        best_problem = max(
            history['problems'], 
            key=lambda problem: get_avg_feedbacks_score(problem.get('feedbacks') or {}) 
            if get_num_feedbacks_scores(problem.get('feedbacks') or {}) > 0 
            else -1
        )

        problem_feedback = best_problem.get('feedbacks')
        #print(problem_feedback)
        context.update(
            problem=best_problem.get('problem'),
            problem_rationale=best_problem.get('rationale'),
            problem_feedbacks=problem_feedback,
        )

        # Method development and validation
        for i in range(self.iterations):
            
            self._log(f"{self._fmt('Method Validator')} Iteration {i + 1}/{self.iterations} — validating method…")
            #print(ctxUpdate)
            context.update(self.method_validator.run(context))
            
            history['methods'].append(
                {
                    'method': context.get('method'),
                    'rationale': context.get('method_rationale'),
                    'feedbacks': context.get('method_feedbacks'),
                }
            )

        # Select the best-scored method to use for context
        best_method = max(
            history['methods'],
            key=lambda method: get_avg_feedbacks_score(method.get('feedbacks') or {})
            if get_num_feedbacks_scores(method.get('feedbacks') or {}) > 0
            else -1,
        )
        context.update(
            method=best_method.get('method'),
            method_rationale=best_method.get('rationale'),
            method_feedbacks=best_method.get('feedbacks'),
        )

        # Experiment design and validation
        for i in range(self.iterations):
            
            self._log(f"{self._fmt('Experiment Validator')} Iteration {i + 1}/{self.iterations} — validating experiment…")
            context.update(self.experiment_validator.run(context))
            
            history['experiments'].append(
                {
                    'experiment': context.get('experiment'),
                    'rationale': context.get('experiment_rationale'),
                    'feedbacks': context.get('experiment_feedbacks'),
                }
            )

        # Select the best-scored experiment to use for context
        best_experiment = max(
            history['experiments'],
            key=lambda experiment: get_avg_feedbacks_score(experiment.get('feedbacks') or {})
            if get_num_feedbacks_scores(experiment.get('feedbacks') or {}) > 0
            else -1,
        )
        context.update(
            experiment=best_experiment.get('experiment'),
            experiment_rationale=best_experiment.get('rationale'),
            experiment_feedbacks=best_experiment.get('feedbacks'),
        )

        context.update({'history': history})
        return context
