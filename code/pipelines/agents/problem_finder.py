import re
from typing import Dict, List

from .base import BaseAgent
from utils.evaluation import get_low_score_feedbacks
from utils.formatting import list_of_items_to_grammatical_text


class ProblemFinder(BaseAgent):
    def __init__(self, api_client=None):
        super().__init__(api_client)
        self.system_prompt = (
            "You are an AI assistant whose primary goal is to identify key scientific problems "
            "based on existing scientific literature, in order to aid researchers in discovering novel and significant "
            "research opportunities that can advance the field."
        )
        self.messages: List[Dict[str, str]] = [
            {'role': 'system', 'content': self.system_prompt}
        ]
        self.generated: bool = False

    def reset(self):
        self.messages = [{'role': 'system', 'content': self.system_prompt}]
        self.generated = False

    def run(self, context: Dict) -> Dict:
        do_generation = (not self.generated) or (context.get('problem') is None)

        if do_generation:
            self.reset()
            response = self._chat(user_prompt=self._idfentify_prolem_form_sections(context))
            self.generated = True
            return self.parse_output(response)
        else:
            response = self._chat(user_prompt=self._build_refinement_prompt(context))
            return self.parse_output(response)

    def _chat(self, user_prompt: str) -> str:
        self.messages.append({'role': 'user', 'content': user_prompt})
        assistant_reply = self.call(messages=self.messages)
        self.messages.append({'role': 'assistant', 'content': assistant_reply})
        return assistant_reply

    def _idfentify_prolem_form_sections(self, section) -> str:
        # Intro
        prompt = (
            "You are going to find a research problem starting from reading each section in a research paper. "
            f"This will be based on the full text of the target paper, that you will read in instalments in order to better identify the main problem the paper wants to adress"
        )
        # Understanding
        prompt += (
            "Understanding of the target paper is essential: \n"
            "- The target paper is the primary research study you aim to enhance or build upon through future research, serving as the central source and focus for identifying and developing the specific research problem. \n"
        )
        # Approach
        prompt += (
            "Your approach should be systematic: \n"
            "- Start by thoroughly reading the assigned section. \n"
            "- Then generate a  \n"
           )
        # Materials
        prompt += (
            "I am going to provide a section of the target paper as follows: \n\n"
            + section.key() + "\n" + section.value()
        )
        # Final
        prompt += (
            "With the provided target paper, related papers, and entities, your objective now is to formulate a research problem that not only builds upon these existing studies but also strives to be original, clear, feasible, relevant, and significant. "
            "Before crafting the research problem, revisit the title and abstract of the target paper, to ensure it remains the focal point of your research problem identification process. \n\n"
            f"{self._format_target_paper(context['paper'])}"
            "Then, following your review of the above content, please proceed to generate one research problem with the rationale, in the format of \nProblem: \nRationale: \n"
        )
        return prompt

    def _build_refinement_prompt(self, context) -> str:
        target_feedbacks, other_feedbacks = get_low_score_feedbacks(context['problem_feedbacks'])

        # Intro
        prompt = f"You are going to refine the research problem that you formulated, which is: '{context.get('problem')}'. \n\n"
        # Understanding
        prompt += (
            f"Expert reviewers have evaluated this problem across five dimensions: Clarity, Relevance, Originality, Feasibility, and Significance, and identified key areas for improvement in {list_of_items_to_grammatical_text(target_feedbacks)}. "
            f"Your challenge is to enhance these aspects while maintaining the strengths in {list_of_items_to_grammatical_text(other_feedbacks)}. \n\n"
        )
        # Approach
        prompt += (
            "Please follow this systematic approach for refinement: \n"
            "- Begin with a comprehensive review of your research problem and its underlying rationale, while revisiting the context above in which it was formulated, including the target paper, related papers, and entities. \n"
            f"- Next, familiarize yourself with the definitions of the target evaluation criteria identified as the primary areas for improvement: {list_of_items_to_grammatical_text(target_feedbacks)}, and then proceed to carefully read the reviews and feedback provided by expert reviewers on these criteria. \n"
            "- Finally, based on the insights gained from the reviews and feedback, refine your research problem and its rationale, ensuring that the revised problem is original, clear, feasible, relevant, and significant to its field. \n\n"
        )
        # Materials
        prompt += (
            f"I am going to provide the previous problem with its rationale, followed by each of the evaluation criteria, reviews, and feedback for {list_of_items_to_grammatical_text(target_feedbacks)} needing improvement, as follows: \n\n"
            f"Problem: {context.get('problem')} \nRationale: {context.get('problem_rationale')} \n\n"
            + self._format_feedbacks('problem', {metric: feedback for metric, feedback in context['problem_feedbacks'].items() if metric in target_feedbacks})
        )
        # Final
        prompt += (
            f"Finally, with these reviews and feedback about {list_of_items_to_grammatical_text(target_feedbacks)} in mind while maintaining the strengths in {list_of_items_to_grammatical_text(other_feedbacks)}, please craft a revised version of the problem with the rationale, in the format of \nProblem: \nRationale: \n"
        )
        return prompt

    def parse_output(self, text: str) -> Dict[str, str]:
        match = re.search(r"Problem:\s*(.*?)\s*Rationale:\s*(.*)", text, re.DOTALL)
        return (
            {'problem': match.group(1).strip(), 'problem_rationale': match.group(2).strip()}
            if match else {'problem': None, 'problem_rationale': None}
        )
