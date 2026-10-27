import json
import unittest

from tests.helpers import KNOWN_NOISY_LABELS, requires_data, split

from lifemate.agent import Agent
from lifemate.agent.fewshot import Retriever
from lifemate.agent.llm import OracleBackend, ScriptedBackend
from lifemate.agent.prompt import PromptBuilder
from lifemate.config import AgentConfig
from lifemate.eval.scorer import Scorer
from lifemate.train.sft_data import trajectory_messages


@requires_data
class OracleAgentTest(unittest.TestCase):
    """Replaying gold actions through prompt + guard + environment must not lose points."""

    @classmethod
    def setUpClass(cls):
        cls.split = split("train")
        cls.agent = Agent(cls.split, OracleBackend(), retriever=Retriever(cls.split.samples))
        cls.scorer = Scorer(cls.split)

    def test_oracle_is_perfect_except_known_noise(self):
        lost = []
        for s in self.split:
            result = self.agent.run(s, s.user_turn_texts())
            self.assertIsNone(result.error, s.id)
            if self.scorer.score_sample(s.id, result.tool_calls).score < 1.0:
                lost.append(s.id)
        self.assertLessEqual(set(lost), KNOWN_NOISY_LABELS)

    def test_training_data_matches_inference_prompts(self):
        cfg = AgentConfig()
        cfg.prompt.fewshot_k = 0
        agent = Agent(self.split, OracleBackend(), cfg)
        builder = PromptBuilder(self.split, cfg.prompt)
        for s in self.split.samples[:300]:
            if s.id in KNOWN_NOISY_LABELS:
                continue
            t = s.user_turn_texts()
            self.assertEqual(agent.run(s, t).messages, trajectory_messages(builder, s, t), s.id)

    def test_prompt_never_mentions_uninstalled_tools(self):
        builder = PromptBuilder(self.split)
        s = self.split.samples[0]
        system, _ = builder.build(s, s.user_turn_texts())[0]
        for name in self.split.tools:
            if name not in s.installed_tools and name != "sys.ask_missing_info":
                self.assertNotIn(f"- {name}(", system["content"])

    def test_parse_errors_are_recovered(self):
        s = next(x for x in self.split if x.gold_calls and x.gold_calls[0]["name"] == "alarm.create")
        call = json.dumps(s.gold_calls[0], ensure_ascii=False)
        result = Agent(self.split, ScriptedBackend(["không phải json", call])).run(s, s.user_turn_texts())
        self.assertEqual(result.tool_calls[0]["name"], "alarm.create")


if __name__ == "__main__":
    unittest.main()
