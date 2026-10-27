import unittest

from tests.helpers import REPO  # noqa: F401

from lifemate.train.sft_lora import assistant_char_spans


class SpanTest(unittest.TestCase):
    def test_spans_cover_content_and_end_token(self):
        msgs = [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
            {"role": "assistant", "content": '{"name": "finish"}'},
        ]
        text = "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n" + '{"name": "finish"}<|im_end|>\n'
        (a, b), = assistant_char_spans(text, msgs, "<|im_end|>")
        self.assertEqual(text[a:b], '{"name": "finish"}<|im_end|>')


if __name__ == "__main__":
    unittest.main()
