import unittest

from tests.helpers import REPO  # noqa: F401  (adds src/ to the path)

from lifemate.text import fold, token_f1, tokens


class TextTest(unittest.TestCase):
    def test_tokens_normalise_case_and_unicode(self):
        self.assertEqual(tokens("Nhắn  Hương\u0300"), tokens("nhắn hương̀"))

    def test_token_f1(self):
        self.assertEqual(token_f1("mai chạy bộ dời sang 9 giờ", "mai chạy bộ dời sang 9 giờ"), 1.0)
        self.assertGreater(token_f1("mai chạy bộ dời sang 09:00 nhé", "mai chạy bộ dời sang 09:00"), 0.6)
        self.assertLess(token_f1("đã đóng tiền", "nhớ mang hộ cái sạc"), 0.6)
        self.assertEqual(token_f1("", ""), 1.0)
        self.assertEqual(token_f1("a", ""), 0.0)

    def test_fold(self):
        self.assertEqual(fold("Đèn Phòng Khách"), "den phong khach")


if __name__ == "__main__":
    unittest.main()
