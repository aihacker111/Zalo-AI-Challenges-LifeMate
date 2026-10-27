import unittest

from lifemate.asr.normalize_vi import read_number, spoken_form, to_spoken


class ReadNumberTest(unittest.TestCase):
    def test_cardinals(self):
        cases = {
            0: "không",
            15: "mười lăm",
            21: "hai mươi mốt",
            24: "hai mươi bốn",
            105: "một trăm linh năm",
            124: "một trăm hai mươi bốn",
            1005: "một nghìn không trăm linh năm",
            2000000: "hai triệu",
            1050000: "một triệu không trăm năm mươi nghìn",
        }
        for n, words in cases.items():
            self.assertEqual(read_number(n), words, n)


class SpokenFormTest(unittest.TestCase):
    def test_unambiguous(self):
        self.assertEqual(spoken_form("Bật điều hoà 24 độ giúp tôi."), ("bật điều hoà hai mươi bốn độ giúp tôi", True))
        self.assertEqual(spoken_form("Nạp 50.000 nhé"), ("nạp năm mươi nghìn nhé", True))
        self.assertEqual(spoken_form("Báo thức 7 rưỡi, 9h nhé!")[0], "báo thức bảy rưỡi chín giờ nhé")

    def test_ambiguous_readings_are_flagged(self):
        for text in ["chuyến RV245", "kênh VTV3", "vé số 412750", "lúc 7h30", "ngày 6/10", "gọi 0905123456"]:
            self.assertFalse(spoken_form(text)[1], text)

    def test_asr_output_is_unchanged(self):
        hyp = "đặt báo thức bảy giờ sáng mai"
        self.assertEqual(to_spoken(hyp), hyp)


if __name__ == "__main__":
    unittest.main()
