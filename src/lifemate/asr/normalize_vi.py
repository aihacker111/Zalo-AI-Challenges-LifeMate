"""Vietnamese text in spoken form, the way Vietnamese ASR models write their output.

gipformer and most Vietnamese ASR corpora transcribe what was *said*: lower case, no punctuation,
numbers read out in words ("Bật điều hoà 24 độ." -> "bật điều hoà hai mươi bốn độ"). Gold transcripts
in the LifeMate data are written text, so they are converted to this form

* to fine-tune gipformer (the training text must match what the model is supposed to output), and
* to compute a fair WER (an ASR that writes "24" and one that writes "hai mươi bốn" are both right).

:func:`spoken_form` also says whether every conversion was unambiguous. Some written forms can be read
aloud in several ways ("RV245", "412750", "7h30", "6/10"); those utterances are left out of the
fine-tuning data, since a wrong reading would teach the model wrong output. They are about 2% of train.

Only the standard library is used, so the gipformer training environment can load this file directly.
"""

from __future__ import annotations

import re
import unicodedata

DIGITS = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
_SCALES = ["", "nghìn", "triệu", "tỷ"]


def _read_hundreds(n: int, inner: bool) -> list[str]:
    """0 < n < 1000. ``inner``: the group follows a larger one (1005 -> "một nghìn không trăm linh năm")."""
    h, t, u = n // 100, n // 10 % 10, n % 10
    words: list[str] = []
    if h or inner:
        words += [DIGITS[h], "trăm"]
    if t == 0:
        if u:
            if h or inner:
                words.append("linh")
            words.append(DIGITS[u])
    elif t == 1:
        words.append("mười")
        if u:
            words.append("lăm" if u == 5 else DIGITS[u])
    else:
        words += [DIGITS[t], "mươi"]
        if u == 1:
            words.append("mốt")
        elif u == 5:
            words.append("lăm")
        elif u:
            words.append(DIGITS[u])
    return words


def read_digits(s: str) -> str:
    """Digit by digit: "0905" -> "không chín không năm"."""
    return " ".join(DIGITS[int(c)] for c in s if c.isdigit())


def read_number(n: int) -> str:
    """Cardinal number in words: 124 -> "một trăm hai mươi bốn", 2000000 -> "hai triệu"."""
    if n == 0:
        return DIGITS[0]
    groups: list[int] = []
    while n:
        groups.append(n % 1000)
        n //= 1000
    if len(groups) > len(_SCALES):
        return read_digits("".join(str(g).zfill(3) for g in reversed(groups)).lstrip("0"))
    words: list[str] = []
    for i in range(len(groups) - 1, -1, -1):
        if groups[i]:
            words += _read_hundreds(groups[i], inner=i < len(groups) - 1)
            if _SCALES[i]:
                words.append(_SCALES[i])
    return " ".join(words)


# Patterns, applied in this order. Each returns (replacement, unambiguous).
_THOUSANDS = re.compile(r"(?<![\w.,])\d{1,3}(?:\.\d{3})+(?![.,]?\d)")  # 50.000, 1.200.000
_PERCENT = re.compile(r"(\d+(?:[.,]\d+)?)\s*%")
_DECIMAL = re.compile(r"(?<![\w.,])(\d+)[.,](\d+)(?!\w|[.,]\d)")
_CLOCK = re.compile(r"(?<!\w)(\d{1,2})\s*(?:h|g|:)\s*(\d{2})(?!\w)")  # 7h30, 7:30
_HOUR = re.compile(r"(?<!\w)(\d{1,2})h(?!\w)")  # 7h
_DATE = re.compile(r"(?<![\w/])(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?(?![\w/])")
_ALNUM = re.compile(r"\w*(?:[^\W\d_]\d|\d[^\W\d_])\w*")  # RV245, VTV3, E5, 2km
_INT = re.compile(r"\d+")


def _number(digits: str) -> tuple[str, bool]:
    if len(digits) > 1 and digits.startswith("0"):  # phone numbers, codes: read digit by digit
        return read_digits(digits), False
    if len(digits) >= 5:  # 412750: a ticket number may be read either way
        return read_number(int(digits)), False
    return read_number(int(digits)), True


def _alnum(token: str) -> tuple[str, bool]:
    """Letters stay as they are, digit runs are read out: "vtv3" -> "vtv ba"."""
    parts = re.findall(r"\d+|\D+", token)
    return " ".join(_number(p)[0] if p.isdigit() else p for p in parts), False


def spoken_form(text: str) -> tuple[str, bool]:
    """``(spoken text, unambiguous)``: NFC, lower case, numbers in words, no punctuation."""
    s = unicodedata.normalize("NFC", str(text)).lower()
    sure = True

    def sub(pattern: re.Pattern, fn) -> None:
        nonlocal s, sure

        def repl(m: re.Match) -> str:
            nonlocal sure
            out, ok = fn(m)
            sure = sure and ok
            return f" {out} "

        s = pattern.sub(repl, s)

    sub(_THOUSANDS, lambda m: (read_number(int(m.group(0).replace(".", ""))), True))
    sub(_PERCENT, lambda m: (f"{m.group(1)} phần trăm", True))
    sub(_DECIMAL, lambda m: (f"{read_number(int(m.group(1)))} phẩy {read_digits(m.group(2))}", False))
    sub(_CLOCK, lambda m: (f"{read_number(int(m.group(1)))} giờ {read_number(int(m.group(2)))}", False))
    sub(_HOUR, lambda m: (f"{read_number(int(m.group(1)))} giờ", True))
    sub(
        _DATE,
        lambda m: (
            f"{read_number(int(m.group(1)))} tháng {read_number(int(m.group(2)))}"
            + (f" năm {read_number(int(m.group(3)))}" if m.group(3) else ""),
            False,
        ),
    )
    sub(_ALNUM, lambda m: _alnum(m.group(0)))
    sub(_INT, lambda m: _number(m.group(0)))
    if "+" in s:  # "+84", "1+1": no single reading
        sure = False
    s = re.sub(r"[^\w\s]|_", " ", s)
    return " ".join(s.split()), sure


def to_spoken(text: str) -> str:
    """:func:`spoken_form` without the flag (for WER)."""
    return spoken_form(text)[0]
