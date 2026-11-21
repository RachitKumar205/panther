"""The winner (on the winner set) and the verdict (on the verdict set). See README.md."""

from panther.verdict.gsm8k import GSM8K
from panther.verdict.verdict import Verdict, judge
from panther.verdict.winner import Winner

__all__ = ["GSM8K", "Verdict", "Winner", "judge"]
