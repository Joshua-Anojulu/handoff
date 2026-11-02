"""The generate-verify-repair loop.

Simplify, re-extract with the identical extractor, diff, and if the diff is not
empty feed it back as a constraint and try again. Capped at three rounds; a
note that has not converged by then is surfaced as needing human review rather
than shipped.

Two decisions worth stating because they are easy to get wrong:

**The loop returns its best round, not its last.** Repair is not monotone: a
round that fixes an omission can introduce a corruption, and round three can be
worse than round one. Keeping the last attempt would throw away a better
rewrite and would make the aggregate numbers depend on where the cap happened
to fall.

**Not converging is an output, not a failure.** `needs_review` is the honest
answer for a note the loop cannot simplify without losing something, and a
system that always returns prose would be hiding that case rather than handling
it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .diff import DiffResult, diff
from .extract import extract
from .readability import Readability, assess
from .schema import FactSet

MAX_ROUNDS = 3
TARGET_GRADE = 6.0


@dataclass
class Round:
    index: int
    text: str
    facts: FactSet
    result: DiffResult
    readability: Readability
    constraints_given: list[str] = field(default_factory=list)

    @property
    def meets_grade(self) -> bool:
        return self.readability.grade <= TARGET_GRADE

    def as_dict(self) -> dict:
        return {"round": self.index, "text": self.text,
                "facts": self.facts.as_dict(), "diff": self.result.as_dict(),
                "readability": self.readability.as_dict(),
                "constraints_given": self.constraints_given}


@dataclass
class LoopResult:
    note_id: str
    original: str
    original_facts: FactSet
    original_readability: Readability
    rounds: list[Round] = field(default_factory=list)
    best_index: int = 0

    @property
    def best(self) -> Round:
        return self.rounds[self.best_index]

    @property
    def converged(self) -> bool:
        return self.best.result.clean

    @property
    def needs_review(self) -> bool:
        return not self.converged

    @property
    def grade_drop(self) -> float:
        return round(self.original_readability.grade - self.best.readability.grade, 2)

    @property
    def jargon_drop(self) -> int:
        return (self.original_readability.unexplained_count
                - self.best.readability.unexplained_count)

    def summary(self) -> str:
        best = self.best
        status = "converged" if self.converged else "NEEDS REVIEW"
        return (f"{self.note_id}: {status} after {len(self.rounds)} round(s). "
                f"grade {self.original_readability.grade} -> {best.readability.grade}, "
                f"unexplained terms {self.original_readability.unexplained_count} -> "
                f"{best.readability.unexplained_count}, "
                f"fidelity {best.result.fidelity:.3f}")

    def as_dict(self) -> dict:
        return {
            "note_id": self.note_id,
            "original": self.original,
            "original_facts": self.original_facts.as_dict(),
            "original_readability": self.original_readability.as_dict(),
            "rounds": [r.as_dict() for r in self.rounds],
            "best_round": self.best_index,
            "converged": self.converged,
            "needs_review": self.needs_review,
            "grade_drop": self.grade_drop,
            "jargon_drop": self.jargon_drop,
        }


def _rank(round_: Round) -> tuple:
    """Sort key for picking the best round. Lower is better.

    Fidelity dominates reading level, deliberately and without a tunable knob:
    an easier note that says the wrong dose is not a better note. Among rounds
    that lose nothing, the easiest to read wins; among rounds that all lose
    something, the one that loses least wins, and reading level breaks the tie.
    """
    return (0 if round_.result.clean else 1,
            -round_.result.weighted_fidelity,
            round_.readability.grade,
            round_.readability.unexplained_count,
            round_.index)


def run(text: str, simplifier, *, note_id: str = "note",
        target_grade: float = TARGET_GRADE,
        max_rounds: int = MAX_ROUNDS) -> LoopResult:
    original_facts = extract(text)
    result = LoopResult(note_id=note_id, original=text,
                        original_facts=original_facts,
                        original_readability=assess(text))

    constraints: list[str] = []
    for index in range(1, max_rounds + 1):
        rewrite = simplifier.simplify(text, target_grade,
                                      constraints if constraints else None)
        facts = extract(rewrite)
        comparison = diff(original_facts, facts)
        result.rounds.append(Round(index=index, text=rewrite, facts=facts,
                                   result=comparison,
                                   readability=assess(rewrite),
                                   constraints_given=list(constraints)))
        if comparison.clean:
            break
        next_constraints = comparison.constraints()
        if next_constraints == constraints:
            # The same complaint twice means the repair step is not moving.
            # Spending another round on it buys nothing.
            break
        constraints = next_constraints

    result.best_index = min(range(len(result.rounds)),
                            key=lambda i: _rank(result.rounds[i]))
    return result
