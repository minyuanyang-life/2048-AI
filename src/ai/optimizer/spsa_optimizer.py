from dataclasses import dataclass
import random

@dataclass
class SPSAProposal:
    plus: list[float]
    minus: list[float]
    delta: list[int]
    c_t: float


class SPSAOptimizer:
    """
    Minimal SPSA optimizer for 3-dim heuristic weights.
    """

    def __init__(
        self,
        initial: list[float],
        seed: int | None = None,
        a: float = 0.2,
        c: float = 0.1,
        alpha: float = 0.602,
        gamma: float = 0.101,
        A: float = 10.0,
        lower: float = 0.0,
        upper: float = 5.0,
        early_stop_patience: int | None = None,
        min_improve: float = 0.0,
    ) -> None:
        self._params = list(initial)
        self._rng = random.Random(seed)
        self._a = a
        self._c = c
        self._alpha = alpha
        self._gamma = gamma
        self._A = A
        self._lower = lower
        self._upper = upper
        self._t = 0
        self._early_stop_patience = early_stop_patience
        self._min_improve = min_improve
        self._best_score: float | None = None
        self._no_improve_rounds = 0

    def get_params(self) -> list[float]:
        return list(self._params)

    def propose(self) -> SPSAProposal:
        c_t = self._c / ((self._t + 1) ** self._gamma)
        base = list(self._params)
        delta = [1 if self._rng.random() < 0.5 else -1 for _ in base]

        plus_list = [self._clip(v + c_t * d) for v, d in zip(base, delta)]
        minus_list = [self._clip(v - c_t * d) for v, d in zip(base, delta)]

        return SPSAProposal(
            plus=plus_list,
            minus=minus_list,
            delta=delta,
            c_t=c_t,
        )

    def update(self, proposal: SPSAProposal, score_plus: float, score_minus: float) -> list[float]:
        grad_scale = (score_plus - score_minus) / (2.0 * proposal.c_t)
        g_hat = [grad_scale / d for d in proposal.delta]

        a_t = self._a / ((self._t + 1 + self._A) ** self._alpha)
        cur = list(self._params)
        nxt = [self._clip(v + a_t * g) for v, g in zip(cur, g_hat)]

        self._params = nxt
        self._t += 1
        return self.get_params()

    def observe(self, score: float) -> None:
        if self._best_score is None:
            self._best_score = score
            self._no_improve_rounds = 0
            return

        if score > self._best_score + self._min_improve:
            self._best_score = score
            self._no_improve_rounds = 0
        else:
            self._no_improve_rounds += 1

    def should_stop(self) -> bool:
        return (
            self._early_stop_patience is not None
            and 0 < self._early_stop_patience <= self._no_improve_rounds
        )

    def no_improve_rounds(self) -> int:
        return self._no_improve_rounds

    def _clip(self, value: float) -> float:
        return min(max(value, self._lower), self._upper)
