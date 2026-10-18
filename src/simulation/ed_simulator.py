"""Pure-Python heapq discrete-event ED simulator with freeze/resume snapshots.

Flow: arrival -> triage nurse (ATS 1 bypasses triage) -> acute bed (priority ATS,
then arrival) -> physician consult -> bed held for turnaround (discharge) or
boarding (admitted). Physicians are released after the consult, never for the
whole stay. All randomness is pre-drawn per patient, so the engine itself is
deterministic: identical inputs (CRN) differ only through the action applied.
"""
from __future__ import annotations

import copy
import heapq
import statistics
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

from src.simulation.scenarios import (ATS_TARGET, CAPACITY, LWBS_MIN, SCENARIOS, TURNAROUND,
                                      Arrival, Scenario, generate_arrivals)

ARR, TRI_END, CON_END, DEPART, LWBS, CAPDN = range(6)
WAITING = ("tri_q", "bed_q", "doc_q")
STATES = ("future", "tri_q", "tri", "bed_q", "doc_q", "consult", "board", "done", "lwbs")
COSTS = {"ADD_NURSE": 200.0, "ADD_PHYSICIAN": 600.0, "FAST_TRACK": 150.0,
         "INPATIENT_BUFFER_EXPEDITE": 300.0}
FT_FACTOR, EXPEDITE_FACTOR = 0.6, 0.7   # -40% ATS4/5 physician time, -30% boarding


@dataclass(frozen=True)
class Action:
    nurse: int = 0
    doc: int = 0
    fast_track: bool = False
    expedite: bool = False
    duration: float = 240.0

    @property
    def parts(self) -> list[str]:
        return (["ADD_NURSE"] * (self.nurse > 0) + ["ADD_PHYSICIAN"] * (self.doc > 0)
                + ["FAST_TRACK"] * self.fast_track + ["INPATIENT_BUFFER_EXPEDITE"] * self.expedite)

    @property
    def name(self) -> str:
        return "+".join(self.parts) or "DO_NOTHING"

    @property
    def cost(self) -> float:
        return COSTS["ADD_NURSE"] * self.nurse + COSTS["ADD_PHYSICIAN"] * self.doc \
            + COSTS["FAST_TRACK"] * self.fast_track + COSTS["INPATIENT_BUFFER_EXPEDITE"] * self.expedite

    @classmethod
    def parse(cls, s: Union[str, "Action", None]) -> "Action":
        if s is None:
            return cls()
        if isinstance(s, Action):
            return s
        parts = {p for p in s.upper().replace(" ", "").split("+") if p and p != "DO_NOTHING"}
        bad = parts - set(COSTS)
        if bad:
            raise ValueError(f"Unknown action component(s): {sorted(bad)}")
        return cls(int("ADD_NURSE" in parts), int("ADD_PHYSICIAN" in parts),
                   "FAST_TRACK" in parts, "INPATIENT_BUFFER_EXPEDITE" in parts)


@dataclass(slots=True)
class P:
    pid: int
    t: float
    ats: int
    tri: float
    consult: float
    admit: bool
    hold: float
    st: str = "future"
    t_bed: Optional[float] = None
    t_doc: Optional[float] = None
    t_dep: Optional[float] = None
    t_lwbs: Optional[float] = None
    t_due: Optional[float] = None   # scheduled departure while boarding


@dataclass
class SimResult:
    t0: float
    t1: float
    n: int                     # tracked patients
    n_started: int
    lwbs: int
    breaches: dict             # ats -> count
    totals: dict               # ats -> tracked count
    median_wait: float         # door-to-doctor (censored at t1 for those still waiting)
    mean_d2d: float
    mean_d2b: float
    mean_los: float
    util: dict                 # nurse / physician / bed (time-weighted)
    queues: dict               # end + max depths
    timeline: list = field(default_factory=list)
    cost: float = 0.0

    @property
    def total_breaches(self) -> int:
        return sum(self.breaches.values())


class EDSim:
    def __init__(self, nurses: int = CAPACITY["nurses"], docs: int = CAPACITY["docs"],
                 beds: int = CAPACITY["beds"], sample_dt: float = 15.0, per_patient_stay: bool = False):
        if sample_dt <= 0:
            raise ValueError("sample_dt must be positive")
        # ADDED 2026-10-18 (patient-flow link, default OFF = old behaviour): when True, a DISCHARGED patient
        # keeps the bed for their own p.hold minutes instead of the flat TURNAROUND constant.
        self.per_patient_stay = per_patient_stay
        self.now = 0.0
        self.cap = {"n": nurses, "d": docs, "b": beds}
        self.busy = {"n": 0, "d": 0, "b": 0}
        self.heap: list = []
        self.seq = 0
        self.pat: dict[int, P] = {}
        self.q = {k: [] for k in WAITING}
        self.cnt = dict.fromkeys(STATES, 0)
        self.ft_until = self.ex_until = -1.0
        self.sample_dt = sample_dt
        # ADDED 2026-10-18 (fix #7): ADMITTED patients holding a bed while they wait for a ward. cnt["board"] also
        # counts discharged patients still in a bed after the consult, so it is not a boarding count.
        self.board_admitted = 0
        self.reset_stats()

    # ---- bookkeeping -------------------------------------------------------
    def reset_stats(self) -> None:
        self.area = {"n": 0.0, "d": 0.0, "b": 0.0}
        self.carea = {"n": 0.0, "d": 0.0, "b": 0.0}
        self.samples: list = []
        self.next_sample = self.now
        self.mx = {"tri": self.cnt["tri_q"], "prov": self.cnt["bed_q"] + self.cnt["doc_q"]}

    def _push(self, t: float, kind: int, ref) -> None:
        heapq.heappush(self.heap, (t, self.seq, kind, ref))
        self.seq += 1

    def _set(self, p: P, st: str) -> None:
        self.cnt[p.st] -= 1
        p.st = st
        self.cnt[st] += 1

    def _enq(self, qn: str, p: P) -> None:
        self._set(p, qn)
        heapq.heappush(self.q[qn], (p.ats, p.t, p.pid))

    def _pop(self, qn: str) -> Optional[P]:
        q = self.q[qn]
        while q:
            p = self.pat.get(heapq.heappop(q)[2])
            if p is not None and p.st == qn:   # lazy deletion of LWBS'd/pruned patients
                return p
        return None

    def add_arrivals(self, arrs: Sequence[Arrival]) -> None:
        for a in arrs:
            if a.pid in self.pat:
                raise ValueError(f"duplicate patient id {a.pid}")
            self.pat[a.pid] = P(*a)
            self.cnt["future"] += 1
            self._push(a.t, ARR, a.pid)

    def apply(self, action: Union[str, Action, None]) -> None:
        """Mobilise resources effective now, expiring after ``action.duration``."""
        a = Action.parse(action)
        for key, n in (("n", a.nurse), ("d", a.doc)):
            for _ in range(n):
                self.cap[key] += 1
                self._push(self.now + a.duration, CAPDN, key)
        if a.fast_track:
            self.ft_until = max(self.ft_until, self.now + a.duration)
        if a.expedite:
            self.ex_until = max(self.ex_until, self.now + a.duration)
            for p in self.pat.values():     # also accelerate patients already boarding
                if p.st == "board" and p.admit and p.t_due > self.now:
                    p.t_due = self.now + EXPEDITE_FACTOR * (p.t_due - self.now)
                    self._push(p.t_due, DEPART, p.pid)
        self._dispatch()

    # ---- engine ------------------------------------------------------------
    def _dispatch(self) -> None:
        b, now = self.busy, self.now
        while b["n"] < self.cap["n"] and (p := self._pop("tri_q")):
            b["n"] += 1
            self._set(p, "tri")
            self._push(now + p.tri, TRI_END, p.pid)
        while b["b"] < self.cap["b"] and (p := self._pop("bed_q")):
            b["b"] += 1
            p.t_bed = now
            self._enq("doc_q", p)
        while b["d"] < self.cap["d"] and (p := self._pop("doc_q")):
            b["d"] += 1
            p.t_doc = now
            self._set(p, "consult")
            f = FT_FACTOR if p.ats >= 4 and now < self.ft_until else 1.0
            self._push(now + p.consult * f, CON_END, p.pid)

    def _handle(self, kind: int, ref) -> None:
        b, now = self.busy, self.now
        if kind == ARR:
            p = self.pat[ref]
            self._enq("bed_q" if p.ats == 1 else "tri_q", p)
            self._push(p.t + LWBS_MIN, LWBS, ref)
        elif kind == TRI_END:
            b["n"] -= 1
            self._enq("bed_q", self.pat[ref])
        elif kind == CON_END:
            b["d"] -= 1
            p = self.pat[ref]
            self._set(p, "board")
            if p.admit:
                self.board_admitted += 1
                stay = p.hold * (EXPEDITE_FACTOR if now < self.ex_until else 1.0)
            else:   # ADDED 2026-10-18: per-patient stay only when the flag is on; otherwise the old constant
                stay = p.hold if self.per_patient_stay else TURNAROUND
            p.t_due = now + stay
            self._push(p.t_due, DEPART, ref)
        elif kind == DEPART:
            # CHANGED 2026-10-17 (bug fix): an "expedite" leaves the patient's OLD departure event in
            # the queue. If that patient was removed from self.pat by snapshot(prune=True) before the
            # old event fired, self.pat[ref] raised KeyError and the whole run crashed. Using .get() and
            # skipping a missing patient is safe: that patient had already left, so there is nothing to do.
            p = self.pat.get(ref)
            if p is None or p.st != "board" or abs(p.t_due - now) > 1e-9:   # gone, or superseded by an expedite
                return
            b["b"] -= 1
            p.t_dep = now
            if p.admit:
                self.board_admitted -= 1
            self._set(p, "done")
        elif kind == LWBS:
            p = self.pat.get(ref)
            if p is not None and p.st in WAITING:
                if p.t_bed is not None:
                    b["b"] -= 1
                self._set(p, "lwbs")
                p.t_lwbs = now
        elif kind == CAPDN:
            self.cap[ref] -= 1
        self._dispatch()
        self.mx["tri"] = max(self.mx["tri"], self.cnt["tri_q"])
        self.mx["prov"] = max(self.mx["prov"], self.cnt["bed_q"] + self.cnt["doc_q"])

    def _adv(self, t: float) -> None:
        while self.next_sample <= t:
            self.samples.append({"t": self.next_sample, "beds": self.busy["b"], "docs": self.busy["d"],
                                 "tri_q": self.cnt["tri_q"], "prov_q": self.cnt["bed_q"] + self.cnt["doc_q"],
                                 "board": self.cnt["board"], "board_admitted": self.board_admitted})
            self.next_sample += self.sample_dt
        dt = t - self.now
        for k in "ndb":
            self.area[k] += self.busy[k] * dt
            self.carea[k] += self.cap[k] * dt
        self.now = t

    def run_until(self, T: float) -> None:
        if T < self.now:
            raise ValueError("cannot run backwards in time")
        while self.heap and self.heap[0][0] <= T:
            t, _, kind, ref = heapq.heappop(self.heap)
            self._adv(t)
            self._handle(kind, ref)
        self._adv(T)

    # ---- snapshots ---------------------------------------------------------
    def snapshot(self, strip_future: bool = False, prune: bool = False) -> "EDSim":
        """Deep copy at ``now`` (queues, beds, consults, scheduled departures).

        strip_future drops not-yet-arrived patients so a forecaster cannot peek at
        the live stochastic future; prune drops finished/LWBS patients (lighter copy).
        """
        s = copy.deepcopy(self)
        if strip_future:
            s.heap = [e for e in s.heap if e[2] != ARR]
            heapq.heapify(s.heap)
        for pid in [i for i, p in s.pat.items()
                    if (strip_future and p.st == "future") or (prune and p.st in ("done", "lwbs"))]:
            s.cnt[s.pat[pid].st] -= 1
            del s.pat[pid]
        return s

    @staticmethod
    def restore(snap: "EDSim") -> "EDSim":
        return copy.deepcopy(snap)

    # ---- results -----------------------------------------------------------
    def result(self, t0: float, cost: float = 0.0) -> SimResult:
        t1 = self.now
        inc = [p for p in self.pat.values() if p.st != "future" and (
            (p.t_doc is None and p.st != "lwbs") or (p.t_doc is not None and p.t_doc >= t0)
            or (p.st == "lwbs" and p.t_lwbs >= t0))]
        br, tot, waits = {i: 0 for i in range(1, 6)}, {i: 0 for i in range(1, 6)}, []
        for p in inc:
            w = (p.t_doc if p.t_doc is not None else p.t_lwbs if p.t_lwbs is not None else t1) - p.t
            waits.append(w)
            tot[p.ats] += 1
            br[p.ats] += w > ATS_TARGET[p.ats] + 1e-9 or p.st == "lwbs"
        mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
        ut = lambda k: self.area[k] / self.carea[k] if self.carea[k] > 0 else 0.0
        return SimResult(
            t0, t1, len(inc), sum(p.t_doc is not None for p in inc), sum(p.st == "lwbs" for p in inc),
            {k: int(v) for k, v in br.items()}, tot,
            statistics.median(waits) if waits else 0.0,
            mean([p.t_doc - p.t for p in inc if p.t_doc is not None]),
            mean([p.t_bed - p.t for p in inc if p.t_bed is not None]),
            mean([p.t_dep - p.t for p in inc if p.t_dep is not None]),
            {"nurse": ut("n"), "physician": ut("d"), "bed": ut("b")},
            {"triage_end": self.cnt["tri_q"], "provider_end": self.cnt["bed_q"] + self.cnt["doc_q"],
             "triage_max": self.mx["tri"], "provider_max": self.mx["prov"]},
            list(self.samples), cost)


def simulate(snapshot: Optional[EDSim] = None, arrivals: Optional[Sequence[Arrival]] = None,
             action: Union[str, Action, None] = None, seed: int = 42, hours: float = 4,
             scenario: Optional[Scenario] = None) -> SimResult:
    """Run ``hours`` forward from a snapshot (or an empty ED at t=0).

    arrivals=None -> synthesise a forecast stream for the window from ``scenario``
    (default Normal Monday) and ``seed``, unless the snapshot already holds future arrivals.
    The snapshot is never mutated, so one snapshot can serve many counterfactuals.
    """
    sim = EDSim() if snapshot is None else EDSim.restore(snapshot)
    t0, t1 = sim.now, sim.now + hours * 60.0
    if arrivals is None and not any(e[2] == ARR for e in sim.heap):
        arrivals = generate_arrivals(scenario or SCENARIOS["normal"], seed, t0, t1,
                                     base_pid=max(sim.pat, default=-1) + 1)
    sim.add_arrivals([a for a in arrivals or () if t0 <= a.t < t1])
    act = Action.parse(action)
    sim.reset_stats()
    sim.apply(act)
    sim.run_until(t1)
    return sim.result(t0, act.cost)


class LiveED:
    """Ground-truth day: noisy arrival stream advanced in real time."""

    def __init__(self, scenario: Scenario, seed: int = 42, noise: float = 0.08, hours: float = 24,
                 nurses: int = CAPACITY["nurses"], docs: int = CAPACITY["docs"],
                 beds: int = CAPACITY["beds"], per_patient_stay: bool = False,
                 volume_scale: float = 1.0):
        self.scenario = scenario
        arrivals = generate_arrivals(scenario, seed, 0.0, hours * 60.0, noise=noise)
        if volume_scale < 1.0:
            import random
            arrivals = [a for a in arrivals if random.Random(f"thin:{a.pid}:{a.t:.4f}").random() < volume_scale]
        self.arrivals = arrivals
        self.sim = EDSim(nurses=nurses, docs=docs, beds=beds, per_patient_stay=per_patient_stay)
        self.sim.add_arrivals(self.arrivals)

    @property
    def now(self) -> float:
        return self.sim.now

    def advance_to(self, t: float) -> None:
        self.sim.run_until(t)

    def advance(self, minutes: float) -> None:
        self.sim.run_until(self.sim.now + minutes)

    def snapshot(self) -> EDSim:
        """Optimizer-safe view: current state only, no future arrivals."""
        return self.sim.snapshot(strip_future=True, prune=True)
