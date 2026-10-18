// PulseOps Clinical DES Simulation Engine & Optimizer (TypeScript Edition)
// 100% faithful port of the Python heapq discrete-event simulation engine.

export interface Patient {
  pid: number;
  t: number;            // arrival minute
  ats: number;          // 1..5
  tri: number;          // triage duration
  consult: number;      // physician consult duration
  admit: boolean;       // whether admitted to inpatient ward
  hold: number;         // boarding duration (min)
  st: 'future' | 'tri_q' | 'tri' | 'bed_q' | 'doc_q' | 'consult' | 'board' | 'done' | 'lwbs';
  t_bed: number | null;
  t_doc: number | null;
  t_dep: number | null;
  t_lwbs: number | null;
  t_due: number | null;
}

export interface Scenario {
  key: string;
  name: string;
  title: string;
  description: string;
  badge: string;
  badgeColor: string;
  badgeBg: string;
  volumeMult: number;
  acuity: [number, number, number, number, number];
  admit: [number, number, number, number, number];
  boardingMult: number;
  burst?: [number, number, number]; // [startHour, endHour, multiplier]
  elderlySurgePct?: number;
}

export const HOURLY_ARRIVAL_PROPORTIONS = [
  0.024, 0.017, 0.013, 0.010, 0.009, 0.010,
  0.015, 0.025, 0.038, 0.052, 0.061, 0.065,
  0.064, 0.063, 0.063, 0.063, 0.062, 0.061,
  0.062, 0.061, 0.057, 0.048, 0.041, 0.030,
];
const _sumH = HOURLY_ARRIVAL_PROPORTIONS.reduce((a, b) => a + b, 0);
export const HOURLY = HOURLY_ARRIVAL_PROPORTIONS.map(x => x / _sumH);

export const BASE_VOLUME = 200;
export const BASE_ACUITY: [number, number, number, number, number] = [0.012, 0.131, 0.384, 0.368, 0.105];
export const BASE_ADMIT: [number, number, number, number, number] = [0.45, 0.45, 0.25, 0.05, 0.05];
export const ATS_TARGET: Record<number, number> = { 1: 0, 2: 10, 3: 30, 4: 60, 5: 120 };
export const ATS_WEIGHT: Record<number, number> = { 1: 100, 2: 25, 3: 10, 4: 3, 5: 1 };
export const GAMMA_SCALE: Record<number, number> = { 1: 15.0, 2: 10.0, 3: 6.67, 4: 4.0, 5: 2.67 };
export const TURNAROUND = 45.0;
export const LWBS_MIN = 180.0;
export const CAPACITY = { nurses: 2, docs: 5, beds: 20 };
export const FT_FACTOR = 0.6;
export const EXPEDITE_FACTOR = 0.7;

export interface HospitalPreset {
  key: string;
  name: string;
  beds: number;
  nurses: number;
  docs: number;
  volumeScale: number;
  desc: string;
}

export const HOSPITAL_PRESETS: Record<string, HospitalPreset> = {
  baseline: {
    key: 'baseline',
    name: 'Standard Unit (20 Beds, 5 Docs)',
    beds: 20,
    nurses: 2,
    docs: 5,
    volumeScale: 1.0,
    desc: 'Standard single-unit ED: 20 treatment bays, 5 attendings, 2 triage nurses (~200 visits/day).',
  },
  small: {
    key: 'small',
    name: 'Small Hospital (18 Beds, 2 Docs)',
    beds: 18,
    nurses: 1,
    docs: 2,
    volumeScale: 0.3,
    desc: 'Small community ED: 18 treatment beds, 2 doctors, 1 triage nurse (~60 visits/day).',
  },
  medium: {
    key: 'medium',
    name: 'Medium Hospital (36 Beds, 3 Docs)',
    beds: 36,
    nurses: 2,
    docs: 3,
    volumeScale: 0.6,
    desc: 'Medium regional hospital: 36 treatment beds, 3 doctors, 2 triage nurses (~120 visits/day).',
  },
  large: {
    key: 'large',
    name: 'Large Hospital (60 Beds, 4 Docs)',
    beds: 60,
    nurses: 3,
    docs: 4,
    volumeScale: 1.0,
    desc: 'Large tertiary academic hospital: 60 treatment beds, 4 doctors, 3 triage nurses (~200 visits/day calibrated against CDC NHAMCS).',
  },
};

export const SCENARIOS: Record<string, Scenario> = {
  normal: {
    key: "normal",
    name: "Normal Monday Baseline",
    title: "Baseline Monday Operations",
    description: "Typical diurnal arrival pattern (~200 visits/24h) with standard scheduled staffing. Baseline comparison point.",
    badge: "STANDARD OPERATIONS",
    badgeColor: "#334155",
    badgeBg: "#f1f5f9",
    volumeMult: 1.0,
    acuity: BASE_ACUITY,
    admit: BASE_ADMIT,
    boardingMult: 1.0,
  },
  heatwave: {
    key: "heatwave",
    name: "Extreme Heatwave Surge",
    title: "Extreme Heatwave Surge (Texas 2011 & 2025)",
    description: "Nature Health 2025 / CDC Model: 104°F sustained heat. +8.9% elderly (65+) surge with severe dehydration and cardiovascular exacerbations.",
    badge: "WEATHER ADVISORY: EXTREME HEAT",
    badgeColor: "#b45309",
    badgeBg: "#fffbeb",
    volumeMult: 1.036,
    acuity: [0.012, 0.150, 0.420, 0.330, 0.088],
    admit: [0.50, 0.50, 0.30, 0.06, 0.06],
    boardingMult: 1.25,
    elderlySurgePct: 8.9,
  },
  hurricane: {
    key: "hurricane",
    name: "Hurricane Evacuee Influx",
    title: "Hurricane Evacuee Surge (Harvey Regional Network)",
    description: "Disaster Medicine Model: +11% sustained demand with +40% afternoon evacuee influx presenting with displaced chronic care, injuries, and medication refills.",
    badge: "DISASTER SURGE: HURRICANE INFLUX",
    badgeColor: "#1d4ed8",
    badgeBg: "#eff6ff",
    volumeMult: 1.11,
    acuity: [0.012, 0.120, 0.370, 0.380, 0.118],
    admit: BASE_ADMIT,
    boardingMult: 1.0,
    burst: [14, 20, 1.40],
  },
  compound: {
    key: "compound",
    name: "Compound Calamity (Hurricane Beryl 2024)",
    title: "Compound Calamity (Hurricane Beryl 2024)",
    description: "Category 1 Hurricane landfall followed by 105°F heat index and 2.2M grid power failure. +35% volume, oxygen/device dependency failures, and acute inpatient boarding gridlock.",
    badge: "COMPOUND DISASTER: STORM + OUTAGE + HEAT",
    badgeColor: "#be185d",
    badgeBg: "#fdf2f8",
    volumeMult: 1.35,
    acuity: [0.015, 0.170, 0.440, 0.290, 0.085],
    admit: [0.55, 0.55, 0.33, 0.08, 0.08],
    boardingMult: 1.5,
    elderlySurgePct: 8.9,
  },
};

// PRNG (Mulberry32) for deterministic reproducible simulation
export class PRNG {
  private s: number;
  constructor(seed: number) {
    this.s = seed | 0;
  }
  random(): number {
    let t = (this.s += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  uniform(min: number, max: number): number {
    return min + this.random() * (max - min);
  }
  gauss(mean: number, stdev: number): number {
    let u = 0, v = 0;
    while (u === 0) u = this.random();
    while (v === 0) v = this.random();
    const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    return mean + z * stdev;
  }
  gamma(shape: number, scale: number): number {
    // Marsaglia and Tsang method (shape >= 1)
    const d = shape - 1.0 / 3.0;
    const c = 1.0 / Math.sqrt(9.0 * d);
    while (true) {
      let x = 0, v = 0;
      do {
        x = this.gauss(0, 1);
        v = 1.0 + c * x;
      } while (v <= 0);
      v = v * v * v;
      const u = this.random();
      if (u < 1.0 - 0.0331 * x * x * x * x) return d * v * scale;
      if (Math.log(u) < 0.5 * x * x + d * (1.0 - v + Math.log(v))) return d * v * scale;
    }
  }
  choice<T>(arr: T[], weights: number[]): T {
    const total = weights.reduce((a, b) => a + b, 0);
    let r = this.random() * total;
    for (let i = 0; i < arr.length; i++) {
      r -= weights[i];
      if (r <= 0) return arr[i];
    }
    return arr[arr.length - 1];
  }
}

function poisson(rng: PRNG, lambda: number): number {
  if (lambda <= 0) return 0;
  if (lambda > 500) return Math.max(0, Math.round(rng.gauss(lambda, Math.sqrt(lambda))));
  const limit = Math.exp(-lambda);
  let k = 0, p = 1.0;
  while (true) {
    p *= rng.random();
    if (p <= limit) return k;
    k++;
  }
}

// Generate patient stream
export function generateArrivals(
  sc: Scenario,
  seed: number,
  t0: number = 0,
  t1: number = 1440,
  noise: number = 0,
  volume: number = BASE_VOLUME
): Patient[] {
  const rng = new PRNG(seed);
  const out: Patient[] = [];
  let pid = 0;

  for (let H = Math.floor(t0 / 60); H < Math.ceil(t1 / 60); H++) {
    const f = noise > 0 ? Math.exp(rng.gauss(0, noise)) : 1.0;
    const a = Math.max(t0, H * 60.0);
    const b = Math.min(t1, (H + 1) * 60.0);
    if (b <= a) continue;

    const h = H % 24;
    const burstMult = sc.burst && h >= sc.burst[0] && h < sc.burst[1] ? sc.burst[2] : 1.0;
    const lam = volume * sc.volumeMult * HOURLY[h] * burstMult * f * ((b - a) / 60.0);
    const count = poisson(rng, lam);

    const times: number[] = [];
    for (let i = 0; i < count; i++) {
      times.push(rng.uniform(a, b));
    }
    times.sort((x, y) => x - y);

    for (const ts of times) {
      const ats = rng.choice([1, 2, 3, 4, 5], sc.acuity);
      const admit = rng.random() < sc.admit[ats - 1];
      out.push({
        pid,
        t: ts,
        ats,
        tri: rng.uniform(2.0, 4.0),
        consult: rng.gamma(3.0, GAMMA_SCALE[ats]),
        admit,
        hold: rng.uniform(120.0, 240.0) * sc.boardingMult,
        st: 'future',
        t_bed: null,
        t_doc: null,
        t_dep: null,
        t_lwbs: null,
        t_due: null,
      });
      pid++;
    }
  }
  return out;
}

// Event Queue Item
type EventKind = 'ARR' | 'TRI_END' | 'CON_END' | 'DEPART' | 'LWBS' | 'CAPDN';

interface SimEvent {
  t: number;
  seq: number;
  kind: EventKind;
  ref: any;
}

// Priority Queue for Events
class EventHeap {
  private heap: SimEvent[] = [];
  push(e: SimEvent) {
    this.heap.push(e);
    this.up(this.heap.length - 1);
  }
  pop(): SimEvent | undefined {
    if (this.heap.length === 0) return undefined;
    const top = this.heap[0];
    const bottom = this.heap.pop()!;
    if (this.heap.length > 0) {
      this.heap[0] = bottom;
      this.down(0);
    }
    return top;
  }
  peek(): SimEvent | undefined {
    return this.heap[0];
  }
  size(): number {
    return this.heap.length;
  }
  private up(i: number) {
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (this.heap[i].t < this.heap[p].t || (this.heap[i].t === this.heap[p].t && this.heap[i].seq < this.heap[p].seq)) {
        [this.heap[i], this.heap[p]] = [this.heap[p], this.heap[i]];
        i = p;
      } else break;
    }
  }
  private down(i: number) {
    const len = this.heap.length;
    while ((i << 1) + 1 < len) {
      let left = (i << 1) + 1;
      let right = left + 1;
      let best = i;
      if (this.heap[left].t < this.heap[best].t || (this.heap[left].t === this.heap[best].t && this.heap[left].seq < this.heap[best].seq)) {
        best = left;
      }
      if (right < len && (this.heap[right].t < this.heap[best].t || (this.heap[right].t === this.heap[best].t && this.heap[right].seq < this.heap[best].seq))) {
        best = right;
      }
      if (best !== i) {
        [this.heap[i], this.heap[best]] = [this.heap[best], this.heap[i]];
        i = best;
      } else break;
    }
  }
}

export interface Action {
  nurse: number;
  doc: number;
  fastTrack: boolean;
  expedite: boolean;
  duration: number;
  cost: number;
  name: string;
}

export function parseAction(s: string): Action {
  const parts = s.toUpperCase().split('+').map(x => x.trim()).filter(Boolean);
  const nurse = parts.includes('ADD_NURSE') ? 1 : 0;
  const doc = parts.includes('ADD_PHYSICIAN') ? 1 : 0;
  const fastTrack = parts.includes('FAST_TRACK');
  const expedite = parts.includes('INPATIENT_BUFFER_EXPEDITE');
  const cost = nurse * 200 + doc * 600 + (fastTrack ? 150 : 0) + (expedite ? 300 : 0);
  const activeNames = [];
  if (nurse) activeNames.push('ADD_NURSE');
  if (doc) activeNames.push('ADD_PHYSICIAN');
  if (fastTrack) activeNames.push('FAST_TRACK');
  if (expedite) activeNames.push('INPATIENT_BUFFER_EXPEDITE');
  return {
    nurse,
    doc,
    fastTrack,
    expedite,
    duration: 240.0,
    cost,
    name: activeNames.join('+') || 'DO_NOTHING',
  };
}

// Discrete-Event Hospital Simulator Engine
export class EDSim {
  now: number = 0;
  cap: { n: number; d: number; b: number };
  busy = { n: 0, d: 0, b: 0 };
  pat: Map<number, Patient> = new Map();
  q: Record<'tri_q' | 'bed_q' | 'doc_q', number[]> = { tri_q: [], bed_q: [], doc_q: [] };
  cnt: Record<string, number> = {
    future: 0, tri_q: 0, tri: 0, bed_q: 0, doc_q: 0, consult: 0, board: 0, done: 0, lwbs: 0
  };
  heap = new EventHeap();
  seq = 0;
  ftUntil = -1;
  exUntil = -1;

  constructor(nurses: number = CAPACITY.nurses, docs: number = CAPACITY.docs, beds: number = CAPACITY.beds) {
    this.cap = { n: nurses, d: docs, b: beds };
  }

  addArrivals(arrivals: Patient[]) {
    for (const a of arrivals) {
      const p: Patient = { ...a };
      this.pat.set(p.pid, p);
      this.cnt.future++;
      this.pushEvent(p.t, 'ARR', p.pid);
    }
  }

  private pushEvent(t: number, kind: EventKind, ref: any) {
    this.heap.push({ t, seq: this.seq++, kind, ref });
  }

  private setStatus(p: Patient, st: Patient['st']) {
    this.cnt[p.st]--;
    p.st = st;
    this.cnt[st]++;
  }

  private enqueue(qn: 'tri_q' | 'bed_q' | 'doc_q', p: Patient) {
    this.setStatus(p, qn);
    this.q[qn].push(p.pid);
    // Sort queue by ATS acuity ascending, then arrival time ascending
    this.q[qn].sort((idA, idB) => {
      const a = this.pat.get(idA)!;
      const b = this.pat.get(idB)!;
      if (a.ats !== b.ats) return a.ats - b.ats;
      return a.t - b.t;
    });
  }

  private dequeue(qn: 'tri_q' | 'bed_q' | 'doc_q'): Patient | null {
    while (this.q[qn].length > 0) {
      const pid = this.q[qn].shift()!;
      const p = this.pat.get(pid);
      if (p && p.st === qn) return p;
    }
    return null;
  }

  apply(action: Action) {
    for (let i = 0; i < action.nurse; i++) {
      this.cap.n++;
      this.pushEvent(this.now + action.duration, 'CAPDN', 'n');
    }
    for (let i = 0; i < action.doc; i++) {
      this.cap.d++;
      this.pushEvent(this.now + action.duration, 'CAPDN', 'd');
    }
    if (action.fastTrack) {
      this.ftUntil = Math.max(this.ftUntil, this.now + action.duration);
    }
    if (action.expedite) {
      this.exUntil = Math.max(this.exUntil, this.now + action.duration);
      for (const p of this.pat.values()) {
        if (p.st === 'board' && p.admit && p.t_due !== null && p.t_due > this.now) {
          p.t_due = this.now + EXPEDITE_FACTOR * (p.t_due - this.now);
          this.pushEvent(p.t_due, 'DEPART', p.pid);
        }
      }
    }
    this.dispatch();
  }

  private dispatch() {
    // 1. Triage Nurse Dispatch
    while (this.busy.n < this.cap.n) {
      const p = this.dequeue('tri_q');
      if (!p) break;
      this.busy.n++;
      this.setStatus(p, 'tri');
      this.pushEvent(this.now + p.tri, 'TRI_END', p.pid);
    }
    // 2. Acute Bed Placement
    while (this.busy.b < this.cap.b) {
      const p = this.dequeue('bed_q');
      if (!p) break;
      this.busy.b++;
      p.t_bed = this.now;
      this.enqueue('doc_q', p);
    }
    // 3. Physician Consult Dispatch
    while (this.busy.d < this.cap.d) {
      const p = this.dequeue('doc_q');
      if (!p) break;
      this.busy.d++;
      p.t_doc = this.now;
      this.setStatus(p, 'consult');
      const f = p.ats >= 4 && this.now < this.ftUntil ? FT_FACTOR : 1.0;
      this.pushEvent(this.now + p.consult * f, 'CON_END', p.pid);
    }
  }

  private handleEvent(e: SimEvent) {
    this.now = e.t;
    const kind = e.kind;
    const ref = e.ref;

    if (kind === 'ARR') {
      const p = this.pat.get(ref)!;
      if (p.ats === 1) {
        this.enqueue('bed_q', p); // Resus bypasses triage
      } else {
        this.enqueue('tri_q', p);
      }
      this.pushEvent(p.t + LWBS_MIN, 'LWBS', ref);
    } else if (kind === 'TRI_END') {
      this.busy.n--;
      const p = this.pat.get(ref)!;
      this.enqueue('bed_q', p);
    } else if (kind === 'CON_END') {
      this.busy.d--;
      const p = this.pat.get(ref)!;
      this.setStatus(p, 'board');
      const stay = p.admit
        ? p.hold * (this.now < this.exUntil ? EXPEDITE_FACTOR : 1.0)
        : TURNAROUND;
      p.t_due = this.now + stay;
      this.pushEvent(p.t_due, 'DEPART', ref);
    } else if (kind === 'DEPART') {
      const p = this.pat.get(ref)!;
      if (p.st !== 'board' || (p.t_due !== null && Math.abs(p.t_due - this.now) > 1e-6)) {
        return;
      }
      this.busy.b--;
      p.t_dep = this.now;
      this.setStatus(p, 'done');
    } else if (kind === 'LWBS') {
      const p = this.pat.get(ref);
      if (p && (p.st === 'tri_q' || p.st === 'bed_q' || p.st === 'doc_q')) {
        if (p.t_bed !== null) this.busy.b--;
        this.setStatus(p, 'lwbs');
        p.t_lwbs = this.now;
      }
    } else if (kind === 'CAPDN') {
      if (ref === 'n') this.cap.n--;
      else if (ref === 'd') this.cap.d--;
    }
    this.dispatch();
  }

  runUntil(T: number) {
    while (this.heap.size() > 0 && this.heap.peek()!.t <= T) {
      const e = this.heap.pop()!;
      this.handleEvent(e);
    }
    this.now = T;
  }

  clone(): EDSim {
    const next = new EDSim(this.cap.n, this.cap.d, this.cap.b);
    next.now = this.now;
    next.cap = { ...this.cap };
    next.busy = { ...this.busy };
    next.cnt = { ...this.cnt };
    next.seq = this.seq;
    next.ftUntil = this.ftUntil;
    next.exUntil = this.exUntil;
    next.q = { tri_q: [...this.q.tri_q], bed_q: [...this.q.bed_q], doc_q: [...this.q.doc_q] };
    for (const [k, v] of this.pat.entries()) {
      next.pat.set(k, { ...v });
    }
    // Copy events
    const allEvents: SimEvent[] = [];
    while (this.heap.size() > 0) allEvents.push(this.heap.pop()!);
    for (const e of allEvents) {
      this.heap.push(e);
      next.heap.push({ ...e });
    }
    return next;
  }
}

export interface SimResult {
  breaches: Record<number, number>;
  totalBreaches: number;
  medianWait: number;
  lwbs: number;
  loss: number;
}

export function evaluateForward(
  snapshot: EDSim,
  actionStr: string,
  scenario: Scenario,
  hours: number = 4
): SimResult {
  const sim = snapshot.clone();
  const act = parseAction(actionStr);
  const t0 = sim.now;
  const t1 = sim.now + hours * 60;

  // Add forecast arrivals
  const forecast = generateArrivals(scenario, 11, t0, t1, 0);
  sim.addArrivals(forecast);
  sim.apply(act);
  sim.runUntil(t1);

  // Compute metrics on patients active in window
  const seen = Array.from(sim.pat.values()).filter(p => p.t >= t0 && p.t < t1);
  const breaches: Record<number, number> = { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 };
  const waits: number[] = [];
  let lwbsCount = 0;

  for (const p of seen) {
    if (p.st === 'lwbs') lwbsCount++;
    const wait = p.t_doc !== null ? p.t_doc - p.t : p.t_lwbs !== null ? p.t_lwbs - p.t : t1 - p.t;
    waits.push(Math.max(0, wait));
    if (wait > ATS_TARGET[p.ats]) {
      breaches[p.ats]++;
    }
  }

  waits.sort((a, b) => a - b);
  const medianWait = waits.length > 0 ? waits[Math.floor(waits.length / 2)] : 0;
  const totalBreaches = Object.values(breaches).reduce((a, b) => a + b, 0);

  // Multi-Objective Loss
  const clinicalLoss =
    breaches[1] * 100 +
    breaches[2] * 25 +
    breaches[3] * 10 +
    breaches[4] * 3 +
    breaches[5] * 1 +
    5 * medianWait +
    50 * lwbsCount;
  const totalLoss = clinicalLoss + act.cost / 10.0;

  return { breaches, totalBreaches, medianWait, lwbs: lwbsCount, loss: totalLoss };
}

export const CANDIDATE_ACTIONS = [
  "DO_NOTHING",
  "ADD_NURSE",
  "ADD_PHYSICIAN",
  "FAST_TRACK",
  "INPATIENT_BUFFER_EXPEDITE",
  "ADD_PHYSICIAN+FAST_TRACK",
  "ADD_NURSE+ADD_PHYSICIAN",
  "ADD_PHYSICIAN+INPATIENT_BUFFER_EXPEDITE",
  "FAST_TRACK+INPATIENT_BUFFER_EXPEDITE",
  "ADD_PHYSICIAN+FAST_TRACK+INPATIENT_BUFFER_EXPEDITE",
];

export interface Recommendation {
  action: string;
  headline: string;
  base: SimResult;
  best: SimResult;
  breachReductionPct: number;
  waitReduction: number;
  cost: number;
}

export function recommendAction(snapshot: EDSim, scenario: Scenario): Recommendation {
  const baseRes = evaluateForward(snapshot, "DO_NOTHING", scenario, 4);
  let bestAction = "DO_NOTHING";
  let bestRes = baseRes;

  for (const act of CANDIDATE_ACTIONS) {
    if (act === "DO_NOTHING") continue;
    const res = evaluateForward(snapshot, act, scenario, 4);
    if (res.loss < bestRes.loss) {
      bestRes = res;
      bestAction = act;
    }
  }

  const actParsed = parseAction(bestAction);
  const waitSaved = Math.max(0, baseRes.medianWait - bestRes.medianWait);
  const breachSavedPct = baseRes.totalBreaches > 0
    ? Math.max(0, ((baseRes.totalBreaches - bestRes.totalBreaches) / baseRes.totalBreaches) * 100)
    : 0;

  let headline = "Hold current staffing. No intervention beats baseline.";
  if (bestAction !== "DO_NOTHING") {
    const prettyName = bestAction.split('+').map(p => {
      if (p === 'ADD_NURSE') return 'Surge Triage Nurse';
      if (p === 'ADD_PHYSICIAN') return 'On-Call Attending';
      if (p === 'FAST_TRACK') return 'Fast-Track Protocol';
      if (p === 'INPATIENT_BUFFER_EXPEDITE') return 'Expedite Ward Transfers';
      return p;
    }).join(' + ');
    headline = `${prettyName}. Projected Breach Cut: ${breachSavedPct.toFixed(0)}%, Wait Saved: ${waitSaved.toFixed(0)}m (Cost: $${actParsed.cost})`;
  }

  return {
    action: bestAction,
    headline,
    base: baseRes,
    best: bestRes,
    breachReductionPct: breachSavedPct,
    waitReduction: waitSaved,
    cost: actParsed.cost,
  };
}

export interface EarlyWarningRisk {
  urgentLateRisk: number;
  bedsFullRisk: number;
  lwbsRisk: number;
  breaksFirst: string;
  alarm: boolean;
  message: string;
}

export function evaluateEarlyWarningRisk(
  sim: EDSim,
  scenario: Scenario
): EarlyWarningRisk {
  const bedPct = sim.busy.b / Math.max(1, sim.cap.b);
  const waitingTotal = sim.q.tri_q.length + sim.q.bed_q.length;
  const boarders = sim.cnt.board;
  const docsBusyFrac = sim.busy.d / Math.max(1, sim.cap.d);

  // Calibrated surrogate response surface matching the early warning XGBoost / logistic models
  const zBed = (bedPct - 0.72) * 9.0 + (boarders / Math.max(1, sim.cap.b)) * 4.5 + (scenario.volumeMult - 1.0) * 3.0;
  const riskBedsFull = Math.min(0.999, Math.max(0.01, 1 / (1 + Math.exp(-zBed))));

  const zLwbs = (waitingTotal / Math.max(1, sim.cap.b) - 0.35) * 7.5 + (bedPct - 0.70) * 4.0;
  const riskLwbs = Math.min(0.999, Math.max(0.01, 1 / (1 + Math.exp(-zLwbs))));

  const zUrgent = (docsBusyFrac - 0.60) * 6.0 + (sim.q.bed_q.length / Math.max(1, sim.cap.b)) * 5.0 + (scenario.volumeMult - 1.0) * 2.5;
  const riskUrgentLate = Math.min(0.999, Math.max(0.01, 1 / (1 + Math.exp(-zUrgent))));

  const thr = {
    urgentLate: 0.2525,
    bedsFull: 0.3148,
    lwbs: 0.1382,
  };

  const ratios: Record<string, number> = {
    'urgent patients seen late': riskUrgentLate / thr.urgentLate,
    'beds full (above 95%)': riskBedsFull / thr.bedsFull,
    'patients leaving without being seen': riskLwbs / thr.lwbs,
  };

  let topResource = 'beds full (above 95%)';
  let maxRatio = -1;
  for (const [res, r] of Object.entries(ratios)) {
    if (r > maxRatio) {
      maxRatio = r;
      topResource = res;
    }
  }

  const alarm = maxRatio >= 1.0;
  const topRisk = topResource === 'urgent patients seen late' ? riskUrgentLate
    : topResource === 'beds full (above 95%)' ? riskBedsFull : riskLwbs;

  const message = alarm
    ? `Watch ${topResource}: chance ${(topRisk * 100).toFixed(0)}% in the next 4 hours. Early warning surrogate alarm active.`
    : `No alarm: all three forward risk metrics are below alarm limits.`;

  return {
    urgentLateRisk: riskUrgentLate,
    bedsFullRisk: riskBedsFull,
    lwbsRisk: riskLwbs,
    breaksFirst: alarm ? topResource : 'nothing expected to break',
    alarm,
    message,
  };
}

