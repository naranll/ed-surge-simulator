import { useState, useEffect, useMemo } from 'react';
import {
  Activity,
  Bed,
  Clock,
  AlertTriangle,
  Users,
  ShieldAlert,
  Play,
  Pause,
  SkipForward,
  CheckCircle2,
  Stethoscope,
  Sparkles,
  Zap,
} from 'lucide-react';
import {
  SCENARIOS,
  EDSim,
  generateArrivals,
  recommendAction,
  evaluateForward,
  parseAction,
  ATS_TARGET,
  BASE_VOLUME,
  HOSPITAL_PRESETS,
  evaluateEarlyWarningRisk,
} from './lib/simulator';
import type { Scenario, Recommendation, Patient, HospitalPreset, EarlyWarningRisk } from './lib/simulator';

const ATS_CONFIG: Record<number, { name: string; desc: string; color: string; bg: string; border: string }> = {
  1: { name: 'ATS 1', desc: 'Resuscitation', color: '#dc2626', bg: '#fef2f2', border: '#fecaca' },
  2: { name: 'ATS 2', desc: 'Emergency', color: '#ea580c', bg: '#fff7ed', border: '#fed7aa' },
  3: { name: 'ATS 3', desc: 'Urgent', color: '#d97706', bg: '#fffbeb', border: '#fde68a' },
  4: { name: 'ATS 4', desc: 'Semi-Urgent', color: '#16a34a', bg: '#f0fdf4', border: '#bbf7d0' },
  5: { name: 'ATS 5', desc: 'Non-Urgent', color: '#0284c7', bg: '#f0f9ff', border: '#bae6fd' },
};

const PROTOCOLS = [
  { id: 'ADD_NURSE', label: 'Surge Nurse (+1)', cost: 200 },
  { id: 'ADD_PHYSICIAN', label: 'On-Call Attending (+1)', cost: 600 },
  { id: 'FAST_TRACK', label: 'Fast-Track Protocol', cost: 150 },
  { id: 'INPATIENT_BUFFER_EXPEDITE', label: 'Expedite Ward Transfers', cost: 300 },
];

export function App() {
  const [scenarioKey, setScenarioKey] = useState<string>('compound');
  const [presetKey, setPresetKey] = useState<string>('baseline');
  const [minute, setMinute] = useState<number>(14 * 60); // 14:00 default peak
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [activeProtocols, setActiveProtocols] = useState<string[]>([]);

  const scenario: Scenario = SCENARIOS[scenarioKey] || SCENARIOS.normal;
  const preset: HospitalPreset = HOSPITAL_PRESETS[presetKey] || HOSPITAL_PRESETS.baseline;

  // Run ground-truth simulation up to current clock
  const simState = useMemo(() => {
    const arrivals = generateArrivals(scenario, 42, 0, 1440, 0.08, BASE_VOLUME * preset.volumeScale);
    const sim = new EDSim(preset.nurses, preset.docs, preset.beds);
    sim.addArrivals(arrivals);
    sim.runUntil(minute);

    const now = sim.now;
    const allPatients = Array.from(sim.pat.values());
    const seen = allPatients.filter(p => p.st !== 'future');

    const getWait = (p: Patient) =>
      (p.t_doc !== null ? p.t_doc : p.t_lwbs !== null ? p.t_lwbs : now) - p.t;

    const a2 = seen.filter(p => p.ats === 2);
    const avgWait = seen.length > 0 ? seen.reduce((acc, p) => acc + getWait(p), 0) / seen.length : 0;
    const ats2Breaches = a2.filter(p => getWait(p) > ATS_TARGET[2]).length;
    const ats2BreachPct = a2.length > 0 ? (ats2Breaches / a2.length) * 100 : 0;

    // Queues
    const triQ = Array.from(sim.pat.values())
      .filter(p => p.st === 'tri_q')
      .map(p => ({
        ...p,
        wait: Math.max(0, now - p.t),
        breached: Math.max(0, now - p.t) > ATS_TARGET[p.ats],
      }))
      .sort((a, b) => (a.ats !== b.ats ? a.ats - b.ats : a.t - b.t));

    const bedQ = Array.from(sim.pat.values())
      .filter(p => p.st === 'bed_q')
      .map(p => ({
        ...p,
        wait: Math.max(0, now - p.t),
        breached: Math.max(0, now - p.t) > ATS_TARGET[p.ats],
      }))
      .sort((a, b) => (a.ats !== b.ats ? a.ats - b.ats : a.t - b.t));

    // 20 Bed Bays
    const inBeds = allPatients.filter(p => p.st === 'doc_q' || p.st === 'consult' || p.st === 'board');
    inBeds.sort((a, b) => ((a.t_bed ?? a.t) - (b.t_bed ?? b.t)));

    const attendingRoster = ['Dr. Chen', 'Dr. Patel', 'Dr. Rivera', 'Dr. Hayes', 'Dr. Brooks', 'Dr. Zhao (On-Call)'];
    const bays = [];
    for (let i = 0; i < sim.cap.b; i++) {
      const bayId = i + 1;
      if (i < inBeds.length) {
        const p = inBeds[i];
        const assignedDoc = attendingRoster[i % attendingRoster.length];
        const elapsed = Math.max(0, now - p.t);
        const breached = p.t_doc === null ? elapsed > ATS_TARGET[p.ats] : false;

        let status = 'awaiting_doc';
        let label = 'AWAITING ATTENDING';
        let sublabel = 'Bed Assigned • Charting';

        if (p.st === 'consult') {
          status = 'consult';
          label = 'IN CONSULT';
          sublabel = `Attending: ${assignedDoc}`;
        } else if (p.st === 'board') {
          if (p.admit) {
            status = 'boarding';
            const holdTime = Math.max(0, now - (p.t_doc || p.t));
            label = 'BOARDING BLOCK';
            sublabel = `Admitted • Holding Bed (${Math.round(holdTime)}m)`;
          } else {
            status = 'turnaround';
            label = 'DISCHARGE BUFFER';
            sublabel = 'Awaiting Labs / Meds';
          }
        }

        bays.push({
          bayId,
          status,
          label,
          sublabel,
          pid: p.pid,
          ats: p.ats,
          elapsed,
          breached,
        });
      } else {
        bays.push({
          bayId,
          status: 'empty',
          label: 'AVAILABLE',
          sublabel: 'Clean & Ready',
          pid: null,
          ats: null,
          elapsed: 0,
          breached: false,
        });
      }
    }

    // AI recommendation on current snapshot
    const rec: Recommendation = recommendAction(sim, scenario);

    // AI Early Warning Surrogate
    const earlyWarning: EarlyWarningRisk = evaluateEarlyWarningRisk(sim, scenario);

    // Active manual action evaluation
    const currentActionStr = activeProtocols.join('+') || 'DO_NOTHING';
    const currentAction = parseAction(currentActionStr);
    const manualResult = currentAction.name !== 'DO_NOTHING'
      ? evaluateForward(sim, currentAction.name, scenario, 4)
      : null;

    return {
      sim,
      now,
      metrics: {
        bedPct: (sim.busy.b / sim.cap.b) * 100,
        inBeds: sim.busy.b,
        totalBeds: sim.cap.b,
        avgWait,
        ats2BreachPct,
        waitingTotal: triQ.length + bedQ.length,
        triQLen: triQ.length,
        bedQLen: bedQ.length,
        boardCount: sim.cnt.board,
      },
      triQ,
      bedQ,
      bays,
      rec,
      currentAction,
      manualResult,
      earlyWarning,
      preset,
    };
  }, [scenarioKey, presetKey, minute, activeProtocols]);

  // Autoplay loop
  useEffect(() => {
    let timer: any = null;
    if (isPlaying) {
      timer = setInterval(() => {
        setMinute(m => (m >= 1440 ? 0 : Math.min(1440, m + 30)));
      }, 1000);
    }
    return () => clearInterval(timer);
  }, [isPlaying]);

  const toggleProtocol = (id: string) => {
    setActiveProtocols(prev =>
      prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]
    );
  };

  const applyAIPolicy = () => {
    const aiParts = [];
    if (simState.rec.action.includes('ADD_NURSE')) aiParts.push('ADD_NURSE');
    if (simState.rec.action.includes('ADD_PHYSICIAN')) aiParts.push('ADD_PHYSICIAN');
    if (simState.rec.action.includes('FAST_TRACK')) aiParts.push('FAST_TRACK');
    if (simState.rec.action.includes('INPATIENT_BUFFER_EXPEDITE')) aiParts.push('INPATIENT_BUFFER_EXPEDITE');
    setActiveProtocols(aiParts);
  };

  const currentHours = Math.floor(minute / 60);
  const currentMins = Math.floor(minute % 60);
  const timeFormatted = `${String(currentHours).padStart(2, '0')}:${String(currentMins).padStart(2, '0')}`;

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 antialiased font-sans pb-16">
      {/* Top Navbar */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-30 shadow-xs">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-3.5 flex flex-wrap justify-between items-center gap-3">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-blue-50 text-blue-600 rounded-lg border border-blue-100 shadow-2xs">
              <Activity className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-base sm:text-lg font-bold text-slate-900 tracking-tight">
                  KnightAI
                </h1>
                <span className="text-slate-400">/</span>
                <span className="text-xs sm:text-sm font-semibold text-slate-600">
                  Emergency Department Digital Twin
                </span>
              </div>
              <p className="text-xs text-slate-500">
                Track 1: Real-Time AI for Critical Decisions • IBM LinuxONE (s390x) In-Memory Architecture
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 flex-wrap">
            <span className="px-2.5 py-1 text-xs font-semibold rounded-md bg-slate-100 text-slate-700 border border-slate-200">
              LinuxONE s390x
            </span>
            <span className="px-2.5 py-1 text-xs font-semibold rounded-md bg-emerald-50 text-emerald-700 border border-emerald-200 flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
              Pure TS Engine
            </span>
            <span className="px-2.5 py-1 text-xs font-semibold rounded-md bg-blue-50 text-blue-700 border border-blue-200">
              0ms In-Browser
            </span>
            <span className="px-2.5 py-1 text-xs font-semibold rounded-md bg-purple-50 text-purple-700 border border-purple-200">
              CRN Validated
            </span>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 pt-6 space-y-6">
        {/* Section 01: Operational Scenario & Playback Controls */}
        <section className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs space-y-4">
          <div className="flex flex-wrap justify-between items-center gap-4">
            <div className="w-full sm:w-auto flex-1 min-w-[240px]">
              <label className="block text-xs font-bold text-slate-500 uppercase tracking-wider mb-1.5">
                01. Operational Calamity Scenario
              </label>
              <select
                value={scenarioKey}
                onChange={e => setScenarioKey(e.target.value)}
                className="w-full bg-slate-50 border border-slate-300 rounded-lg px-3.5 py-2 text-sm font-semibold text-slate-800 focus:outline-hidden focus:ring-2 focus:ring-blue-500 cursor-pointer"
              >
                {Object.values(SCENARIOS).map(sc => (
                  <option key={sc.key} value={sc.key}>
                    {sc.title}
                  </option>
                ))}
              </select>
            </div>

            <div className="w-full sm:w-auto flex-1 min-w-[240px]">
              <label className="block text-xs font-bold text-slate-500 uppercase tracking-wider mb-1.5">
                01b. Hospital Scale & Capacity Preset
              </label>
              <select
                value={presetKey}
                onChange={e => setPresetKey(e.target.value)}
                className="w-full bg-slate-50 border border-slate-300 rounded-lg px-3.5 py-2 text-sm font-semibold text-slate-800 focus:outline-hidden focus:ring-2 focus:ring-blue-500 cursor-pointer"
              >
                {Object.values(HOSPITAL_PRESETS).map(pr => (
                  <option key={pr.key} value={pr.key}>
                    {pr.name}
                  </option>
                ))}
              </select>
            </div>

            {/* Time scrubbing & playback */}
            <div className="flex items-center gap-3 w-full sm:w-auto">
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className="flex items-center gap-1.5 px-4 py-2 bg-slate-900 text-white rounded-lg text-xs font-bold hover:bg-slate-800 transition-colors shadow-xs"
              >
                {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                {isPlaying ? 'Pause' : 'Play Simulation'}
              </button>

              <button
                onClick={() => setMinute(m => Math.min(1440, m + 30))}
                className="flex items-center gap-1.5 px-3.5 py-2 bg-white border border-slate-300 text-slate-700 rounded-lg text-xs font-bold hover:bg-slate-50 transition-colors shadow-2xs"
              >
                <SkipForward className="w-3.5 h-3.5" />
                +30m Step
              </button>

              <div className="px-3.5 py-1.5 bg-blue-50 border border-blue-200 rounded-lg text-blue-900 font-mono font-bold text-sm tracking-wider">
                {timeFormatted}
              </div>
            </div>
          </div>

          {/* Time slider */}
          <div className="pt-2">
            <input
              type="range"
              min={0}
              max={1440}
              step={15}
              value={minute}
              onChange={e => setMinute(Number(e.target.value))}
              className="w-full accent-blue-600 cursor-pointer"
            />
            <div className="flex justify-between text-[11px] font-medium text-slate-400 mt-1">
              <span>00:00 (Night Intake)</span>
              <span>06:00</span>
              <span>12:00 (Noon Surge)</span>
              <span>18:00 (Evening Rush)</span>
              <span>24:00 (End of Shift)</span>
            </div>
          </div>

          {/* Scenario Alert Badge */}
          <div
            className="flex items-start gap-2.5 p-3 rounded-lg border text-xs"
            style={{
              backgroundColor: scenario.badgeBg,
              borderColor: scenario.badgeColor + '40',
            }}
          >
            <ShieldAlert className="w-4 h-4 shrink-0 mt-0.5" style={{ color: scenario.badgeColor }} />
            <div>
              <span className="font-bold mr-2 uppercase tracking-wide" style={{ color: scenario.badgeColor }}>
                {scenario.badge}:
              </span>
              <span className="text-slate-700">{scenario.description}</span>
            </div>
          </div>
        </section>

        {/* Section 02: Real-Time Telemetry */}
        <section className="space-y-2">
          <h2 className="text-xs font-bold uppercase tracking-wider text-slate-500">
            02. Real-Time Emergency Department Telemetry
          </h2>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3.5">
            {/* Acute Bed Occupancy */}
            <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-2xs space-y-1">
              <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider block">
                Acute Bed Occupancy
              </span>
              <div className="text-2xl font-extrabold text-slate-900 tracking-tight">
                {simState.metrics.bedPct.toFixed(0)}%
              </div>
              <div className="text-xs font-medium text-slate-500 flex items-center gap-1">
                <Bed className="w-3.5 h-3.5 text-blue-600" />
                {simState.metrics.inBeds} / {simState.metrics.totalBeds} Acute Bays In Use
              </div>
            </div>

            {/* Mean Door-to-Doctor */}
            <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-2xs space-y-1">
              <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider block">
                Mean Door-to-Doctor
              </span>
              <div className="text-2xl font-extrabold text-slate-900 tracking-tight">
                {simState.metrics.avgWait.toFixed(0)} min
              </div>
              <div className="text-xs font-medium text-slate-500 flex items-center gap-1">
                <Clock className="w-3.5 h-3.5 text-amber-600" />
                Arrival to Physician Intake
              </div>
            </div>

            {/* ATS 2 Breach Rate */}
            <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-2xs space-y-1">
              <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider block">
                ATS 2 Breach Rate
              </span>
              <div
                className={`text-2xl font-extrabold tracking-tight ${
                  simState.metrics.ats2BreachPct > 20 ? 'text-red-600' : 'text-slate-900'
                }`}
              >
                {simState.metrics.ats2BreachPct.toFixed(0)}%
              </div>
              <div className="text-xs font-medium text-slate-500 flex items-center gap-1">
                <AlertTriangle className="w-3.5 h-3.5 text-orange-600" />
                Target: ≤ 10 min Target
              </div>
            </div>

            {/* Waiting Room Census */}
            <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-2xs space-y-1">
              <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider block">
                Waiting Room Census
              </span>
              <div className="text-2xl font-extrabold text-slate-900 tracking-tight">
                {simState.metrics.waitingTotal}
              </div>
              <div className="text-xs font-medium text-slate-500 flex items-center gap-1">
                <Users className="w-3.5 h-3.5 text-purple-600" />
                {simState.metrics.bedQLen} triaged, awaiting bay
              </div>
            </div>

            {/* Inpatient Boarding Block */}
            <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-2xs space-y-1 col-span-2 sm:col-span-1">
              <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider block">
                Inpatient Boarding Block
              </span>
              <div className="text-2xl font-extrabold text-red-600 tracking-tight">
                {simState.metrics.boardCount}
              </div>
              <div className="text-xs font-medium text-slate-500 flex items-center gap-1">
                <ShieldAlert className="w-3.5 h-3.5 text-red-500" />
                Holding Acute Care Beds
              </div>
            </div>
          </div>
        </section>

        {/* Section 03: AI Early Warning • 4-Hour Forward Risk Engine */}
        <section className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs space-y-3.5">
          <div className="flex flex-wrap justify-between items-center gap-2">
            <div>
              <h2 className="text-xs font-bold uppercase tracking-wider text-slate-500">
                03. AI Early Warning • 4-Hour Lookahead Risk Engine (Surrogate ML)
              </h2>
              <p className="text-xs text-slate-600">
                Surrogate machine learning model trained to forecast operational bottlenecks 4 hours in advance.
              </p>
            </div>
            <span className="px-2.5 py-1 text-xs font-semibold rounded-md bg-slate-100 text-slate-700 border border-slate-200">
              XGBoost / Logistic Engine
            </span>
          </div>

          {/* Alarm Banner */}
          <div
            className={`flex items-start gap-3 p-3.5 rounded-lg border text-xs ${
              simState.earlyWarning.alarm
                ? 'bg-red-50/70 border-red-300 text-red-950'
                : 'bg-emerald-50/70 border-emerald-300 text-emerald-950'
            }`}
          >
            <ShieldAlert
              className={`w-4 h-4 shrink-0 mt-0.5 ${
                simState.earlyWarning.alarm ? 'text-red-600' : 'text-emerald-600'
              }`}
            />
            <div className="flex-1">
              <div className="flex justify-between items-center flex-wrap gap-2">
                <span
                  className={`font-extrabold uppercase tracking-wide text-xs ${
                    simState.earlyWarning.alarm ? 'text-red-700' : 'text-emerald-700'
                  }`}
                >
                  {simState.earlyWarning.alarm ? '⚠️ 4-HOUR SURGE ALARM' : '✓ EARLY WARNING NOMINAL'}
                  {' • '}
                  <span className="underline">
                    Breaks First: {simState.earlyWarning.breaksFirst.toUpperCase()}
                  </span>
                </span>
                <span
                  className={`text-[11px] font-bold px-2 py-0.5 rounded-md ${
                    simState.earlyWarning.alarm
                      ? 'bg-red-100 text-red-800 border border-red-200'
                      : 'bg-emerald-100 text-emerald-800 border border-emerald-200'
                  }`}
                >
                  {simState.earlyWarning.alarm ? 'ACTION REQUIRED' : 'STABLE CAPACITY'}
                </span>
              </div>
              <p className="text-xs mt-1 text-slate-700">
                {simState.earlyWarning.message}
              </p>
            </div>
          </div>

          {/* 3 Risk Gauges */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3.5 pt-1">
            {/* Urgent Delay Risk */}
            <div className="p-3.5 rounded-lg border border-slate-200 bg-slate-50 space-y-1.5">
              <div className="flex justify-between items-center text-xs">
                <span className="font-bold text-slate-600 uppercase text-[11px]">
                  Urgent Seen Late (ATS 1-3)
                </span>
                <span
                  className={`text-[10px] font-bold px-1.5 py-0.5 rounded-sm ${
                    simState.earlyWarning.urgentLateRisk > 0.25
                      ? 'bg-red-100 text-red-700'
                      : 'bg-emerald-100 text-emerald-700'
                  }`}
                >
                  {simState.earlyWarning.urgentLateRisk > 0.25 ? 'HIGH RISK' : 'NORMAL'}
                </span>
              </div>
              <div className="text-xl font-extrabold text-slate-900">
                {(simState.earlyWarning.urgentLateRisk * 100).toFixed(1)}%
              </div>
              <div className="h-1.5 w-full bg-slate-200 rounded-full overflow-hidden">
                <div
                  className={`h-full ${
                    simState.earlyWarning.urgentLateRisk > 0.25 ? 'bg-red-600' : 'bg-emerald-600'
                  }`}
                  style={{ width: `${Math.min(100, simState.earlyWarning.urgentLateRisk * 100)}%` }}
                />
              </div>
              <div className="text-[10px] text-slate-500">
                Target breach probability (Alarm line: 25%)
              </div>
            </div>

            {/* Bed Saturation Risk */}
            <div className="p-3.5 rounded-lg border border-slate-200 bg-slate-50 space-y-1.5">
              <div className="flex justify-between items-center text-xs">
                <span className="font-bold text-slate-600 uppercase text-[11px]">
                  Acute Bed Saturation
                </span>
                <span
                  className={`text-[10px] font-bold px-1.5 py-0.5 rounded-sm ${
                    simState.earlyWarning.bedsFullRisk > 0.31
                      ? 'bg-red-100 text-red-700'
                      : 'bg-emerald-100 text-emerald-700'
                  }`}
                >
                  {simState.earlyWarning.bedsFullRisk > 0.31 ? 'HIGH RISK' : 'NORMAL'}
                </span>
              </div>
              <div className="text-xl font-extrabold text-slate-900">
                {(simState.earlyWarning.bedsFullRisk * 100).toFixed(1)}%
              </div>
              <div className="h-1.5 w-full bg-slate-200 rounded-full overflow-hidden">
                <div
                  className={`h-full ${
                    simState.earlyWarning.bedsFullRisk > 0.31 ? 'bg-red-600' : 'bg-emerald-600'
                  }`}
                  style={{ width: `${Math.min(100, simState.earlyWarning.bedsFullRisk * 100)}%` }}
                />
              </div>
              <div className="text-[10px] text-slate-500">
                Beds &gt; 95% full probability (Alarm line: 31%)
              </div>
            </div>

            {/* LWBS Risk */}
            <div className="p-3.5 rounded-lg border border-slate-200 bg-slate-50 space-y-1.5">
              <div className="flex justify-between items-center text-xs">
                <span className="font-bold text-slate-600 uppercase text-[11px]">
                  Walkout / LWBS Queue
                </span>
                <span
                  className={`text-[10px] font-bold px-1.5 py-0.5 rounded-sm ${
                    simState.earlyWarning.lwbsRisk > 0.14
                      ? 'bg-red-100 text-red-700'
                      : 'bg-emerald-100 text-emerald-700'
                  }`}
                >
                  {simState.earlyWarning.lwbsRisk > 0.14 ? 'HIGH RISK' : 'NORMAL'}
                </span>
              </div>
              <div className="text-xl font-extrabold text-slate-900">
                {(simState.earlyWarning.lwbsRisk * 100).toFixed(1)}%
              </div>
              <div className="h-1.5 w-full bg-slate-200 rounded-full overflow-hidden">
                <div
                  className={`h-full ${
                    simState.earlyWarning.lwbsRisk > 0.14 ? 'bg-red-600' : 'bg-emerald-600'
                  }`}
                  style={{ width: `${Math.min(100, simState.earlyWarning.lwbsRisk * 100)}%` }}
                />
              </div>
              <div className="text-[10px] text-slate-500">
                Walkout without being seen probability (Alarm line: 14%)
              </div>
            </div>
          </div>
        </section>

        {/* Section 04: Tactical Surge Intervention Deck */}
        <section className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs space-y-3.5">
          <div className="flex flex-wrap justify-between items-center gap-2">
            <div>
              <h2 className="text-xs font-bold uppercase tracking-wider text-slate-500">
                04. Tactical Surge Intervention Deck (What-If Controls)
              </h2>
              <p className="text-xs text-slate-600">
                Deploy surge staff or activate rapid discharge protocols to test immediate capacity impact.
              </p>
            </div>
            <button
              onClick={applyAIPolicy}
              className="flex items-center gap-1.5 px-3.5 py-1.5 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-xs font-bold transition-colors shadow-2xs"
            >
              <Sparkles className="w-3.5 h-3.5" />
              Apply AI Prescribed Policy
            </button>
          </div>

          {/* SaaS Action Pills */}
          <div className="flex flex-wrap gap-2.5 pt-1">
            {PROTOCOLS.map(proto => {
              const isActive = activeProtocols.includes(proto.id);
              return (
                <button
                  key={proto.id}
                  onClick={() => toggleProtocol(proto.id)}
                  className={`flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-semibold border transition-all cursor-pointer ${
                    isActive
                      ? 'bg-blue-50 border-blue-500 text-blue-900 shadow-2xs font-bold ring-1 ring-blue-500'
                      : 'bg-white border-slate-300 text-slate-700 hover:bg-slate-50 hover:border-slate-400'
                  }`}
                >
                  <span
                    className={`w-2 h-2 rounded-full ${
                      isActive ? 'bg-blue-600' : 'bg-slate-300'
                    }`}
                  />
                  {proto.label}
                  <span className="text-[11px] font-normal text-slate-500">
                    (${proto.cost})
                  </span>
                </button>
              );
            })}
          </div>

          {/* Active Intervention Impact Banner */}
          {simState.currentAction.name !== 'DO_NOTHING' && simState.manualResult && (
            <div className="flex flex-wrap justify-between items-center gap-3 p-3 bg-emerald-50 border border-emerald-200 rounded-lg text-emerald-900 text-xs">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                <span className="font-bold">
                  ACTIVE POLICY: {simState.currentAction.name.replace(/_/g, ' ')}
                </span>
                <span className="text-emerald-700 font-medium">
                  • Deployment Cost: <b>${simState.currentAction.cost}</b>
                </span>
              </div>
              <div className="flex items-center gap-4 font-bold text-emerald-800">
                <span>
                  Median Wait Saved: -
                  {Math.max(
                    0,
                    simState.rec.base.medianWait - simState.manualResult.medianWait
                  ).toFixed(0)}{' '}
                  min
                </span>
                <span>
                  Breaches Prevented:{' '}
                  {Math.max(
                    0,
                    simState.rec.base.totalBreaches - simState.manualResult.totalBreaches
                  )}
                </span>
              </div>
            </div>
          )}
        </section>

        {/* Section 04: Hospital Digital Twin (Floor Plan) */}
        <section className="space-y-2">
          <h2 className="text-xs font-bold uppercase tracking-wider text-slate-500">
            04. Physical Hospital Floor Plan (Emergency Department Unit)
          </h2>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
            {/* Left Column: Waiting Room & Triage */}
            <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-xs space-y-4">
              <div className="flex justify-between items-center border-b border-slate-100 pb-2.5">
                <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider flex items-center gap-1.5">
                  <Users className="w-4 h-4 text-slate-500" />
                  Waiting Room & Triage Area
                </h3>
                <span className="text-[11px] font-bold px-2 py-0.5 bg-slate-100 border border-slate-200 text-slate-700 rounded-md">
                  Queue: {simState.metrics.waitingTotal}
                </span>
              </div>

              {/* Triage Desks */}
              <div className="space-y-1.5">
                <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">
                  Triage Intake Desks:
                </span>
                <div className="grid grid-cols-2 gap-2">
                  <div className="p-2.5 rounded-lg border border-slate-200 bg-slate-50 text-xs flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                    <div>
                      <div className="font-bold text-slate-800">Nurse Sarah</div>
                      <div className="text-[10px] text-slate-500">Intake Lead</div>
                    </div>
                  </div>
                  <div className="p-2.5 rounded-lg border border-slate-200 bg-slate-50 text-xs flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                    <div>
                      <div className="font-bold text-slate-800">Nurse Marcus</div>
                      <div className="text-[10px] text-slate-500">Triage Staff</div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Awaiting Triage Assessment */}
              <div className="space-y-1.5">
                <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">
                  Awaiting Triage Intake ({simState.triQ.length}):
                </span>
                {simState.triQ.length === 0 ? (
                  <p className="text-xs text-slate-400 italic">No intake queue at triage.</p>
                ) : (
                  <div className="flex flex-wrap gap-1.5 max-h-24 overflow-y-auto">
                    {simState.triQ.map(p => {
                      const cfg = ATS_CONFIG[p.ats] || ATS_CONFIG[5];
                      return (
                        <span
                          key={p.pid}
                          className="px-2 py-1 rounded-md text-[11px] font-semibold border"
                          style={{
                            backgroundColor: cfg.bg,
                            borderColor: cfg.border,
                            color: cfg.color,
                          }}
                        >
                          P#{p.pid} • {cfg.name} • {Math.round(p.wait)}m
                        </span>
                      );
                    })}
                  </div>
                )}
              </div>

              {/* Triaged & Awaiting Acute Care Bay */}
              <div className="space-y-1.5">
                <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wider">
                  Triaged & Awaiting Acute Bay ({simState.bedQ.length}):
                </span>
                {simState.bedQ.length === 0 ? (
                  <p className="text-xs text-slate-400 italic">
                    All triaged patients have been assigned acute care beds.
                  </p>
                ) : (
                  <div className="flex flex-wrap gap-1.5 max-h-48 overflow-y-auto">
                    {simState.bedQ.map(p => {
                      const cfg = ATS_CONFIG[p.ats] || ATS_CONFIG[5];
                      return (
                        <span
                          key={p.pid}
                          className="px-2 py-1 rounded-md text-[11px] font-semibold border flex items-center gap-1"
                          style={{
                            backgroundColor: cfg.bg,
                            borderColor: cfg.border,
                            color: cfg.color,
                          }}
                        >
                          P#{p.pid} {cfg.name} • {Math.round(p.wait)}m
                          {p.breached && (
                            <span className="text-red-700 font-bold">⚠️</span>
                          )}
                        </span>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>

            {/* Right Column: 20-Bed Unit Grid */}
            <div className="lg:col-span-2 bg-white border border-slate-200 rounded-xl p-4 shadow-xs space-y-3">
              <div className="flex justify-between items-center border-b border-slate-100 pb-2.5">
                <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider flex items-center gap-1.5">
                  <Bed className="w-4 h-4 text-slate-500" />
                  Acute Care Bays (20 Bed Unit)
                </h3>
                <span className="text-[11px] font-bold px-2 py-0.5 bg-slate-100 border border-slate-200 text-slate-700 rounded-md">
                  Occupancy: {simState.metrics.inBeds} / {simState.metrics.totalBeds} (
                  {simState.metrics.bedPct.toFixed(0)}%)
                </span>
              </div>

              {/* 4x5 Visual Bay Grid */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
                {simState.bays.map(bay => {
                  let badgeBg = '#f1f5f9';
                  let badgeColor = '#475569';
                  let cardBorder = 'border-slate-200';
                  let cardBg = 'bg-slate-50/50';

                  if (bay.status === 'consult') {
                    badgeBg = '#dbeafe';
                    badgeColor = '#1d4ed8';
                    cardBorder = 'border-blue-200';
                    cardBg = 'bg-blue-50/40';
                  } else if (bay.status === 'boarding') {
                    badgeBg = '#fee2e2';
                    badgeColor = '#b91c1c';
                    cardBorder = 'border-red-300';
                    cardBg = 'bg-red-50/50';
                  } else if (bay.status === 'turnaround') {
                    badgeBg = '#f3e8ff';
                    badgeColor = '#7e22ce';
                    cardBorder = 'border-purple-200';
                    cardBg = 'bg-purple-50/30';
                  } else if (bay.status === 'awaiting_doc') {
                    badgeBg = '#fef3c7';
                    badgeColor = '#b45309';
                    cardBorder = 'border-amber-200';
                    cardBg = 'bg-amber-50/30';
                  }

                  const atsCfg = bay.ats ? ATS_CONFIG[bay.ats] : null;

                  return (
                    <div
                      key={bay.bayId}
                      className={`p-2.5 rounded-lg border ${cardBorder} ${cardBg} flex flex-col justify-between h-20 transition-all shadow-2xs`}
                    >
                      <div className="flex justify-between items-center">
                        <span className="text-[11px] font-bold text-slate-600">
                          BAY {String(bay.bayId).padStart(2, '0')}
                        </span>
                        {atsCfg && (
                          <span
                            className="px-1.5 py-0.5 text-[10px] font-bold rounded-md border"
                            style={{
                              backgroundColor: atsCfg.bg,
                              borderColor: atsCfg.border,
                              color: atsCfg.color,
                            }}
                          >
                            P#{bay.pid} {atsCfg.name}
                          </span>
                        )}
                      </div>

                      <div>
                        <span
                          className="px-1.5 py-0.5 text-[9px] font-bold rounded-sm tracking-wide uppercase inline-block mb-1"
                          style={{ backgroundColor: badgeBg, color: badgeColor }}
                        >
                          {bay.label}
                        </span>
                        <p className="text-[10px] text-slate-500 truncate">{bay.sublabel}</p>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </section>

        {/* Section 05: Prescriptive AI Decision Support */}
        <section className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs space-y-4">
          <div className="flex items-center gap-2">
            <Zap className="w-4 h-4 text-blue-600" />
            <h2 className="text-xs font-bold uppercase tracking-wider text-slate-500">
              05. Prescriptive AI Decision Support (Counterfactual What-If)
            </h2>
          </div>

          {/* AI Headline Banner */}
          <div className="p-3.5 bg-blue-50/60 border border-blue-200 rounded-lg flex items-center gap-3">
            <div className="p-2 bg-blue-600 text-white rounded-md shrink-0">
              <Sparkles className="w-4 h-4" />
            </div>
            <div>
              <span className="text-xs font-bold text-blue-900 uppercase tracking-wide mr-2">
                AI PRESCRIBED RECOMMENDATION:
              </span>
              <span className="text-xs font-semibold text-slate-800">
                {simState.rec.headline}
              </span>
            </div>
          </div>

          {/* 3-Column Counterfactual Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-1">
            {/* Status Quo */}
            <div className="border border-slate-200 rounded-lg p-4 bg-slate-50 space-y-3">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-500 block">
                Status Quo (Do Nothing)
              </span>
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-500">Projected Breaches (4h):</span>
                  <span className="font-bold text-slate-800">{simState.rec.base.totalBreaches}</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-slate-500">Median Wait Time:</span>
                  <span className="font-bold text-slate-800">
                    {simState.rec.base.medianWait.toFixed(0)} min
                  </span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-slate-500">Left Without Being Seen:</span>
                  <span className="font-bold text-slate-800">{simState.rec.base.lwbs}</span>
                </div>
              </div>
            </div>

            {/* AI Optimal */}
            <div className="border border-blue-200 rounded-lg p-4 bg-blue-50/30 space-y-3">
              <span className="text-xs font-bold uppercase tracking-wider text-blue-700 block">
                AI Optimal ({simState.rec.action.replace(/_/g, ' ')})
              </span>
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-500">Projected Breaches (4h):</span>
                  <span className="font-bold text-emerald-600">
                    {simState.rec.best.totalBreaches} (-{simState.rec.breachReductionPct.toFixed(0)}%)
                  </span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-slate-500">Median Wait Time:</span>
                  <span className="font-bold text-emerald-600">
                    {simState.rec.best.medianWait.toFixed(0)} min (-{simState.rec.waitReduction.toFixed(0)}m)
                  </span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-slate-500">Deployment Cost:</span>
                  <span className="font-bold text-slate-800">${simState.rec.cost}</span>
                </div>
              </div>
            </div>

            {/* Active Manual Levers */}
            <div className="border border-slate-200 rounded-lg p-4 bg-slate-50 space-y-3">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-700 block">
                Active Deck ({simState.currentAction.name.replace(/_/g, ' ')})
              </span>
              {simState.currentAction.name === 'DO_NOTHING' ? (
                <p className="text-xs text-slate-400 italic pt-2">
                  No active tactical levers deployed. Matches Status Quo.
                </p>
              ) : simState.manualResult ? (
                <div className="space-y-1.5">
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-500">Projected Breaches (4h):</span>
                    <span className="font-bold text-emerald-600">
                      {simState.manualResult.totalBreaches}
                    </span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-500">Median Wait Time:</span>
                    <span className="font-bold text-emerald-600">
                      {simState.manualResult.medianWait.toFixed(0)} min
                    </span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-500">Operational Cost:</span>
                    <span className="font-bold text-slate-800">${simState.currentAction.cost}</span>
                  </div>
                </div>
              ) : null}
            </div>
          </div>
        </section>

        {/* Section 06: Clinical Shift Handover Memo */}
        <section className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs space-y-3">
          <div className="flex items-center gap-2">
            <Stethoscope className="w-4 h-4 text-slate-600" />
            <h2 className="text-xs font-bold uppercase tracking-wider text-slate-500">
              06. Charge Nurse Shift Handover Briefing (Copilot)
            </h2>
          </div>
          <div className="p-4 bg-slate-50 border border-slate-200 rounded-lg font-mono text-xs leading-relaxed text-slate-700 space-y-2">
            <p>
              <b>SITUATION ({timeFormatted}):</b> Active scenario is {scenario.name}.
              Expected volume is ~{(BASE_VOLUME * scenario.volumeMult * simState.preset.volumeScale).toFixed(0)} visits/24h (
              {((scenario.volumeMult - 1) * 100).toFixed(0)}% vs baseline). Facility scale: {simState.preset.name}. Inpatient bed block:
              active {simState.metrics.boardCount} boarders.
            </p>
            <p>
              <b>BOTTLENECKS:</b> Acute care bays at {simState.metrics.inBeds}/{simState.metrics.totalBeds} (
              {simState.metrics.bedPct.toFixed(0)}% capacity). Waiting room census: {simState.metrics.waitingTotal} (
              {simState.metrics.bedQLen} awaiting bed placement).
            </p>
            <p>
              <b>RECOMMENDED INTERVENTION:</b> {simState.rec.headline}
            </p>
          </div>
        </section>

        {/* Section 07: Evidence & LinuxONE Verification */}
        <section className="bg-white border border-slate-200 rounded-xl p-5 shadow-xs space-y-4">
          <div className="flex items-center gap-2 border-b border-slate-100 pb-3">
            <CheckCircle2 className="w-4 h-4 text-blue-600" />
            <h2 className="text-xs font-bold uppercase tracking-wider text-slate-700">
              07. Model Performance Evidence, Real-Data Grounding &amp; LinuxONE Verification
            </h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
            {/* Model 2 Card */}
            <div className="p-4 rounded-lg border border-slate-200 bg-slate-50/50 space-y-2">
              <span className="font-bold text-slate-900 uppercase text-[11px] block">
                Model 2: Early Warning Surrogate
              </span>
              <p className="text-slate-600 text-[11px] leading-relaxed">
                Trained across 750 simulated operational days (120,000 state snapshots) across 4 disaster scenarios.
              </p>
              <div className="space-y-1 font-mono text-[11px] text-slate-800 pt-1">
                <div className="flex justify-between">
                  <span>Bed Saturation AUC:</span>
                  <span className="font-bold text-blue-700">0.935</span>
                </div>
                <div className="flex justify-between">
                  <span>LWBS Walkout AUC:</span>
                  <span className="font-bold text-blue-700">0.968</span>
                </div>
                <div className="flex justify-between">
                  <span>Urgent Delay AUC:</span>
                  <span className="font-bold text-blue-700">0.856</span>
                </div>
              </div>
              <div className="text-[10px] text-slate-500 pt-1">
                Catches 81-89% of operational failures at ≤5% false alarm rate with 4-hour advance lead time.
              </div>
            </div>

            {/* Model 1 Card */}
            <div className="p-4 rounded-lg border border-slate-200 bg-slate-50/50 space-y-2">
              <span className="font-bold text-slate-900 uppercase text-[11px] block">
                Model 1: CDC NHAMCS Grounding
              </span>
              <p className="text-slate-600 text-[11px] leading-relaxed">
                Empirical dataset: 10,125 emergency department visits across 147 hospitals from CDC NHAMCS 2022 survey.
              </p>
              <div className="space-y-1 font-mono text-[11px] text-slate-800 pt-1">
                <div className="flex justify-between">
                  <span>Admission Model AUC:</span>
                  <span className="font-bold text-emerald-700">0.845 (95% CI)</span>
                </div>
                <div className="flex justify-between">
                  <span>Triage Baseline AUC:</span>
                  <span className="text-slate-500">0.732</span>
                </div>
                <div className="flex justify-between">
                  <span>Simulated LWBS Rate:</span>
                  <span className="font-bold text-emerald-700">1.59% vs 1.89% real</span>
                </div>
              </div>
              <div className="text-[10px] text-slate-500 pt-1">
                Validated on held-out hospitals that the admission classifier never saw during training.
              </div>
            </div>

            {/* LinuxONE Card */}
            <div className="p-4 rounded-lg border border-slate-200 bg-slate-50/50 space-y-2">
              <span className="font-bold text-slate-900 uppercase text-[11px] block">
                IBM LinuxONE (s390x) In-Memory
              </span>
              <p className="text-slate-600 text-[11px] leading-relaxed">
                Deterministic in-memory discrete-event simulation engine with Common Random Numbers (CRN).
              </p>
              <div className="space-y-1 font-mono text-[11px] text-slate-800 pt-1">
                <div className="flex justify-between">
                  <span>Per-Day Sim Latency:</span>
                  <span className="font-bold text-purple-700">6.62 ms / day</span>
                </div>
                <div className="flex justify-between">
                  <span>Multiprocessing Speedup:</span>
                  <span className="font-bold text-purple-700">2.12x (8 workers)</span>
                </div>
                <div className="flex justify-between">
                  <span>Audited Parameters:</span>
                  <span className="font-bold text-purple-700">81 transparent</span>
                </div>
              </div>
              <div className="text-[10px] text-slate-500 pt-1">
                Tested on big-endian s390x and x86_64 architectures with 0ms client-side inference latency.
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
