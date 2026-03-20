import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  AreaChart,
  Area,
  ReferenceLine,
} from 'recharts';
import { getAllTrainingMetrics, getDashboardData } from '../api/metrics';
import { getArenaRatings, getArenaMatches, getUniqueSimulationCounts, filterRatingsBySimulations, sortRatingsByElo, sortRatingsByIteration } from '../api/arena';
import { getLeaderboard } from '../api/leaderboard';
import type { TrainingMetrics, TrainingDashboardData, ArenaRating, ArenaMatch, PlayerProfile } from '../types';

interface Props {
  onClose?: () => void;
  refreshInterval?: number;
}

type TabType = 'overview' | 'policy' | 'value' | 'pass' | 'loss' | 'infra' | 'arena' | 'leaderboard';

interface InfraSnapshot {
  timestamp: number;
  games_pending: number;
  games_total: number;
  active_workers: number;
  games_per_hour: number;  // Raw measurement
  games_per_hour_filtered: number;  // Kalman filtered estimate
}

// Simple 1D Kalman filter state
interface KalmanState {
  estimate: number;  // Current estimate of games/hour
  errorCovariance: number;  // Uncertainty in estimate
}

export default function TrainingDashboard({ onClose, refreshInterval = 10000 }: Props) {
  const [metrics, setMetrics] = useState<TrainingMetrics[]>([]);
  const [dashboard, setDashboard] = useState<TrainingDashboardData | null>(null);
  const [infraHistory, setInfraHistory] = useState<InfraSnapshot[]>([]);
  const [arenaRatings, setArenaRatings] = useState<ArenaRating[]>([]);
  const [arenaMatches, setArenaMatches] = useState<ArenaMatch[]>([]);
  const [leaderboard, setLeaderboard] = useState<PlayerProfile[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [isPaused, setIsPaused] = useState(false);
  const [activeTab, setActiveTab] = useState<TabType>('overview');
  const [iterationRange, setIterationRange] = useState<[number, number] | null>(null);

  // Kalman filter state for games/hour smoothing
  const kalmanRef = useRef<KalmanState>({
    estimate: 0,
    errorCovariance: 1000,  // High initial uncertainty
  });

  const fetchData = useCallback(async () => {
    try {
      const [metricsData, dashboardData, ratingsData, matchesData, leaderboardData] = await Promise.all([
        getAllTrainingMetrics(),
        getDashboardData(),
        getArenaRatings().catch((e) => { console.error('Arena ratings error:', e); return []; }),
        getArenaMatches().catch((e) => { console.error('Arena matches error:', e); return []; }),
        getLeaderboard(undefined, 100, 0).catch((e) => { console.error('Leaderboard error:', e); return []; }),
      ]);
      setMetrics(metricsData);
      setDashboard(dashboardData);
      setArenaRatings(ratingsData);
      setArenaMatches(matchesData);
      setLeaderboard(leaderboardData);
      setError(null);
      setLastUpdate(new Date());

      // Update iteration range - always track max, set initial min
      if (metricsData.length > 0) {
        const minIter = metricsData[0].iteration;
        const maxIter = metricsData[metricsData.length - 1].iteration;
        setIterationRange(prev => {
          if (!prev) return [minIter, maxIter];
          // Keep user's min selection but update max if it increased
          return [prev[0], Math.max(prev[1], maxIter)];
        });
      }

      // Track infrastructure history
      const activeWorkers = Object.keys(dashboardData.workers).length;
      const now = Date.now();
      setInfraHistory(prev => {
        const newSnapshot: InfraSnapshot = {
          timestamp: now,
          games_pending: dashboardData.games_pending,
          games_total: dashboardData.games_total,
          active_workers: activeWorkers,
          games_per_hour: 0, // Raw measurement
          games_per_hour_filtered: 0, // Kalman filtered
        };

        // Calculate raw games per hour from recent window
        const lastSnapshot = prev.length > 0 ? prev[prev.length - 1] : null;
        let rawGamesPerHour = 0;

        if (lastSnapshot && lastSnapshot.timestamp < now) {
          const timeDiffHours = (now - lastSnapshot.timestamp) / (1000 * 60 * 60);
          const gamesDiff = dashboardData.games_total - lastSnapshot.games_total;
          if (timeDiffHours > 0 && gamesDiff >= 0) {
            rawGamesPerHour = gamesDiff / timeDiffHours;
          }
        }

        newSnapshot.games_per_hour = Math.round(rawGamesPerHour);

        // Apply Kalman filter for smooth estimate
        // Process noise: how much we expect true rate to change between measurements
        const processNoise = 100;  // games/hour variance per update
        // Measurement noise: how noisy each raw measurement is
        const measurementNoise = 2000;  // High because short-term rate is noisy

        const kalman = kalmanRef.current;

        // Prediction step: estimate stays same, uncertainty increases
        const predictedCovariance = kalman.errorCovariance + processNoise;

        // Update step: incorporate new measurement
        const kalmanGain = predictedCovariance / (predictedCovariance + measurementNoise);
        const newEstimate = kalman.estimate + kalmanGain * (rawGamesPerHour - kalman.estimate);
        const newCovariance = (1 - kalmanGain) * predictedCovariance;

        // Update filter state
        kalmanRef.current = {
          estimate: newEstimate,
          errorCovariance: newCovariance,
        };

        newSnapshot.games_per_hour_filtered = Math.round(Math.max(0, newEstimate));

        // Keep last 360 snapshots (1 hour at 10s intervals)
        const history = [...prev, newSnapshot].slice(-360);
        return history;
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    }
  }, []);

  useEffect(() => {
    fetchData();

    if (!isPaused) {
      const interval = setInterval(fetchData, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [fetchData, refreshInterval, isPaused]);

  // Filter metrics by iteration range
  const filteredMetrics = useMemo(() => {
    if (!iterationRange) return metrics;
    return metrics.filter(
      m => m.iteration >= iterationRange[0] && m.iteration <= iterationRange[1]
    );
  }, [metrics, iterationRange]);

  const latest = metrics.length > 0 ? metrics[metrics.length - 1] : null;

  const formatPercent = (v: number | null | undefined): string => {
    if (v == null) return '-';
    return `${(v * 100).toFixed(1)}%`;
  };

  const formatNumber = (v: number | null | undefined, decimals: number = 3): string => {
    if (v == null) return '-';
    return v.toFixed(decimals);
  };

  const formatTime = (seconds: number | null | undefined): string => {
    if (seconds == null) return '-';
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`;
    return `${(seconds / 3600).toFixed(1)}h`;
  };

  // Prepare chart data
  const chartData = useMemo(() => filteredMetrics.map(m => ({
    iteration: m.iteration,
    // Policy
    top1_accuracy: m.policy_top1_accuracy ? m.policy_top1_accuracy * 100 : null,
    top3_accuracy: m.policy_top3_accuracy ? m.policy_top3_accuracy * 100 : null,
    entropy: m.policy_entropy,
    legal_mass: m.policy_legal_mass ? m.policy_legal_mass * 100 : null,
    ebf: m.policy_ebf,
    confidence: m.policy_confidence ? m.policy_confidence * 100 : null,
    // Value
    value_mean: m.value_mean,
    value_std: m.value_std,
    value_extremity: m.value_extremity,
    calibration_error: m.value_calibration_error,
    value_sign_accuracy: m.value_sign_accuracy ? m.value_sign_accuracy * 100 : null,
    value_mse: m.value_mse,
    // Pass
    pass_decision_rate: m.pass_decision_rate ? m.pass_decision_rate * 100 : null,
    // Loss
    loss_total: m.loss_total,
    loss_policy: m.loss_policy,
    loss_value: m.loss_value,
    loss_difficulty: m.loss_difficulty,
    loss_illegal: m.loss_illegal_penalty,
    // Game stats
    num_games: m.num_games,
    num_examples: m.num_examples,
    avg_game_length: m.avg_game_length,
    learning_rate: m.learning_rate,
  })), [filteredMetrics]);

  // Total stats
  const totalGames = metrics.reduce((sum, m) => sum + (m.num_games || 0), 0);
  const totalExamples = metrics.reduce((sum, m) => sum + (m.num_examples || 0), 0);
  const totalTime = metrics.reduce((sum, m) => sum + (m.train_time_sec || 0), 0);

  if (error && metrics.length === 0) {
    return (
      <div className="fixed inset-0 bg-gray-900 bg-opacity-95 z-50 overflow-auto p-4">
        <div className="max-w-6xl mx-auto">
          <DashboardHeader onClose={onClose} title="Training Dashboard" />
          <div className="bg-red-900 text-red-200 p-4 rounded-lg">
            Error: {error}
            <p className="text-sm mt-2">Make sure the engine server is running.</p>
          </div>
        </div>
      </div>
    );
  }

  if (metrics.length === 0) {
    return (
      <div className="fixed inset-0 bg-gray-900 bg-opacity-95 z-50 overflow-auto p-4">
        <div className="max-w-6xl mx-auto">
          <DashboardHeader onClose={onClose} title="Training Dashboard" />
          {dashboard && dashboard.games_total > 0 ? (
            <div className="space-y-4">
              <div className="bg-gray-800 text-gray-300 p-6 rounded-lg text-center">
                <p className="text-xl mb-2">Collecting self-play games...</p>
                <p className="text-3xl font-bold text-blue-400 mb-1">
                  {dashboard.games_pending} / {dashboard.latest_model?.games_trained_on !== undefined ? '512' : '512'} games
                </p>
                <p className="text-sm text-gray-500">
                  Training will begin once enough games are collected.
                </p>
              </div>
              <InfrastructureTab dashboard={dashboard} infraHistory={infraHistory} />
            </div>
          ) : (
            <div className="bg-gray-800 text-gray-300 p-8 rounded-lg text-center">
              <p className="text-xl mb-4">No training data yet</p>
              <p className="text-sm text-gray-500">
                Start distributed training with: <code className="bg-gray-700 px-2 py-1 rounded">python scripts/launch_training.py</code>
              </p>
            </div>
          )}
        </div>
      </div>
    );
  }

  const tabs: { id: TabType; label: string }[] = [
    { id: 'overview', label: 'Overview' },
    { id: 'policy', label: 'Policy' },
    { id: 'value', label: 'Value' },
    { id: 'pass', label: 'Pass Stats' },
    { id: 'loss', label: 'Loss' },
    { id: 'infra', label: 'Infrastructure' },
    { id: 'arena', label: 'Arena' },
    { id: 'leaderboard', label: 'Leaderboard' },
  ];

  return (
    <div className="fixed inset-0 bg-gray-900 bg-opacity-95 z-50 overflow-auto p-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex justify-between items-center mb-4">
          <div>
            <h1 className="text-2xl font-bold text-white">Training Dashboard</h1>
            <p className="text-sm text-gray-400">
              {latest?.model_version && `Model: ${latest.model_version} | `}
              Iteration {latest?.iteration || 0}
            </p>
          </div>
          <div className="flex items-center gap-4">
            <button
              onClick={() => setIsPaused(!isPaused)}
              className={`px-3 py-1 rounded text-sm ${
                isPaused ? 'bg-green-600 hover:bg-green-700' : 'bg-yellow-600 hover:bg-yellow-700'
              }`}
            >
              {isPaused ? '▶ Resume' : '⏸ Pause'}
            </button>
            <button
              onClick={fetchData}
              className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-sm"
            >
              ↻ Refresh
            </button>
            {onClose && (
              <button onClick={onClose} className="text-gray-400 hover:text-white text-2xl">
                ×
              </button>
            )}
          </div>
        </div>

        {/* Summary Stats */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-4">
          <StatCard label="Iterations" value={metrics.length} />
          <StatCard label="Total Games" value={totalGames.toLocaleString()} />
          <StatCard label="Total Examples" value={totalExamples.toLocaleString()} />
          <StatCard label="Total Time" value={formatTime(totalTime)} />
          <StatCard label="Learning Rate" value={latest?.learning_rate?.toExponential(1) || '-'} />
        </div>

        {/* Latest Iteration Summary */}
        {latest && (
          <div className="bg-gray-800 rounded-lg p-3 mb-4">
            <h2 className="text-md font-semibold text-white mb-2">
              Latest: Iteration {latest.iteration}
            </h2>
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3 text-sm">
              <div>
                <span className="text-gray-400">Loss:</span>
                <span className="ml-2 text-blue-400">{formatNumber(latest.loss_total, 4)}</span>
              </div>
              <div>
                <span className="text-gray-400">Top-1 Acc:</span>
                <span className="ml-2 text-green-400">{formatPercent(latest.policy_top1_accuracy)}</span>
              </div>
              <div>
                <span className="text-gray-400">EBF:</span>
                <span className="ml-2 text-yellow-400">{formatNumber(latest.policy_ebf, 1)}</span>
              </div>
              <div>
                <span className="text-gray-400">Value Std:</span>
                <span className="ml-2 text-purple-400">{formatNumber(latest.value_std)}</span>
              </div>
              <div>
                <span className="text-gray-400">Calibration:</span>
                <span className="ml-2 text-orange-400">{formatNumber(latest.value_calibration_error, 4)}</span>
              </div>
              <div>
                <span className="text-gray-400">Pass Rate:</span>
                <span className="ml-2 text-pink-400">{formatPercent(latest.pass_decision_rate)}</span>
              </div>
            </div>
          </div>
        )}

        {/* Iteration Range Selector */}
        {metrics.length > 1 && iterationRange && (
          <div className="bg-gray-800 rounded-lg p-3 mb-4">
            <div className="flex items-center gap-4">
              <span className="text-gray-400 text-sm">Iteration Range:</span>
              <input
                type="range"
                min={metrics[0].iteration}
                max={metrics[metrics.length - 1].iteration}
                value={iterationRange[0]}
                onChange={(e) => setIterationRange([parseInt(e.target.value), iterationRange[1]])}
                className="w-24"
              />
              <span className="text-white text-sm">{iterationRange[0]}</span>
              <span className="text-gray-400">to</span>
              <span className="text-white text-sm">{iterationRange[1]}</span>
              <input
                type="range"
                min={metrics[0].iteration}
                max={metrics[metrics.length - 1].iteration}
                value={iterationRange[1]}
                onChange={(e) => setIterationRange([iterationRange[0], parseInt(e.target.value)])}
                className="w-24"
              />
              <button
                onClick={() => setIterationRange([metrics[0].iteration, metrics[metrics.length - 1].iteration])}
                className="px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded text-xs"
              >
                Reset
              </button>
            </div>
          </div>
        )}

        {/* Tabs */}
        <div className="flex gap-1 mb-4">
          {tabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-4 py-2 rounded-t text-sm font-medium ${
                activeTab === tab.id
                  ? 'bg-gray-800 text-white'
                  : 'bg-gray-700 text-gray-400 hover:text-white'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Tab Content */}
        <div className="bg-gray-800 rounded-lg rounded-tl-none p-4">
          {activeTab === 'overview' && <OverviewTab data={chartData} />}
          {activeTab === 'policy' && <PolicyTab data={chartData} />}
          {activeTab === 'value' && <ValueTab data={chartData} />}
          {activeTab === 'pass' && <PassTab data={chartData} />}
          {activeTab === 'loss' && <LossTab data={chartData} />}
          {activeTab === 'infra' && <InfrastructureTab dashboard={dashboard} infraHistory={infraHistory} />}
          {activeTab === 'arena' && <ArenaTab ratings={arenaRatings} matches={arenaMatches} />}
          {activeTab === 'leaderboard' && <LeaderboardTab players={leaderboard} />}
        </div>

        {/* Footer */}
        <div className="mt-4 text-center text-xs text-gray-500">
          Last updated: {lastUpdate?.toLocaleTimeString() || 'Never'} |{' '}
          {isPaused ? 'Paused' : `Auto-refresh every ${refreshInterval / 1000}s`}
        </div>
      </div>
    </div>
  );
}

// Dashboard Header Component
function DashboardHeader({ onClose, title }: { onClose?: () => void; title: string }) {
  return (
    <div className="flex justify-between items-center mb-6">
      <h1 className="text-2xl font-bold text-white">{title}</h1>
      {onClose && (
        <button onClick={onClose} className="text-gray-400 hover:text-white text-2xl">
          ×
        </button>
      )}
    </div>
  );
}

// Stat Card Component
function StatCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="bg-gray-700 rounded-lg p-3">
      <div className="text-xs text-gray-400">{label}</div>
      <div className="text-lg font-bold text-white">{value}</div>
    </div>
  );
}

// Chart wrapper component
function ChartCard({ title, children }: { title: string; children: React.ReactElement }) {
  return (
    <div className="bg-gray-700 rounded-lg p-4">
      <h3 className="text-sm font-semibold text-white mb-3">{title}</h3>
      <ResponsiveContainer width="100%" height={220}>
        {children}
      </ResponsiveContainer>
    </div>
  );
}

// Tooltip style for charts
const tooltipStyle = {
  contentStyle: { backgroundColor: '#1F2937', border: 'none', borderRadius: '8px' },
  labelStyle: { color: '#fff' },
};

// Overview Tab
function OverviewTab({ data }: { data: any[] }) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <ChartCard title="Training Loss">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="iteration" stroke="#9CA3AF" type="number" domain={['dataMin', 'dataMax']} />
          <YAxis stroke="#9CA3AF" />
          <Tooltip {...tooltipStyle} />
          <Legend />
          <Line type="monotone" dataKey="loss_total" stroke="#3B82F6" strokeWidth={2} dot={false} name="Total" />
          <Line type="monotone" dataKey="loss_policy" stroke="#EF4444" strokeWidth={1} dot={false} name="Policy" />
          <Line type="monotone" dataKey="loss_value" stroke="#22C55E" strokeWidth={1} dot={false} name="Value" />
        </LineChart>
      </ChartCard>

      <ChartCard title="Policy Accuracy (%)">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="iteration" stroke="#9CA3AF" type="number" domain={['dataMin', 'dataMax']} />
          <YAxis stroke="#9CA3AF" domain={[0, 100]} />
          <Tooltip {...tooltipStyle} formatter={(v: number) => `${v?.toFixed(1)}%`} />
          <Legend />
          <Line type="monotone" dataKey="top1_accuracy" stroke="#22C55E" strokeWidth={2} dot={false} name="Top-1" />
          <Line type="monotone" dataKey="top3_accuracy" stroke="#3B82F6" strokeWidth={2} dot={false} name="Top-3" />
        </LineChart>
      </ChartCard>

      <ChartCard title="Effective Branching Factor">
        <AreaChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="iteration" stroke="#9CA3AF" type="number" domain={['dataMin', 'dataMax']} />
          <YAxis stroke="#9CA3AF" />
          <Tooltip {...tooltipStyle} formatter={(v: number) => v?.toFixed(2)} />
          <Area type="monotone" dataKey="ebf" stroke="#F59E0B" fill="#F59E0B40" name="EBF" />
        </AreaChart>
      </ChartCard>

      <ChartCard title="Value Calibration Error">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="iteration" stroke="#9CA3AF" type="number" domain={['dataMin', 'dataMax']} />
          <YAxis stroke="#9CA3AF" />
          <Tooltip {...tooltipStyle} formatter={(v: number) => v?.toFixed(4)} />
          <Line type="monotone" dataKey="calibration_error" stroke="#EC4899" strokeWidth={2} dot={false} name="Error" />
        </LineChart>
      </ChartCard>
    </div>
  );
}

// Policy Tab
function PolicyTab({ data }: { data: any[] }) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <ChartCard title="Policy Accuracy (%)">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="iteration" stroke="#9CA3AF" type="number" domain={['dataMin', 'dataMax']} />
          <YAxis stroke="#9CA3AF" domain={[0, 100]} />
          <Tooltip {...tooltipStyle} formatter={(v: number) => `${v?.toFixed(1)}%`} />
          <Legend />
          <Line type="monotone" dataKey="top1_accuracy" stroke="#22C55E" strokeWidth={2} dot={false} name="Top-1" />
          <Line type="monotone" dataKey="top3_accuracy" stroke="#3B82F6" strokeWidth={2} dot={false} name="Top-3" />
        </LineChart>
      </ChartCard>

      <ChartCard title="Policy Entropy">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="iteration" stroke="#9CA3AF" type="number" domain={['dataMin', 'dataMax']} />
          <YAxis stroke="#9CA3AF" />
          <Tooltip {...tooltipStyle} formatter={(v: number) => v?.toFixed(3)} />
          <Line type="monotone" dataKey="entropy" stroke="#8B5CF6" strokeWidth={2} dot={false} name="Entropy" />
        </LineChart>
      </ChartCard>

      <ChartCard title="Legal Move Mass (%)">
        <AreaChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="iteration" stroke="#9CA3AF" type="number" domain={['dataMin', 'dataMax']} />
          <YAxis stroke="#9CA3AF" domain={[0, 100]} />
          <Tooltip {...tooltipStyle} formatter={(v: number) => `${v?.toFixed(1)}%`} />
          <Area type="monotone" dataKey="legal_mass" stroke="#10B981" fill="#10B98140" name="Legal Mass" />
        </AreaChart>
      </ChartCard>

      <ChartCard title="Effective Branching Factor">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="iteration" stroke="#9CA3AF" type="number" domain={['dataMin', 'dataMax']} />
          <YAxis stroke="#9CA3AF" />
          <Tooltip {...tooltipStyle} formatter={(v: number) => v?.toFixed(2)} />
          <Line type="monotone" dataKey="ebf" stroke="#F59E0B" strokeWidth={2} dot={false} name="EBF" />
        </LineChart>
      </ChartCard>

      <ChartCard title="Policy Confidence (%)">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="iteration" stroke="#9CA3AF" type="number" domain={['dataMin', 'dataMax']} />
          <YAxis stroke="#9CA3AF" domain={[0, 100]} />
          <Tooltip {...tooltipStyle} formatter={(v: number) => `${v?.toFixed(1)}%`} />
          <Line type="monotone" dataKey="confidence" stroke="#06B6D4" strokeWidth={2} dot={false} name="Confidence" />
        </LineChart>
      </ChartCard>
    </div>
  );
}

// Value Tab
function ValueTab({ data }: { data: any[] }) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <ChartCard title="Sign Accuracy (%) - Key Validation Metric">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="iteration" stroke="#9CA3AF" type="number" domain={['dataMin', 'dataMax']} />
          <YAxis stroke="#9CA3AF" domain={[40, 60]} />
          <Tooltip {...tooltipStyle} formatter={(v: number) => `${v?.toFixed(1)}%`} />
          <ReferenceLine y={50} stroke="#6B7280" strokeDasharray="5 5" label={{ value: 'Random', fill: '#6B7280', fontSize: 10 }} />
          <Line type="monotone" dataKey="value_sign_accuracy" stroke="#22C55E" strokeWidth={2} dot={false} name="Sign Accuracy" />
        </LineChart>
      </ChartCard>

      <ChartCard title="Value MSE (Validation)">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="iteration" stroke="#9CA3AF" type="number" domain={['dataMin', 'dataMax']} />
          <YAxis stroke="#9CA3AF" />
          <Tooltip {...tooltipStyle} formatter={(v: number) => v?.toFixed(4)} />
          <Line type="monotone" dataKey="value_mse" stroke="#3B82F6" strokeWidth={2} dot={false} name="MSE" />
        </LineChart>
      </ChartCard>

      <ChartCard title="Value Predictions">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="iteration" stroke="#9CA3AF" type="number" domain={['dataMin', 'dataMax']} />
          <YAxis stroke="#9CA3AF" domain={[-1, 1]} />
          <Tooltip {...tooltipStyle} formatter={(v: number) => v?.toFixed(3)} />
          <Legend />
          <Line type="monotone" dataKey="value_mean" stroke="#3B82F6" strokeWidth={2} dot={false} name="Mean" />
        </LineChart>
      </ChartCard>

      <ChartCard title="Value Extremity">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="iteration" stroke="#9CA3AF" type="number" domain={['dataMin', 'dataMax']} />
          <YAxis stroke="#9CA3AF" domain={[0, 1]} />
          <Tooltip {...tooltipStyle} formatter={(v: number) => v?.toFixed(3)} />
          <Line type="monotone" dataKey="value_extremity" stroke="#F59E0B" strokeWidth={2} dot={false} name="Extremity" />
        </LineChart>
      </ChartCard>

      <ChartCard title="Calibration Error">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="iteration" stroke="#9CA3AF" type="number" domain={['dataMin', 'dataMax']} />
          <YAxis stroke="#9CA3AF" />
          <Tooltip {...tooltipStyle} formatter={(v: number) => v?.toFixed(4)} />
          <Line type="monotone" dataKey="calibration_error" stroke="#EC4899" strokeWidth={2} dot={false} name="Error" />
        </LineChart>
      </ChartCard>

      <ChartCard title="Value Standard Deviation">
        <AreaChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="iteration" stroke="#9CA3AF" type="number" domain={['dataMin', 'dataMax']} />
          <YAxis stroke="#9CA3AF" />
          <Tooltip {...tooltipStyle} formatter={(v: number) => v?.toFixed(3)} />
          <Area type="monotone" dataKey="value_std" stroke="#8B5CF6" fill="#8B5CF640" name="Std Dev" />
        </AreaChart>
      </ChartCard>
    </div>
  );
}

// Pass Tab
function PassTab({ data }: { data: any[] }) {
  const hasPassData = data.some(d => d.pass_decision_rate != null);

  if (!hasPassData) {
    return (
      <div className="text-center text-gray-400 py-8">
        <p>No pass statistics available yet.</p>
        <p className="text-sm mt-2">Pass decision rate will be computed from game data during training.</p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <ChartCard title="Pass Decision Rate (%)">
        <AreaChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="iteration" stroke="#9CA3AF" type="number" domain={['dataMin', 'dataMax']} />
          <YAxis stroke="#9CA3AF" domain={[0, 100]} />
          <Tooltip {...tooltipStyle} formatter={(v: number) => `${v?.toFixed(1)}%`} />
          <Area type="monotone" dataKey="pass_decision_rate" stroke="#EC4899" fill="#EC489940" name="Pass Rate" />
        </AreaChart>
      </ChartCard>

      <ChartCard title="Average Game Length">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="iteration" stroke="#9CA3AF" type="number" domain={['dataMin', 'dataMax']} />
          <YAxis stroke="#9CA3AF" />
          <Tooltip {...tooltipStyle} formatter={(v: number) => v?.toFixed(1)} />
          <Line type="monotone" dataKey="avg_game_length" stroke="#10B981" strokeWidth={2} dot={false} name="Moves" />
        </LineChart>
      </ChartCard>

      <ChartCard title="Games per Iteration">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="iteration" stroke="#9CA3AF" type="number" domain={['dataMin', 'dataMax']} />
          <YAxis stroke="#9CA3AF" />
          <Tooltip {...tooltipStyle} />
          <Line type="monotone" dataKey="num_games" stroke="#6366F1" strokeWidth={2} dot={false} name="Games" />
        </LineChart>
      </ChartCard>

      <ChartCard title="Examples per Iteration">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="iteration" stroke="#9CA3AF" type="number" domain={['dataMin', 'dataMax']} />
          <YAxis stroke="#9CA3AF" />
          <Tooltip {...tooltipStyle} />
          <Line type="monotone" dataKey="num_examples" stroke="#F59E0B" strokeWidth={2} dot={false} name="Examples" />
        </LineChart>
      </ChartCard>
    </div>
  );
}

// Loss Tab
function LossTab({ data }: { data: any[] }) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <ChartCard title="Total Loss">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="iteration" stroke="#9CA3AF" type="number" domain={['dataMin', 'dataMax']} />
          <YAxis stroke="#9CA3AF" />
          <Tooltip {...tooltipStyle} formatter={(v: number) => v?.toFixed(4)} />
          <Line type="monotone" dataKey="loss_total" stroke="#3B82F6" strokeWidth={2} dot={false} name="Total" />
        </LineChart>
      </ChartCard>

      <ChartCard title="Loss Components">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="iteration" stroke="#9CA3AF" type="number" domain={['dataMin', 'dataMax']} />
          <YAxis stroke="#9CA3AF" />
          <Tooltip {...tooltipStyle} formatter={(v: number) => v?.toFixed(4)} />
          <Legend />
          <Line type="monotone" dataKey="loss_policy" stroke="#EF4444" strokeWidth={2} dot={false} name="Policy" />
          <Line type="monotone" dataKey="loss_value" stroke="#22C55E" strokeWidth={2} dot={false} name="Value" />
        </LineChart>
      </ChartCard>

      <ChartCard title="Difficulty Loss">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="iteration" stroke="#9CA3AF" type="number" domain={['dataMin', 'dataMax']} />
          <YAxis stroke="#9CA3AF" />
          <Tooltip {...tooltipStyle} formatter={(v: number) => v?.toFixed(4)} />
          <Line type="monotone" dataKey="loss_difficulty" stroke="#F59E0B" strokeWidth={2} dot={false} name="Difficulty" />
        </LineChart>
      </ChartCard>

      <ChartCard title="Illegal Move Penalty">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="iteration" stroke="#9CA3AF" type="number" domain={['dataMin', 'dataMax']} />
          <YAxis stroke="#9CA3AF" />
          <Tooltip {...tooltipStyle} formatter={(v: number) => v?.toFixed(4)} />
          <Line type="monotone" dataKey="loss_illegal" stroke="#EC4899" strokeWidth={2} dot={false} name="Illegal Penalty" />
        </LineChart>
      </ChartCard>

      <ChartCard title="Learning Rate">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="iteration" stroke="#9CA3AF" type="number" domain={['dataMin', 'dataMax']} />
          <YAxis stroke="#9CA3AF" tickFormatter={(v) => v.toExponential(0)} />
          <Tooltip {...tooltipStyle} formatter={(v: number) => v?.toExponential(2)} />
          <Line type="monotone" dataKey="learning_rate" stroke="#6366F1" strokeWidth={2} dot={false} name="LR" />
        </LineChart>
      </ChartCard>
    </div>
  );
}

// Infrastructure Tab
function InfrastructureTab({
  dashboard,
  infraHistory
}: {
  dashboard: TrainingDashboardData | null;
  infraHistory: InfraSnapshot[]
}) {
  if (!dashboard) {
    return (
      <div className="text-center text-gray-400 py-8">
        <p>Loading infrastructure data...</p>
      </div>
    );
  }

  // Filter workers: only show those seen in the last 5 minutes
  const WORKER_EXPIRY_MS = 5 * 60 * 1000; // 5 minutes
  const now = Date.now();
  const allWorkers = Object.entries(dashboard.workers);
  const activeWorkers = allWorkers.filter(([, w]) => {
    const lastSeen = new Date(w.last_seen).getTime();
    return (now - lastSeen) < WORKER_EXPIRY_MS;
  });
  const staleWorkers = allWorkers.length - activeWorkers.length;
  const totalWorkerGames = activeWorkers.reduce((sum, [, w]) => sum + w.games, 0);

  // Get current games per hour (Kalman filtered for stable display)
  const currentGamesPerHour = infraHistory.length > 0
    ? infraHistory[infraHistory.length - 1].games_per_hour_filtered
    : 0;

  // Prepare chart data for infraHistory
  const infraChartData = infraHistory.map((snap, idx) => ({
    time: idx,
    games_pending: snap.games_pending,
    games_total: snap.games_total,
    active_workers: snap.active_workers,
    games_per_hour: snap.games_per_hour,  // Raw for reference
    games_per_hour_filtered: snap.games_per_hour_filtered,  // Smoothed
  }));

  const formatTimestamp = (ts: string) => {
    const date = new Date(ts);
    return date.toLocaleTimeString();
  };

  const formatTimeAgo = (ts: string) => {
    const now = Date.now();
    const then = new Date(ts).getTime();
    const diffSec = Math.floor((now - then) / 1000);
    // Handle negative values (clock drift between server and client)
    if (diffSec < 0) return 'just now';
    if (diffSec < 60) return `${diffSec}s ago`;
    if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m ago`;
    return `${Math.floor(diffSec / 3600)}h ago`;
  };

  return (
    <div className="space-y-4">
      {/* Summary Stats */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <div className="bg-gray-700 rounded-lg p-3">
          <div className="text-xs text-gray-400">Status</div>
          <div className={`text-lg font-bold ${dashboard.status === 'active' ? 'text-green-400' : 'text-gray-400'}`}>
            {dashboard.status === 'active' ? 'Active' : 'Idle'}
          </div>
        </div>
        <div className="bg-gray-700 rounded-lg p-3">
          <div className="text-xs text-gray-400">Games/Hour</div>
          <div className="text-lg font-bold text-blue-400">{currentGamesPerHour}</div>
        </div>
        <div className="bg-gray-700 rounded-lg p-3">
          <div className="text-xs text-gray-400">Queue Size</div>
          <div className="text-lg font-bold text-yellow-400">{dashboard.games_pending}</div>
        </div>
        <div className="bg-gray-700 rounded-lg p-3">
          <div className="text-xs text-gray-400">Total Games</div>
          <div className="text-lg font-bold text-white">{dashboard.games_total}</div>
        </div>
        <div className="bg-gray-700 rounded-lg p-3">
          <div className="text-xs text-gray-400">Active Workers</div>
          <div className="text-lg font-bold text-purple-400">{activeWorkers.length}</div>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {infraChartData.length > 1 && (
          <>
            <ChartCard title="Games Per Hour (Kalman Filtered)">
              <AreaChart data={infraChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="time" stroke="#9CA3AF" />
                <YAxis stroke="#9CA3AF" />
                <Tooltip {...tooltipStyle} formatter={(v: number) => `${v} games/hr`} labelFormatter={() => ''} />
                <Area type="monotone" dataKey="games_per_hour_filtered" stroke="#3B82F6" fill="#3B82F640" name="Filtered" />
              </AreaChart>
            </ChartCard>

            <ChartCard title="Training Queue">
              <AreaChart data={infraChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="time" stroke="#9CA3AF" />
                <YAxis stroke="#9CA3AF" />
                <Tooltip {...tooltipStyle} formatter={(v: number) => `${v} games`} labelFormatter={() => ''} />
                <Area type="monotone" dataKey="games_pending" stroke="#F59E0B" fill="#F59E0B40" name="Pending" />
              </AreaChart>
            </ChartCard>
          </>
        )}
      </div>

      {/* Workers Table */}
      {activeWorkers.length > 0 && (
        <div className="bg-gray-700 rounded-lg p-4">
          <h3 className="text-sm font-semibold text-white mb-3">
            Active Workers ({activeWorkers.length})
            {staleWorkers > 0 && <span className="text-gray-500 font-normal ml-2">({staleWorkers} stale)</span>}
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-400 text-left">
                  <th className="pb-2 pr-4">Worker ID</th>
                  <th className="pb-2 pr-4">Games</th>
                  <th className="pb-2">Last Seen</th>
                </tr>
              </thead>
              <tbody>
                {activeWorkers.sort((a, b) => b[1].games - a[1].games).map(([workerId, worker]) => (
                  <tr key={workerId} className="text-gray-300 border-t border-gray-600">
                    <td className="py-2 pr-4 font-mono text-xs">{workerId}</td>
                    <td className="py-2 pr-4">{worker.games}</td>
                    <td className="py-2 text-gray-400">{formatTimeAgo(worker.last_seen)}</td>
                  </tr>
                ))}
              </tbody>
              <tfoot>
                <tr className="text-white border-t border-gray-500 font-semibold">
                  <td className="pt-2 pr-4">Total</td>
                  <td className="pt-2 pr-4">{totalWorkerGames}</td>
                  <td className="pt-2"></td>
                </tr>
              </tfoot>
            </table>
          </div>
        </div>
      )}

      {/* Models Table */}
      {dashboard.models.length > 0 && (
        <div className="bg-gray-700 rounded-lg p-4">
          <h3 className="text-sm font-semibold text-white mb-3">Models ({dashboard.models.length})</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-400 text-left">
                  <th className="pb-2 pr-4">Version</th>
                  <th className="pb-2 pr-4">Iteration</th>
                  <th className="pb-2 pr-4">Games Trained</th>
                  <th className="pb-2 pr-4">Final Loss</th>
                  <th className="pb-2">Created</th>
                </tr>
              </thead>
              <tbody>
                {dashboard.models.slice().reverse().map((model) => (
                  <tr key={model.version} className="text-gray-300 border-t border-gray-600">
                    <td className="py-2 pr-4 font-mono text-xs">{model.version}</td>
                    <td className="py-2 pr-4">{model.iteration}</td>
                    <td className="py-2 pr-4">{model.games_trained_on ?? '-'}</td>
                    <td className="py-2 pr-4">{model.final_loss?.toFixed(4) ?? '-'}</td>
                    <td className="py-2 text-gray-400">{formatTimestamp(model.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Empty state */}
      {activeWorkers.length === 0 && dashboard.models.length === 0 && (
        <div className="text-center text-gray-400 py-8">
          <p>No workers or models yet.</p>
          <p className="text-sm mt-2">Start distributed training to see infrastructure metrics.</p>
        </div>
      )}
    </div>
  );
}

// Arena Tab
function ArenaTab({
  ratings,
  matches,
}: {
  ratings: ArenaRating[];
  matches: ArenaMatch[];
}) {
  const [selectedSims, setSelectedSims] = useState<number | 'all'>('all');

  // Get unique simulation counts
  const simCounts = useMemo(() => getUniqueSimulationCounts(ratings), [ratings]);

  // Filter ratings by selected simulation count
  const filteredRatings = useMemo(() => {
    if (selectedSims === 'all') return ratings;
    return filterRatingsBySimulations(ratings, selectedSims);
  }, [ratings, selectedSims]);

  // Sort by ELO for rankings table
  const rankedRatings = useMemo(() => sortRatingsByElo(filteredRatings), [filteredRatings]);

  // Sort by iteration for chart
  const chartRatings = useMemo(() => sortRatingsByIteration(filteredRatings), [filteredRatings]);

  // Prepare chart data - group by simulation count for multi-line chart
  const chartData = useMemo(() => {
    if (selectedSims !== 'all') {
      // Single line
      return chartRatings.map(r => ({
        iteration: r.iteration,
        elo: r.elo_rating,
        version: r.model_version,
      }));
    }

    // Multiple lines - one per simulation count
    // Group ratings by iteration
    const byIteration: Record<number, Record<number, number>> = {};
    for (const r of ratings) {
      if (!byIteration[r.iteration]) {
        byIteration[r.iteration] = {};
      }
      byIteration[r.iteration][r.simulations] = r.elo_rating;
    }

    return Object.entries(byIteration)
      .map(([iter, sims]) => ({
        iteration: parseInt(iter),
        ...Object.fromEntries(
          Object.entries(sims).map(([s, elo]) => [`elo_${s}`, elo])
        ),
      }))
      .sort((a, b) => a.iteration - b.iteration);
  }, [selectedSims, chartRatings, ratings]);

  // Filter matches by selected simulation count
  const filteredMatches = useMemo(() => {
    if (selectedSims === 'all') return matches.slice(0, 20);
    return matches.filter(m => m.simulations === selectedSims).slice(0, 20);
  }, [matches, selectedSims]);

  // Colors for different simulation counts
  const simColors: Record<number, string> = {
    400: '#EF4444',
    800: '#3B82F6',
    1600: '#22C55E',
    3200: '#F59E0B',
  };

  if (ratings.length === 0) {
    return (
      <div className="text-center text-gray-400 py-8">
        <p className="text-xl mb-4">No arena data available</p>
        <p className="text-sm">
          Run the arena tournament to measure model strength:
          <code className="bg-gray-700 px-2 py-1 rounded ml-2">
            python scripts/run_arena.py --stride 10 --games 20
          </code>
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Simulation Count Selector */}
      <div className="flex items-center gap-4">
        <span className="text-gray-400 text-sm">Search Depth:</span>
        <select
          value={selectedSims}
          onChange={(e) => setSelectedSims(e.target.value === 'all' ? 'all' : parseInt(e.target.value))}
          className="bg-gray-700 text-white rounded px-3 py-1 text-sm"
        >
          <option value="all">All Depths</option>
          {simCounts.map(s => (
            <option key={s} value={s}>{s} simulations</option>
          ))}
        </select>
        <span className="text-gray-500 text-xs">
          ({rankedRatings.length} models)
        </span>
      </div>

      {/* ELO Chart */}
      <ChartCard title={`ELO Rating Over Iterations${selectedSims !== 'all' ? ` (${selectedSims} sims)` : ''}`}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="iteration" stroke="#9CA3AF" type="number" domain={['dataMin', 'dataMax']} />
          <YAxis stroke="#9CA3AF" domain={['auto', 'auto']} />
          <Tooltip {...tooltipStyle} formatter={(v: number) => v?.toFixed(0)} />
          <Legend />
          <ReferenceLine y={1000} stroke="#6B7280" strokeDasharray="5 5" label={{ value: 'Anchor', fill: '#6B7280', fontSize: 10 }} />
          {selectedSims !== 'all' ? (
            <Line
              type="monotone"
              dataKey="elo"
              stroke="#3B82F6"
              strokeWidth={2}
              dot={true}
              name="ELO"
            />
          ) : (
            simCounts.map(s => (
              <Line
                key={s}
                type="monotone"
                dataKey={`elo_${s}`}
                stroke={simColors[s] || '#9CA3AF'}
                strokeWidth={2}
                dot={false}
                name={`${s} sims`}
                connectNulls
              />
            ))
          )}
        </LineChart>
      </ChartCard>

      {/* Rankings Table */}
      <div className="bg-gray-700 rounded-lg p-4">
        <h3 className="text-sm font-semibold text-white mb-3">
          ELO Rankings {selectedSims !== 'all' && `(${selectedSims} sims)`}
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-gray-400 text-left">
                <th className="pb-2 pr-4">Rank</th>
                <th className="pb-2 pr-4">Model</th>
                <th className="pb-2 pr-4">Iteration</th>
                {selectedSims === 'all' && <th className="pb-2 pr-4">Sims</th>}
                <th className="pb-2 pr-4">ELO</th>
                <th className="pb-2">Games</th>
              </tr>
            </thead>
            <tbody>
              {rankedRatings.slice(0, 20).map((rating, idx) => (
                <tr key={`${rating.model_version}-${rating.simulations}`} className="text-gray-300 border-t border-gray-600">
                  <td className="py-2 pr-4 text-gray-500">{idx + 1}</td>
                  <td className="py-2 pr-4 font-mono text-xs">{rating.model_version}</td>
                  <td className="py-2 pr-4">{rating.iteration}</td>
                  {selectedSims === 'all' && <td className="py-2 pr-4">{rating.simulations}</td>}
                  <td className="py-2 pr-4 font-semibold" style={{ color: rating.elo_rating >= 1000 ? '#22C55E' : '#EF4444' }}>
                    {rating.elo_rating.toFixed(0)}
                  </td>
                  <td className="py-2">{rating.games_played}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Recent Matches Table */}
      {filteredMatches.length > 0 && (
        <div className="bg-gray-700 rounded-lg p-4">
          <h3 className="text-sm font-semibold text-white mb-3">
            Recent Matches {selectedSims !== 'all' && `(${selectedSims} sims)`}
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-400 text-left">
                  <th className="pb-2 pr-4">Model 1</th>
                  <th className="pb-2 pr-4">Result</th>
                  <th className="pb-2 pr-4">Model 2</th>
                  {selectedSims === 'all' && <th className="pb-2 pr-4">Sims</th>}
                  <th className="pb-2">Win Rate</th>
                </tr>
              </thead>
              <tbody>
                {filteredMatches.map((match, idx) => {
                  const total = match.model1_wins + match.model2_wins + match.draws;
                  const winRate1 = total > 0 ? (match.model1_wins / total * 100).toFixed(0) : '-';
                  return (
                    <tr key={idx} className="text-gray-300 border-t border-gray-600">
                      <td className="py-2 pr-4 font-mono text-xs">{match.model1_version}</td>
                      <td className="py-2 pr-4">
                        <span className={match.model1_wins > match.model2_wins ? 'text-green-400' : 'text-gray-400'}>
                          {match.model1_wins}
                        </span>
                        <span className="text-gray-500"> - </span>
                        <span className={match.model2_wins > match.model1_wins ? 'text-green-400' : 'text-gray-400'}>
                          {match.model2_wins}
                        </span>
                        {match.draws > 0 && (
                          <span className="text-gray-500 ml-1">({match.draws}d)</span>
                        )}
                      </td>
                      <td className="py-2 pr-4 font-mono text-xs">{match.model2_version}</td>
                      {selectedSims === 'all' && <td className="py-2 pr-4">{match.simulations}</td>}
                      <td className="py-2 text-gray-400">{winRate1}%</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

// Leaderboard Tab - Unified human/AI player rankings
function LeaderboardTab({ players }: { players: PlayerProfile[] }) {
  const [filterType, setFilterType] = useState<'all' | 'human' | 'ai'>('all');

  // Filter players by type
  const filteredPlayers = useMemo(() => {
    if (filterType === 'all') return players;
    return players.filter(p => p.player_type === filterType);
  }, [players, filterType]);

  // Separate stats
  const humanCount = players.filter(p => p.player_type === 'human').length;
  const aiCount = players.filter(p => p.player_type === 'ai').length;
  const topElo = players.length > 0 ? Math.max(...players.map(p => p.elo_rating)) : 0;
  const avgElo = players.length > 0
    ? players.reduce((sum, p) => sum + p.elo_rating, 0) / players.length
    : 0;

  // Format player name based on type
  const formatPlayerName = (player: PlayerProfile): string => {
    if (player.player_type === 'ai') {
      const modelPart = player.model_version || 'unknown';
      const simsPart = player.simulations ? ` (${player.simulations}s)` : '';
      return `${modelPart}${simsPart}`;
    }
    return player.username || player.display_name;
  };

  // Get ELO color
  const getEloColor = (elo: number): string => {
    if (elo >= 1200) return 'text-yellow-400';  // Gold
    if (elo >= 1100) return 'text-purple-400';  // Purple
    if (elo >= 1000) return 'text-green-400';   // Green
    if (elo >= 900) return 'text-blue-400';     // Blue
    return 'text-gray-400';                      // Gray
  };

  // Get player type badge
  const getTypeBadge = (type: string) => {
    if (type === 'human') {
      return <span className="px-1.5 py-0.5 bg-blue-600 text-blue-100 rounded text-xs">Human</span>;
    }
    return <span className="px-1.5 py-0.5 bg-purple-600 text-purple-100 rounded text-xs">AI</span>;
  };

  if (players.length === 0) {
    return (
      <div className="text-center text-gray-400 py-8">
        <p className="text-xl mb-4">No players yet</p>
        <p className="text-sm">
          Play some games to see the leaderboard!
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Summary Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="bg-gray-700 rounded-lg p-3">
          <div className="text-xs text-gray-400">Total Players</div>
          <div className="text-lg font-bold text-white">{players.length}</div>
        </div>
        <div className="bg-gray-700 rounded-lg p-3">
          <div className="text-xs text-gray-400">Humans / AIs</div>
          <div className="text-lg font-bold">
            <span className="text-blue-400">{humanCount}</span>
            <span className="text-gray-500"> / </span>
            <span className="text-purple-400">{aiCount}</span>
          </div>
        </div>
        <div className="bg-gray-700 rounded-lg p-3">
          <div className="text-xs text-gray-400">Top ELO</div>
          <div className="text-lg font-bold text-yellow-400">{topElo.toFixed(0)}</div>
        </div>
        <div className="bg-gray-700 rounded-lg p-3">
          <div className="text-xs text-gray-400">Average ELO</div>
          <div className="text-lg font-bold text-gray-300">{avgElo.toFixed(0)}</div>
        </div>
      </div>

      {/* Filter Controls */}
      <div className="flex items-center gap-4">
        <span className="text-gray-400 text-sm">Filter:</span>
        <div className="flex gap-2">
          {(['all', 'human', 'ai'] as const).map(type => (
            <button
              key={type}
              onClick={() => setFilterType(type)}
              className={`px-3 py-1 rounded text-sm ${
                filterType === type
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-400 hover:text-white'
              }`}
            >
              {type === 'all' ? 'All' : type === 'human' ? 'Humans' : 'AIs'}
            </button>
          ))}
        </div>
        <span className="text-gray-500 text-xs ml-2">
          ({filteredPlayers.length} players)
        </span>
      </div>

      {/* Leaderboard Table */}
      <div className="bg-gray-700 rounded-lg p-4">
        <h3 className="text-sm font-semibold text-white mb-3">
          ELO Leaderboard
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-gray-400 text-left">
                <th className="pb-2 pr-4 w-12">Rank</th>
                <th className="pb-2 pr-4">Player</th>
                <th className="pb-2 pr-4 w-20">Type</th>
                <th className="pb-2 pr-4 w-20 text-right">ELO</th>
                <th className="pb-2 w-20 text-right">Games</th>
              </tr>
            </thead>
            <tbody>
              {filteredPlayers.map((player, idx) => (
                <tr key={player.player_id} className="text-gray-300 border-t border-gray-600">
                  <td className="py-2 pr-4 text-gray-500">
                    {idx === 0 && filteredPlayers.length > 2 && <span className="text-yellow-400">1</span>}
                    {idx === 1 && filteredPlayers.length > 2 && <span className="text-gray-300">2</span>}
                    {idx === 2 && filteredPlayers.length > 2 && <span className="text-orange-400">3</span>}
                    {(idx > 2 || filteredPlayers.length <= 2) && idx + 1}
                  </td>
                  <td className="py-2 pr-4">
                    <span className={player.player_type === 'ai' ? 'font-mono text-xs' : ''}>
                      {formatPlayerName(player)}
                    </span>
                  </td>
                  <td className="py-2 pr-4">
                    {getTypeBadge(player.player_type)}
                  </td>
                  <td className={`py-2 pr-4 font-semibold text-right ${getEloColor(player.elo_rating)}`}>
                    {player.elo_rating.toFixed(0)}
                  </td>
                  <td className="py-2 text-right text-gray-400">
                    {player.elo_games_played}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* ELO Distribution Info */}
      <div className="bg-gray-700 rounded-lg p-4 text-sm text-gray-400">
        <h4 className="font-semibold text-white mb-2">ELO Rating Guide</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
          <div><span className="text-yellow-400">1200+</span> - Elite</div>
          <div><span className="text-purple-400">1100-1199</span> - Expert</div>
          <div><span className="text-green-400">1000-1099</span> - Skilled</div>
          <div><span className="text-blue-400">900-999</span> - Intermediate</div>
        </div>
        <p className="mt-2 text-xs">
          All players start at 1000 ELO. Win against stronger opponents to gain more rating.
        </p>
      </div>
    </div>
  );
}
