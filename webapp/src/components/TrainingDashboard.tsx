import { useState, useEffect, useCallback } from 'react';
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
  BarChart,
  Bar,
} from 'recharts';

const API_BASE = import.meta.env.VITE_API_URL || '/api';

interface TrainingIteration {
  iteration: number;
  timestamp: string;
  num_games: number;
  p1_wins: number;
  p2_wins: number;
  draws: number;
  avg_game_length: number;
  min_game_length: number;
  max_game_length: number;
  std_game_length: number;
  training_examples: number;
  selfplay_time_sec: number;
  training_time_sec: number;
  total_time_sec: number;
  final_loss: number;
  final_policy_loss: number;
  final_value_loss: number;
  gpu_memory_used_mb: number;
  gpu_memory_total_mb: number;
  gpu_utilization_pct: number;
  cpu_percent: number;
  device: string;
  win_rate_vs_random: number | null;
  elo_rating: number | null;
}

interface TrainingStatus {
  status: string;
  run_id: string | null;
  start_time: string | null;
  total_games: number;
  total_examples: number;
  total_time_sec: number;
  iterations: TrainingIteration[];
  config: Record<string, unknown>;
}

interface Props {
  onClose?: () => void;
  refreshInterval?: number; // in milliseconds
}

export default function TrainingDashboard({ onClose, refreshInterval = 10000 }: Props) {
  const [data, setData] = useState<TrainingStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [isPaused, setIsPaused] = useState(false);

  const fetchData = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE}/training/status`);
      if (!response.ok) throw new Error('Failed to fetch training status');
      const result = await response.json();
      setData(result);
      setError(null);
      setLastUpdate(new Date());
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

  const formatTime = (seconds: number): string => {
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`;
    return `${(seconds / 3600).toFixed(1)}h`;
  };

  if (error && !data) {
    return (
      <div className="fixed inset-0 bg-gray-900 bg-opacity-95 z-50 overflow-auto p-4">
        <div className="max-w-6xl mx-auto">
          <div className="flex justify-between items-center mb-6">
            <h1 className="text-2xl font-bold text-white">Training Dashboard</h1>
            {onClose && (
              <button onClick={onClose} className="text-gray-400 hover:text-white text-2xl">
                ×
              </button>
            )}
          </div>
          <div className="bg-red-900 text-red-200 p-4 rounded-lg">
            Error: {error}
            <p className="text-sm mt-2">Make sure the engine server is running.</p>
          </div>
        </div>
      </div>
    );
  }

  if (!data || data.status === 'no_training') {
    return (
      <div className="fixed inset-0 bg-gray-900 bg-opacity-95 z-50 overflow-auto p-4">
        <div className="max-w-6xl mx-auto">
          <div className="flex justify-between items-center mb-6">
            <h1 className="text-2xl font-bold text-white">Training Dashboard</h1>
            {onClose && (
              <button onClick={onClose} className="text-gray-400 hover:text-white text-2xl">
                ×
              </button>
            )}
          </div>
          <div className="bg-gray-800 text-gray-300 p-8 rounded-lg text-center">
            <p className="text-xl mb-4">No training data available</p>
            <p className="text-sm text-gray-500">
              Start training with: <code className="bg-gray-700 px-2 py-1 rounded">python scripts/train_local.py</code>
            </p>
          </div>
        </div>
      </div>
    );
  }

  const iterations = data.iterations;
  const latest = iterations.length > 0 ? iterations[iterations.length - 1] : null;

  // Prepare chart data
  const chartData = iterations.map((it) => ({
    iteration: it.iteration,
    avgLength: it.avg_game_length,
    minLength: it.min_game_length,
    maxLength: it.max_game_length,
    loss: it.final_loss,
    policyLoss: it.final_policy_loss,
    valueLoss: it.final_value_loss,
    p1Wins: it.p1_wins,
    p2Wins: it.p2_wins,
    draws: it.draws,
    examples: it.training_examples,
    selfplayTime: it.selfplay_time_sec,
    trainTime: it.training_time_sec,
    gpuMemory: it.gpu_memory_used_mb,
    gpuUtil: it.gpu_utilization_pct,
    cpuUtil: it.cpu_percent,
  }));

  // Check if we have GPU data
  const hasGpuData = iterations.some(it => it.gpu_memory_total_mb > 0);

  return (
    <div className="fixed inset-0 bg-gray-900 bg-opacity-95 z-50 overflow-auto p-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex justify-between items-center mb-6">
          <div>
            <h1 className="text-2xl font-bold text-white">Training Dashboard</h1>
            <p className="text-sm text-gray-400">
              Run: {data.run_id} | Started: {data.start_time?.split('T')[0]}
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
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <StatCard label="Iterations" value={iterations.length} />
          <StatCard label="Total Games" value={data.total_games.toLocaleString()} />
          <StatCard label="Total Examples" value={data.total_examples.toLocaleString()} />
          <StatCard label="Total Time" value={formatTime(data.total_time_sec)} />
        </div>

        {/* Latest Iteration */}
        {latest && (
          <div className="bg-gray-800 rounded-lg p-4 mb-6">
            <h2 className="text-lg font-semibold text-white mb-3">
              Latest: Iteration {latest.iteration}
              {latest.device !== 'cpu' && (
                <span className="ml-2 text-xs px-2 py-1 bg-green-600 rounded">
                  {latest.device.toUpperCase()}
                </span>
              )}
            </h2>
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-4 text-sm">
              <div>
                <span className="text-gray-400">Win Rate:</span>
                <span className="ml-2 text-white">
                  P1 {latest.p1_wins} / P2 {latest.p2_wins} / Draw {latest.draws}
                </span>
              </div>
              <div>
                <span className="text-gray-400">Game Length:</span>
                <span className="ml-2 text-green-400">
                  {latest.avg_game_length.toFixed(1)} avg
                  {latest.min_game_length > 0 && (
                    <span className="text-gray-500 text-xs ml-1">
                      ({latest.min_game_length}-{latest.max_game_length})
                    </span>
                  )}
                </span>
              </div>
              <div>
                <span className="text-gray-400">Loss:</span>
                <span className="ml-2 text-blue-400">
                  {latest.final_loss.toFixed(4)}
                </span>
              </div>
              <div>
                <span className="text-gray-400">Time:</span>
                <span className="ml-2 text-white">
                  {formatTime(latest.total_time_sec)}
                </span>
              </div>
              {(latest.gpu_memory_total_mb > 0 || latest.cpu_percent > 0) && (
                <div>
                  <span className="text-gray-400">Hardware:</span>
                  <span className="ml-2 text-yellow-400">
                    {latest.gpu_memory_total_mb > 0 ? (
                      <>GPU {latest.gpu_utilization_pct.toFixed(0)}% / {(latest.gpu_memory_used_mb / 1024).toFixed(1)}GB</>
                    ) : (
                      <>CPU {latest.cpu_percent.toFixed(0)}%</>
                    )}
                  </span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Charts */}
        {chartData.length > 1 && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Game Length Chart */}
            <div className="bg-gray-800 rounded-lg p-4">
              <h3 className="text-md font-semibold text-white mb-4">
                Game Length (Early Learning Indicator)
              </h3>
              <ResponsiveContainer width="100%" height={250}>
                <AreaChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="iteration" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1F2937', border: 'none' }}
                    labelStyle={{ color: '#fff' }}
                  />
                  <Legend />
                  <Area
                    type="monotone"
                    dataKey="maxLength"
                    stackId="1"
                    stroke="#4ADE80"
                    fill="#4ADE8020"
                    name="Max"
                  />
                  <Area
                    type="monotone"
                    dataKey="avgLength"
                    stackId="2"
                    stroke="#22C55E"
                    fill="#22C55E40"
                    name="Average"
                  />
                  <Area
                    type="monotone"
                    dataKey="minLength"
                    stackId="3"
                    stroke="#16A34A"
                    fill="#16A34A60"
                    name="Min"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* Loss Chart */}
            <div className="bg-gray-800 rounded-lg p-4">
              <h3 className="text-md font-semibold text-white mb-4">Training Loss</h3>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="iteration" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1F2937', border: 'none' }}
                    labelStyle={{ color: '#fff' }}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="loss"
                    stroke="#3B82F6"
                    strokeWidth={2}
                    dot={false}
                    name="Total Loss"
                  />
                  <Line
                    type="monotone"
                    dataKey="policyLoss"
                    stroke="#EF4444"
                    strokeWidth={1}
                    dot={false}
                    name="Policy Loss"
                  />
                  <Line
                    type="monotone"
                    dataKey="valueLoss"
                    stroke="#22C55E"
                    strokeWidth={1}
                    dot={false}
                    name="Value Loss"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Win Distribution Chart */}
            <div className="bg-gray-800 rounded-lg p-4">
              <h3 className="text-md font-semibold text-white mb-4">Win Distribution</h3>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="iteration" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1F2937', border: 'none' }}
                    labelStyle={{ color: '#fff' }}
                  />
                  <Legend />
                  <Bar dataKey="p1Wins" fill="#3B82F6" name="P1 Wins" stackId="wins" />
                  <Bar dataKey="p2Wins" fill="#EF4444" name="P2 Wins" stackId="wins" />
                  <Bar dataKey="draws" fill="#6B7280" name="Draws" stackId="wins" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Time per Iteration */}
            <div className="bg-gray-800 rounded-lg p-4">
              <h3 className="text-md font-semibold text-white mb-4">Time per Iteration</h3>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="iteration" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1F2937', border: 'none' }}
                    labelStyle={{ color: '#fff' }}
                    formatter={(value: number) => `${value.toFixed(1)}s`}
                  />
                  <Legend />
                  <Bar dataKey="selfplayTime" fill="#F59E0B" name="Self-play" stackId="time" />
                  <Bar dataKey="trainTime" fill="#10B981" name="Training" stackId="time" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* GPU/CPU Utilization (if available) */}
            {hasGpuData && (
              <div className="bg-gray-800 rounded-lg p-4">
                <h3 className="text-md font-semibold text-white mb-4">Hardware Utilization</h3>
                <ResponsiveContainer width="100%" height={250}>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="iteration" stroke="#9CA3AF" />
                    <YAxis stroke="#9CA3AF" domain={[0, 100]} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#1F2937', border: 'none' }}
                      labelStyle={{ color: '#fff' }}
                      formatter={(value: number, name: string) => {
                        if (name === 'GPU Memory') return `${(value / 1024).toFixed(2)} GB`;
                        return `${value.toFixed(1)}%`;
                      }}
                    />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="gpuUtil"
                      stroke="#10B981"
                      strokeWidth={2}
                      dot={false}
                      name="GPU %"
                    />
                    <Line
                      type="monotone"
                      dataKey="cpuUtil"
                      stroke="#3B82F6"
                      strokeWidth={2}
                      dot={false}
                      name="CPU %"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* GPU Memory Usage (if available) */}
            {hasGpuData && (
              <div className="bg-gray-800 rounded-lg p-4">
                <h3 className="text-md font-semibold text-white mb-4">GPU Memory Usage</h3>
                <ResponsiveContainer width="100%" height={250}>
                  <AreaChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="iteration" stroke="#9CA3AF" />
                    <YAxis
                      stroke="#9CA3AF"
                      tickFormatter={(value) => `${(value / 1024).toFixed(1)}GB`}
                    />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#1F2937', border: 'none' }}
                      labelStyle={{ color: '#fff' }}
                      formatter={(value: number) => `${(value / 1024).toFixed(2)} GB`}
                    />
                    <Legend />
                    <Area
                      type="monotone"
                      dataKey="gpuMemory"
                      stroke="#F59E0B"
                      fill="#F59E0B40"
                      name="GPU Memory"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
        )}

        {/* Config */}
        {Object.keys(data.config).length > 0 && (
          <div className="mt-6 bg-gray-800 rounded-lg p-4">
            <h3 className="text-md font-semibold text-white mb-3">Configuration</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm text-gray-400">
              {Object.entries(data.config).map(([key, value]) => (
                <div key={key}>
                  <span className="text-gray-500">{key}:</span>{' '}
                  <span className="text-white">{String(value)}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Footer */}
        <div className="mt-6 text-center text-xs text-gray-500">
          Last updated: {lastUpdate?.toLocaleTimeString() || 'Never'} |{' '}
          {isPaused ? 'Paused' : `Auto-refresh every ${refreshInterval / 1000}s`}
        </div>
      </div>
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <div className="text-sm text-gray-400">{label}</div>
      <div className="text-2xl font-bold text-white">{value}</div>
    </div>
  );
}
