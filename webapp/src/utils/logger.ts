// Simple logging utility for debugging
// Logs are stored in memory, sent to server, and can be downloaded

interface LogEntry {
  timestamp: string;
  level: 'debug' | 'info' | 'warn' | 'error';
  message: string;
  data?: unknown;
}

class Logger {
  private logs: LogEntry[] = [];
  private pendingLogs: LogEntry[] = [];
  private maxLogs = 1000;
  private sessionId: string;
  private serverUrl = '/api/logs';

  constructor() {
    // Generate a unique session ID
    this.sessionId = `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;

    // Start periodic flush to server
    this.startPeriodicFlush();

    // Flush on page unload
    if (typeof window !== 'undefined') {
      window.addEventListener('beforeunload', () => this.flush());
    }
  }

  private startPeriodicFlush() {
    // Flush every 5 seconds
    setInterval(() => this.flush(), 5000);
  }

  private formatTimestamp(): string {
    return new Date().toISOString();
  }

  private addLog(level: LogEntry['level'], message: string, data?: unknown) {
    const entry: LogEntry = {
      timestamp: this.formatTimestamp(),
      level,
      message,
      data,
    };

    this.logs.push(entry);
    this.pendingLogs.push(entry);

    // Keep only the last maxLogs entries
    if (this.logs.length > this.maxLogs) {
      this.logs = this.logs.slice(-this.maxLogs);
    }

    // Also log to console
    const consoleMethod = level === 'error' ? console.error
      : level === 'warn' ? console.warn
      : level === 'debug' ? console.debug
      : console.log;

    const prefix = `[${entry.timestamp}] [${level.toUpperCase()}]`;
    if (data !== undefined) {
      consoleMethod(prefix, message, data);
    } else {
      consoleMethod(prefix, message);
    }
  }

  // Send pending logs to server
  async flush(): Promise<void> {
    if (this.pendingLogs.length === 0) return;

    const logsToSend = [...this.pendingLogs];
    this.pendingLogs = [];

    try {
      await fetch(this.serverUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          entries: logsToSend.map(e => ({
            timestamp: e.timestamp,
            level: e.level,
            message: e.message,
            data: e.data ? (typeof e.data === 'object' ? e.data : { value: e.data }) : null,
          })),
          session_id: this.sessionId,
        }),
      });
    } catch {
      // If flush fails, add logs back to pending
      this.pendingLogs = [...logsToSend, ...this.pendingLogs];
    }
  }

  debug(message: string, data?: unknown) {
    this.addLog('debug', message, data);
  }

  info(message: string, data?: unknown) {
    this.addLog('info', message, data);
  }

  warn(message: string, data?: unknown) {
    this.addLog('warn', message, data);
  }

  error(message: string, data?: unknown) {
    this.addLog('error', message, data);
  }

  // Get all logs as a string for download
  getLogsAsText(): string {
    return this.logs
      .map(entry => {
        const dataStr = entry.data !== undefined
          ? ` | ${JSON.stringify(entry.data)}`
          : '';
        return `[${entry.timestamp}] [${entry.level.toUpperCase()}] ${entry.message}${dataStr}`;
      })
      .join('\n');
  }

  // Download logs as a file
  downloadLogs() {
    const text = this.getLogsAsText();
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `razzle-logs-${new Date().toISOString().replace(/[:.]/g, '-')}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  // Clear logs
  clear() {
    this.logs = [];
  }

  // Get log count
  get count(): number {
    return this.logs.length;
  }
}

// Singleton instance
export const logger = new Logger();

// Expose to window for debugging
if (typeof window !== 'undefined') {
  (window as unknown as { razzleLogger: Logger }).razzleLogger = logger;
}
