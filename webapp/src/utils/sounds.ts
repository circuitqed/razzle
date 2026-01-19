// Sound effect generator using Web Audio API
// No external audio files needed - generates sounds programmatically

let audioContext: AudioContext | null = null;
let soundEnabled = true;

function getAudioContext(): AudioContext {
  if (!audioContext) {
    audioContext = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
  }
  return audioContext;
}

export function setSoundEnabled(enabled: boolean) {
  soundEnabled = enabled;
}

export function isSoundEnabled(): boolean {
  return soundEnabled;
}

// Play a simple tone
function playTone(frequency: number, duration: number, type: OscillatorType = 'sine', volume = 0.3) {
  if (!soundEnabled) return;

  try {
    const ctx = getAudioContext();

    // Resume context if suspended (browsers require user interaction first)
    if (ctx.state === 'suspended') {
      ctx.resume();
    }

    const oscillator = ctx.createOscillator();
    const gainNode = ctx.createGain();

    oscillator.connect(gainNode);
    gainNode.connect(ctx.destination);

    oscillator.frequency.value = frequency;
    oscillator.type = type;

    // Fade in and out for smoother sound
    const now = ctx.currentTime;
    gainNode.gain.setValueAtTime(0, now);
    gainNode.gain.linearRampToValueAtTime(volume, now + 0.01);
    gainNode.gain.exponentialRampToValueAtTime(0.01, now + duration);

    oscillator.start(now);
    oscillator.stop(now + duration);
  } catch {
    // Silently fail if audio isn't available
  }
}

// Piece movement sound - short click
export function playMoveSound() {
  playTone(800, 0.08, 'square', 0.15);
}

// Ball pass sound - softer whoosh-like
export function playPassSound() {
  if (!soundEnabled) return;

  try {
    const ctx = getAudioContext();
    if (ctx.state === 'suspended') ctx.resume();

    const now = ctx.currentTime;

    // Create white noise for whoosh effect
    const bufferSize = ctx.sampleRate * 0.15;
    const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
    const data = buffer.getChannelData(0);

    for (let i = 0; i < bufferSize; i++) {
      data[i] = (Math.random() * 2 - 1) * (1 - i / bufferSize);
    }

    const source = ctx.createBufferSource();
    source.buffer = buffer;

    const filter = ctx.createBiquadFilter();
    filter.type = 'bandpass';
    filter.frequency.value = 1000;
    filter.Q.value = 0.5;

    const gain = ctx.createGain();
    gain.gain.setValueAtTime(0.1, now);
    gain.gain.exponentialRampToValueAtTime(0.01, now + 0.15);

    source.connect(filter);
    filter.connect(gain);
    gain.connect(ctx.destination);

    source.start(now);
  } catch {
    // Silently fail
  }
}

// Win sound - triumphant ascending tones
export function playWinSound() {
  playTone(523.25, 0.15, 'sine', 0.25); // C5
  setTimeout(() => playTone(659.25, 0.15, 'sine', 0.25), 100); // E5
  setTimeout(() => playTone(783.99, 0.3, 'sine', 0.3), 200); // G5
}

// Lose sound - descending tones
export function playLoseSound() {
  playTone(392, 0.2, 'sine', 0.2); // G4
  setTimeout(() => playTone(349.23, 0.2, 'sine', 0.2), 150); // F4
  setTimeout(() => playTone(293.66, 0.3, 'sine', 0.25), 300); // D4
}

// Select piece sound - subtle pop
export function playSelectSound() {
  playTone(600, 0.05, 'sine', 0.1);
}

// End turn sound - soft confirmation
export function playEndTurnSound() {
  playTone(440, 0.1, 'triangle', 0.15);
  setTimeout(() => playTone(550, 0.1, 'triangle', 0.15), 80);
}

// Error/invalid move sound - low buzz
export function playErrorSound() {
  playTone(200, 0.15, 'sawtooth', 0.1);
}
