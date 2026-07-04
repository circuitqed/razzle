import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.lazybrains.knightball',
  appName: 'KnightBall',
  webDir: 'dist',
  ios: {
    contentInset: 'never',
  },
};

export default config;
