# Apple Developer Support ticket — Sign in with Apple "Sign Up Not Completed"

**Team ID:** 5J862PN572
**App:** KnightBall — App ID 6787309741, bundle com.lazybrains.knightball

## Problem
Sign in with Apple fails on a real device (and simulator) with "Sign Up Not
Completed" **after** Face ID authentication succeeds. The native
ASAuthorizationController sheet presents correctly; the failure is returned
from Apple's servers after the user authenticates. No error reaches our code
beyond a generic authorization failure. Reproduces for the account holder and
every tester, first-time and repeat, whether or not Hide My Email is chosen.

## Configuration verified correct
- App ID has the "Sign in with Apple" capability enabled (Primary App consent).
  Deleted + re-created the capability; no change.
- Provisioning profiles regenerated after every capability change (via match).
- Shipped .ipa entitlements (codesign -d --entitlements) contain:
  application-identifier 5J862PN572.com.lazybrains.knightball,
  com.apple.developer.team-identifier 5J862PN572,
  com.apple.developer.applesignin = [Default].
- No pending Apple Developer Program License Agreement (checked developer
  portal + App Store Connect + Agreements/Tax/Banking).
- Apple system status: Sign in with Apple + Apple Account services all green.
- Not listed under Settings > Apple ID > Sign-In & Security > Sign in with
  Apple on the test device (no stale association to remove).
- Native flow only (no Service ID / web); app uses the bundle ID directly.

## Ask
Please check the server-side rejection reason for Sign in with Apple sign-up
requests for App ID com.lazybrains.knightball (Team 5J862PN572). Everything
configurable on our side is verified correct; the rejection is only visible in
Apple's backend.
