fastlane documentation
----

# Installation

Make sure you have the latest version of the Xcode command line tools installed:

```sh
xcode-select --install
```

For _fastlane_ installation instructions, see [Installing _fastlane_](https://docs.fastlane.tools/#installing-fastlane)

# Available Actions

## iOS

### ios register

```sh
[bundle exec] fastlane ios register
```

Register the bundle ID (Developer Portal) + verify ASC app record exists (idempotent)

### ios listapps

```sh
[bundle exec] fastlane ios listapps
```

List all ASC app records (debug helper)

### ios certs

```sh
[bundle exec] fastlane ios certs
```

Fetch (or create on first run) signing certs + profiles via match

### ios beta

```sh
[bundle exec] fastlane ios beta
```

Build the web assets, sync Capacitor, archive, and upload to TestFlight

----

This README.md is auto-generated and will be re-generated every time [_fastlane_](https://fastlane.tools) is run.

More information about _fastlane_ can be found on [fastlane.tools](https://fastlane.tools).

The documentation of _fastlane_ can be found on [docs.fastlane.tools](https://docs.fastlane.tools).
