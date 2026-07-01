# Interview walkthrough

## 90-second version

I worked on a private event-driven project where a Python runtime and a C# platform adapter had to agree about connection state, commands, acknowledgements, and recovered external state. A transport could appear connected while the actual command lifecycle was incomplete, so I treated readiness, admission, execution, and terminal evidence as separate checks.

The reliability work centered on command IDs, fail-closed correlation, reconnect handling, bounded queues, stale-state detection, explicit stop controls, and structured event ledgers. This public repository recreates those patterns with generic commands, synthetic data, and automated tests. It demonstrates the design without releasing the private strategy, research, configuration, or production integration.

## Questions this project can answer

- Why is a healthy socket not proof that work completed?
- How do command IDs prevent duplicate side effects after retries?
- Why should reconnects create a new generation?
- How does exact receipt correlation prevent false success?
- Why must recovery be reconcile-only?
- What should happen when a queue reaches capacity?
- What evidence would you require before declaring the system healthy?

## Honest boundary

Describe the reliability requirements, debugging process, validation, and operating decisions you can personally explain. Do not claim that this demonstration is the full private project or disclose private strategy details.
