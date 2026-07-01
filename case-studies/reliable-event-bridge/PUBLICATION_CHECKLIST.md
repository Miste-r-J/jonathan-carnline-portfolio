# GitHub publication checklist

## Suggested repository settings

- Repository name: `reliable-event-bridge`
- Description: `Sanitized reference implementation of fail-closed event integration, idempotency, reconnect safety, correlation, and structured observability.`
- Visibility: Public
- Topics: `python`, `systems-integration`, `reliability-engineering`, `event-driven`, `idempotency`, `observability`, `testing`
- License: None initially; source remains copyrighted unless Jonathan intentionally chooses a license later.

## Before the first push

1. Run `python -m pip install .`.
2. Run `python -m unittest discover -s tests -v`.
3. Run the credential and private-identifier scan used during preparation.
4. Confirm `EVIDENCE.md` still describes the publication boundary accurately.
5. Review the full staged diff before committing.
6. Do not add screenshots or logs taken from the private runtime.

## Recommended first commit

```text
Add sanitized reliable event bridge case study
```

## Portfolio integration after publication

Add the public repository URL to the existing Real-Time Trading Systems Reliability case study using a label such as `Review sanitized engineering proof`. Do not describe the reference implementation as the production repository.
