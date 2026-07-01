# Architecture

```mermaid
flowchart LR
    A[Upstream producer] --> B[Admission control]
    B --> C[Bounded command queue]
    C --> D[Downstream adapter]
    D --> E[Correlated receipt]
    E --> F[Terminal state]

    G[Connection identity] --> B
    H[Durable command IDs] --> B
    I[Kill switch] --> B
    J[Recovery snapshot] --> K[Reconciliation only]
    K --> F

    B --> L[(Structured event ledger)]
    C --> L
    D --> L
    E --> L
    K --> L
```

## Reliability contracts

| Contract | Failure prevented | Public demonstration |
| --- | --- | --- |
| Connection generation and nonce | A stale session issuing work after reconnect | Stale generations and incorrect handshake identities are rejected |
| Durable command ID | Duplicate execution after retry or replay | A repeated command is suppressed without growing the queue |
| Bounded queue | Unbounded memory growth and silent backpressure | Capacity breaches fail closed and mark the bridge degraded |
| Exact receipt correlation | Success being attributed to the wrong request | Mismatched receipt IDs cannot produce terminal success |
| Reconcile-only recovery | Recovered external state causing another execution | Snapshots update evidence without invoking the executor |
| Explicit kill switch | Unsafe admission during operator intervention | New work is rejected while the stop control is active |
| Structured event ledger | “Connected” being mistaken for “completed” | Admission, rejection, execution, and recovery remain separate events |

## State sequence

```mermaid
sequenceDiagram
    participant P as Producer
    participant B as Bridge
    participant A as Adapter
    participant L as Event ledger

    P->>B: Connect generation N + nonce
    B->>L: connection_started
    P->>B: Matching handshake acknowledgement
    B->>L: handshake_accepted
    P->>B: Command(command_id, generation N)
    B->>L: command_accepted
    B->>A: Execute command
    A-->>B: Receipt(same command_id)
    B->>L: execution_terminal
```

