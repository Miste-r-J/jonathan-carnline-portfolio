# Evidence and disclosure

## What this repository proves

This repository is a runnable, domain-neutral demonstration of reliability patterns used while hardening a larger private Python/C# event-driven integration:

- fail-closed readiness and handshake validation;
- generation-aware connection state;
- durable command identifiers and replay suppression;
- bounded queues with explicit degraded state;
- exact command-to-receipt correlation;
- reconcile-only recovery semantics;
- operator-controlled stop behavior; and
- structured, machine-readable lifecycle evidence.

The included verification suite exercises each public contract independently. The demonstration is intentionally small enough to review during an interview.

## What was inspected privately

The case study was prepared after reviewing the active Python runtime, deployed C# integration surface, launch contract, and focused regression tests. The review confirmed that the private system implements the categories of behavior demonstrated here.

No private source file, production artifact, model, configuration, or event record was copied into this repository.

## What is deliberately excluded

- signal, prediction, ranking, or decision research;
- model features, weights, thresholds, training data, or evaluation results;
- instruments, account identifiers, financial performance, and risk parameters;
- broker-specific implementation details;
- production endpoints, hostnames, credentials, keys, and environment values;
- operational launch configuration; and
- enough domain logic to recreate the private system.

## Scope statement

This is engineering evidence, not a claim about investment performance and not a release of a production trading system. The public implementation uses generic commands such as `APPLY_CHANGE` and `APPLY_CONFIGURATION` so the reliability contracts can be evaluated without exposing the private domain.

