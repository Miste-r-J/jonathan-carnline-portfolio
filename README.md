# Jonathan Carnline — Engineering Portfolio

This repository contains my portfolio site and the real source for my reliable trading runtime.

## Start here

- [`projects/reliable-trading-runtime/README.md`](projects/reliable-trading-runtime/README.md) — what I built and why.
- [`projects/reliable-trading-runtime/docs/HOW_IT_WORKS.md`](projects/reliable-trading-runtime/docs/HOW_IT_WORKS.md) — end-to-end runtime walkthrough.
- [`projects/reliable-trading-runtime/docs/WHAT_I_OWNED.md`](projects/reliable-trading-runtime/docs/WHAT_I_OWNED.md) — the work I personally did and the engineering decisions I made.
- [`projects/reliable-trading-runtime/simplified/na/discord_addons/cli/stream_live_csv.py`](projects/reliable-trading-runtime/simplified/na/discord_addons/cli/stream_live_csv.py) — the main operating runtime.
- [`projects/reliable-trading-runtime/simplified/na/tests/`](projects/reliable-trading-runtime/simplified/na/tests) — execution, reliability, replay, data, and model-contract tests.

## Portfolio site

The site is built with React and Vite.

```powershell
pnpm install
pnpm dev
```

Production build:

```powershell
pnpm build
```

## Repository layout

```text
src/                                portfolio website
public/                             downloadable resume
projects/reliable-trading-runtime/  actual engineering project source
.github/workflows/                  site deployment and project checks
```

## Public-repository boundary

The source, tests, configuration structure, and technical explanations are public. Credentials, account identifiers, webhook URLs, private keys, raw logs, datasets, trained model binaries, generated runs, and machine-specific secrets are excluded.

The technical project is independent work. My paid operations experience is listed separately on the portfolio and resume.
