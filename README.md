# Jonathan Carnline — Portfolio

Portfolio for operations technology, automation, application support, production support, Python/C# work, logistics technology, and reliability roles.

## What this repo contains

- `src/` — React/Vite portfolio site.
- `case-studies/reliable-event-bridge/` — cleaned reliability case study with runnable tests.
- `public/reliable-event-bridge.zip` — downloadable copy of the case study.
- `public/Jonathan_Carnline_Technical_Resume.pdf` — resume download.

## What this repo intentionally excludes

- Production trading strategies, private research, credentials, account identifiers, private logs, private screenshots, and private configuration.
- Generated dependency folders such as `node_modules/`.
- Local cache/build output; GitHub Pages builds the site from source.

## Local development

```powershell
pnpm install
pnpm dev
```

Open `http://127.0.0.1:5173`.

## Production build

```powershell
pnpm build
```

The deployable static output is written to `dist/`.

## GitHub Pages deployment

This repo includes `.github/workflows/pages.yml`. After pushing to a GitHub repository:

1. Open the repository on GitHub.
2. Go to **Settings → Pages**.
3. Set **Source** to **GitHub Actions**.
4. Push to `main` or run the workflow manually.

## Content rules

- Independent technical projects are labeled separately from paid professional experience.
- Education is stated as coursework with 30 semester credits completed.
- Trading credentials, account identifiers, strategies, private infrastructure details, and private research logic are excluded.
- Project claims must remain traceable to the career evidence inventory in the parent resume workspace.
