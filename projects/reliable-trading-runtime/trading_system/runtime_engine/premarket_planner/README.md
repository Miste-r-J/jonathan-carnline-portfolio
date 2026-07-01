# Premarket Planner

Generates the morning ES summary using intraday CSV data and emits a Discord-ready embed. The planner reads `/etc/trading-runtime/premarket_planner.yaml` (override with `--config`) and produces a single webhook payload; no Discord bot token is required.

## Usage

```bash
venv/bin/python -m trading_system.runtime_engine.premarket_planner.cli --config /etc/trading-runtime/premarket_planner.yaml
```

Use `--dry-run` to skip webhook delivery and `--print-json` to dump the payload body.

## systemd integration

Install the CLI on a timer so the payload is generated every trading day at **06:30 America/Denver**.

`/etc/systemd/system/premarket_planner.service`:

```
[Unit]
Description=Generate ES premarket plan
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=ubuntu
WorkingDirectory=/home/ubuntu/Desktop
Environment="PYTHONPATH=/home/ubuntu/Desktop"
EnvironmentFile=/home/ubuntu/Desktop/systemd/premarket_planner.env
ExecStart=/home/ubuntu/Desktop/venv/bin/python -m trading_system.runtime_engine.premarket_planner.cli --config /etc/trading-runtime/premarket_planner.yaml
```

`/etc/systemd/system/premarket_planner.timer`:

```
[Unit]
Description=Run the premarket planner at 06:30 MT

[Timer]
OnCalendar=Mon-Fri 06:30
Persistent=true

[Install]
WantedBy=timers.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now premarket_planner.timer
```
