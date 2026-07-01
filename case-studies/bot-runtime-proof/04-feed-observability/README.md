# 04 — Feed Observability

The live feed work was about proving that fresh data was actually moving through the system.

A socket can be open and still not deliver useful data. A heartbeat can be alive while bars are not progressing. A queue can be full and silently drop the message that mattered.

## What I built

- heartbeat tracking;
- bar sequence checks;
- stale generation rejection;
- queue depth and drop counters;
- backpressure reporting;
- bar-vs-nonbar drop tracking;
- transport state and ingress state fields in status output;
- watchdog behavior that distinguishes sender silence from transport failure;
- re-arm logic that requires true bar progress, not just activity.

## Status fields I cared about

```text
fiber_transport_state
fiber_ingress_state
fiber_queue_depth
fiber_queue_dropped
fiber_bar_drop_count
fiber_nonbar_drop_count
fiber_last_bar_age_sec
fiber_last_heartbeat_age_sec
bar_seq_gap_count
bar_send_to_recv_ms
bar_receive_to_process_ms
bar_last_skip_reason
```

## The lesson

For live systems, "the connection is up" is not the same thing as "the system is healthy."

I wanted the runtime to explain whether it was receiving new bars, whether the bars were in order, whether the queue was backing up, whether non-bar messages were crowding out bar messages, and whether recovery had actually happened.

## What this proves

I can debug live systems by adding the right observability instead of guessing. I know how to separate connectivity, freshness, ordering, queue pressure, and processing delay.

