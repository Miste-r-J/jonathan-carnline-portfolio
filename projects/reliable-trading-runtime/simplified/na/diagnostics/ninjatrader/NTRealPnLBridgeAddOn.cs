// NT8 AddOn: authoritative PnL producer for stream_live_csv.py
// Drop into NinjaTrader 8 AddOns project and wire endpoint/filters as needed.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using NinjaTrader.Cbi;
using NinjaTrader.NinjaScript;

namespace NinjaTrader.NinjaScript.AddOns
{
    public class NTRealPnLBridgeAddOn : AddOnBase
    {
        private const int SchemaVersion = 1;
        private const string Source = "nt_account_api";

        private readonly object sync = new object();
        private readonly ConcurrentDictionary<string, long> seqByKey = new ConcurrentDictionary<string, long>(StringComparer.OrdinalIgnoreCase);
        private readonly ConcurrentDictionary<string, SnapshotState> stateByKey = new ConcurrentDictionary<string, SnapshotState>(StringComparer.OrdinalIgnoreCase);
        private readonly FixedRingBuffer<string> outboundBuffer = new FixedRingBuffer<string>(1024);

        private string host = "127.0.0.1";
        private int port = 5019;
        private string outputPath = string.Empty;
        private bool emitJsonl = true;
        private bool sendSocket = true;
        private bool includeSim = true;

        private Timer heartbeatTimer;
        private volatile bool running;
        private TcpClient client;
        private StreamWriter socketWriter;

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
                Name = "NTRealPnLBridgeAddOn";
            else if (State == State.Active)
                Start();
            else if (State == State.Terminated)
                Stop();
        }

        private void Start()
        {
            if (running) return;
            running = true;

            // Configure output path under Documents\NinjaTrader 8\log
            if (string.IsNullOrWhiteSpace(outputPath))
            {
                var baseDir = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments);
                outputPath = Path.Combine(baseDir, "NinjaTrader 8", "log", "nt_bridge.jsonl");
            }

            TryConnectSocket();
            SubscribeAccounts();
            heartbeatTimer = new Timer(_ => EmitHeartbeat(), null, TimeSpan.FromSeconds(1), TimeSpan.FromSeconds(1));
            EmitBridgeStatus("up", "addon_started");
        }

        private void Stop()
        {
            running = false;
            try { heartbeatTimer?.Dispose(); } catch { }
            heartbeatTimer = null;
            UnsubscribeAccounts();
            EmitBridgeStatus("down", "addon_stopped");
            try { socketWriter?.Dispose(); } catch { }
            try { client?.Dispose(); } catch { }
            socketWriter = null;
            client = null;
        }

        private void SubscribeAccounts()
        {
            foreach (var acct in Account.All)
            {
                if (!includeSim && IsSimAccount(acct))
                    continue;

                acct.AccountItemUpdate += OnAccountItemUpdate;
                acct.PositionUpdate += OnPositionUpdate;
                acct.ExecutionUpdate += OnExecutionUpdate;
            }
        }

        private void UnsubscribeAccounts()
        {
            foreach (var acct in Account.All)
            {
                acct.AccountItemUpdate -= OnAccountItemUpdate;
                acct.PositionUpdate -= OnPositionUpdate;
                acct.ExecutionUpdate -= OnExecutionUpdate;
            }
        }

        private void OnAccountItemUpdate(object sender, AccountItemEventArgs e)
        {
            if (!running || sender is not Account acct) return;
            if (e == null) return;

            // Only emit aggregate items relevant to PnL state snapshots.
            if (e.AccountItem != AccountItem.RealizedProfitLoss && e.AccountItem != AccountItem.UnrealizedProfitLoss)
                return;

            foreach (var pos in acct.Positions)
                EmitSnapshot(acct, pos, "PNL_SNAPSHOT", "account_item_update");
        }

        private void OnPositionUpdate(object sender, PositionEventArgs e)
        {
            if (!running || sender is not Account acct || e?.Position == null) return;
            EmitSnapshot(acct, e.Position, "POSITION_UPDATE", "position_update");
        }

        private void OnExecutionUpdate(object sender, ExecutionEventArgs e)
        {
            if (!running || sender is not Account acct || e?.Execution == null) return;
            var inst = e.Execution.Instrument;
            if (inst == null) return;
            var pos = acct.Positions.FirstOrDefault(p => p.Instrument != null && string.Equals(p.Instrument.FullName, inst.FullName, StringComparison.OrdinalIgnoreCase));
            if (pos != null)
                EmitSnapshot(acct, pos, "PNL_SNAPSHOT", "execution_update", e.Execution);
        }

        private void EmitHeartbeat()
        {
            if (!running) return;
            var payload = new Dictionary<string, object>
            {
                ["schema_version"] = SchemaVersion,
                ["event_type"] = "HEARTBEAT",
                ["source"] = Source,
                ["ts_local"] = DateTimeOffset.Now.ToString("o"),
                ["ts_exchange"] = DateTimeOffset.UtcNow.ToString("o"),
                ["is_live_market_data"] = true,
                ["is_simulated"] = false,
                ["is_reconstructed"] = false
            };
            Emit(payload);
        }

        private void EmitBridgeStatus(string status, string reason)
        {
            var payload = new Dictionary<string, object>
            {
                ["schema_version"] = SchemaVersion,
                ["event_type"] = "BRIDGE_STATUS",
                ["source"] = Source,
                ["status"] = status,
                ["reason"] = reason,
                ["ts_local"] = DateTimeOffset.Now.ToString("o"),
                ["ts_exchange"] = DateTimeOffset.UtcNow.ToString("o"),
                ["is_live_market_data"] = true,
                ["is_simulated"] = false,
                ["is_reconstructed"] = false
            };
            Emit(payload);
        }

        private void EmitSnapshot(Account acct, Position pos, string eventType, string trigger, Execution execution = null)
        {
            try
            {
                var instrument = pos?.Instrument?.FullName ?? string.Empty;
                var account = acct?.Name ?? string.Empty;
                if (string.IsNullOrWhiteSpace(account) || string.IsNullOrWhiteSpace(instrument))
                    return;

                var key = account + "|" + instrument;
                var seq = seqByKey.AddOrUpdate(key, 1L, (_, old) => old + 1L);

                var qty = SafeDouble(pos.Quantity);
                var avg = SafeDouble(pos.AveragePrice);
                var last = ResolveLastPrice(pos);
                var unrealized = ResolveUnrealized(acct, pos);
                var realized = ResolveRealized(acct, pos);
                var commission = ResolveCommission(acct, pos);
                var entryOrderId = execution?.Id.ToString() ?? string.Empty;
                var entryNinjaOrderId = execution?.Order?.Id.ToString() ?? string.Empty;

                var snap = new SnapshotState
                {
                    Account = account,
                    Instrument = instrument,
                    Qty = qty,
                    AvgPrice = avg,
                    LastPrice = last,
                    Unrealized = unrealized,
                    Realized = realized,
                    Commission = commission,
                    LastSeq = seq,
                    LastTs = DateTimeOffset.UtcNow,
                    EntryOrderId = entryOrderId,
                    EntryNinjaOrderId = entryNinjaOrderId
                };
                stateByKey[key] = snap;

                var payload = new Dictionary<string, object>
                {
                    ["schema_version"] = SchemaVersion,
                    ["event_type"] = eventType,
                    ["seq"] = seq,
                    ["source"] = Source,
                    ["trigger"] = trigger,
                    ["account"] = account,
                    ["instrument"] = instrument,
                    ["position_qty"] = qty,
                    ["avg_price"] = avg,
                    ["last_price"] = last,
                    ["unrealized_pnl_currency"] = unrealized,
                    ["realized_pnl_currency"] = realized,
                    ["commission"] = commission,
                    ["entry_order_id"] = entryOrderId,
                    ["entry_ninja_order_id"] = entryNinjaOrderId,
                    ["ts_local"] = DateTimeOffset.Now.ToString("o"),
                    ["ts_exchange"] = DateTimeOffset.UtcNow.ToString("o"),
                    ["is_live_market_data"] = true,
                    ["is_simulated"] = IsSimAccount(acct),
                    ["is_reconstructed"] = false
                };
                Emit(payload);
            }
            catch
            {
                EmitBridgeStatus("degraded", "emit_snapshot_failed");
            }
        }

        private void Emit(IDictionary<string, object> payload)
        {
            var line = ToJson(payload);
            outboundBuffer.Add(line);

            if (emitJsonl)
                AppendJsonl(line);

            if (sendSocket)
                SendLine(line);
        }

        private void AppendJsonl(string line)
        {
            try
            {
                var dir = Path.GetDirectoryName(outputPath);
                if (!string.IsNullOrWhiteSpace(dir) && !Directory.Exists(dir))
                    Directory.CreateDirectory(dir);
                File.AppendAllText(outputPath, line + Environment.NewLine, Encoding.UTF8);
            }
            catch
            {
                // avoid crashing AddOn for I/O issues
            }
        }

        private void TryConnectSocket()
        {
            try
            {
                client = new TcpClient();
                client.Connect(host, port);
                socketWriter = new StreamWriter(client.GetStream(), new UTF8Encoding(false)) { AutoFlush = true };

                // replay buffered messages on reconnect to reduce state loss
                foreach (var cached in outboundBuffer.Snapshot())
                    socketWriter.WriteLine(cached);
            }
            catch
            {
                socketWriter = null;
                client = null;
            }
        }

        private void SendLine(string line)
        {
            try
            {
                if (socketWriter == null || client == null || !client.Connected)
                    TryConnectSocket();
                socketWriter?.WriteLine(line);
            }
            catch
            {
                try { socketWriter?.Dispose(); } catch { }
                try { client?.Dispose(); } catch { }
                socketWriter = null;
                client = null;
            }
        }

        private static bool IsSimAccount(Account acct)
        {
            if (acct == null) return false;
            var name = acct.Name ?? string.Empty;
            return name.StartsWith("Sim", StringComparison.OrdinalIgnoreCase) || name.StartsWith("Demo", StringComparison.OrdinalIgnoreCase);
        }

        private static double SafeDouble(object value)
        {
            if (value == null) return 0.0;
            if (value is double d) return d;
            if (value is float f) return f;
            if (value is int i) return i;
            if (double.TryParse(Convert.ToString(value, CultureInfo.InvariantCulture), NumberStyles.Any, CultureInfo.InvariantCulture, out var p))
                return p;
            return 0.0;
        }

        private static double ResolveLastPrice(Position pos)
        {
            // AddOn context does not have direct chart-series access; publish avg as safe baseline
            // and allow bridge consumer to combine with separate snapshot price fields when available.
            return SafeDouble(pos?.AveragePrice);
        }

        private static double ResolveUnrealized(Account acct, Position pos)
        {
            try
            {
                return SafeDouble(acct.Get(AccountItem.UnrealizedProfitLoss, Currency.UsDollar));
            }
            catch
            {
                return 0.0;
            }
        }

        private static double ResolveRealized(Account acct, Position pos)
        {
            try
            {
                return SafeDouble(acct.Get(AccountItem.RealizedProfitLoss, Currency.UsDollar));
            }
            catch
            {
                return 0.0;
            }
        }

        private static double ResolveCommission(Account acct, Position pos)
        {
            try
            {
                return SafeDouble(acct.Get(AccountItem.Commission, Currency.UsDollar));
            }
            catch
            {
                return 0.0;
            }
        }

        private sealed class SnapshotState
        {
            public string Account { get; set; }
            public string Instrument { get; set; }
            public double Qty { get; set; }
            public double AvgPrice { get; set; }
            public double LastPrice { get; set; }
            public double Unrealized { get; set; }
            public double Realized { get; set; }
            public double Commission { get; set; }
            public long LastSeq { get; set; }
            public DateTimeOffset LastTs { get; set; }
            public string EntryOrderId { get; set; }
            public string EntryNinjaOrderId { get; set; }
        }

        private sealed class FixedRingBuffer<T>
        {
            private readonly T[] buffer;
            private int next;
            private int count;
            private readonly object gate = new object();

            public FixedRingBuffer(int capacity)
            {
                buffer = new T[Math.Max(1, capacity)];
            }

            public void Add(T item)
            {
                lock (gate)
                {
                    buffer[next] = item;
                    next = (next + 1) % buffer.Length;
                    if (count < buffer.Length) count++;
                }
            }

            public IReadOnlyList<T> Snapshot()
            {
                lock (gate)
                {
                    var outList = new List<T>(count);
                    var start = (next - count + buffer.Length) % buffer.Length;
                    for (var i = 0; i < count; i++)
                    {
                        var idx = (start + i) % buffer.Length;
                        outList.Add(buffer[idx]);
                    }
                    return outList;
                }
            }
        }

        private static string ToJson(IDictionary<string, object> payload)
        {
            IEnumerable<string> parts = payload.Select(kvp =>
            {
                var key = Escape(kvp.Key);
                var val = SerializeValue(kvp.Value);
                return "\"" + key + "\":" + val;
            });
            return "{" + string.Join(",", parts) + "}";
        }

        private static string SerializeValue(object value)
        {
            if (value == null) return "null";
            if (value is bool b) return b ? "true" : "false";
            if (value is int or long or short or byte) return Convert.ToString(value, CultureInfo.InvariantCulture);
            if (value is float or double or decimal) return Convert.ToString(value, CultureInfo.InvariantCulture);
            return "\"" + Escape(Convert.ToString(value, CultureInfo.InvariantCulture) ?? string.Empty) + "\"";
        }

        private static string Escape(string s)
        {
            return (s ?? string.Empty)
                .Replace("\\", "\\\\")
                .Replace("\"", "\\\"")
                .Replace("\r", "\\r")
                .Replace("\n", "\\n")
                .Replace("\t", "\\t");
        }
    }
}
