import json
import socket
import threading
import time


def _server(port: int) -> None:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(1)
    conn, _ = srv.accept()
    with conn:
        data = conn.recv(4096).decode("utf-8").strip()
        msg = json.loads(data)
        ok = msg.get("type") == "ORDER" and "order" in msg and msg.get("protocol_version") == 1
        resp = {
            "protocol_version": 1,
            "type": "ACK",
            "request_id": msg.get("order", {}).get("client_order_id"),
            "status": "EXITS_WORKING" if ok else "REJECTED",
            "message": "ok" if ok else "bad_schema",
            "client_order_id": msg.get("order", {}).get("client_order_id"),
            "stop_price": msg.get("order", {}).get("stop_price"),
            "target_price": msg.get("order", {}).get("target_price"),
        }
        conn.sendall((json.dumps(resp) + "\n").encode("utf-8"))
    srv.close()


def main() -> None:
    port = 40123
    th = threading.Thread(target=_server, args=(port,), daemon=True)
    th.start()
    time.sleep(0.1)
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("127.0.0.1", port))
    payload = {
        "protocol_version": 1,
        "type": "ORDER",
        "order": {
            "instrument": "MES 03-26",
            "side": "BUY",
            "quantity": 1,
            "entry_type": "MARKET",
            "entry_price": None,
            "stop_price": 4990.0,
            "target_price": 5010.0,
            "client_order_id": "smoke-1",
        },
    }
    client.sendall((json.dumps(payload) + "\n").encode("utf-8"))
    resp = client.recv(4096).decode("utf-8").strip()
    print(resp)
    client.close()


if __name__ == "__main__":
    main()
