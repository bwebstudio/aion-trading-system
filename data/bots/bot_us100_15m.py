import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta, time
import time as t

# =========================
# CONFIG
# =========================
SYMBOL = "US100.cash"
MAGIC = 151505
RR = 2.0
MAX_TRADES_PER_DAY = 1
BUFFER_POINTS = 3.0

NY_RANGE_START_TIME = time(16, 30)
OPERATING_WINDOW_MIN = 90

# =========================
# BOT
# =========================
class FVG15m5mBot:

    def __init__(self):
        self.range = None
        self.trades = 0
        self.end_time = None
        self.last_candle_time = None
        self.buffer = []
        self.range_close_time = None
        self.pending_fvg = None

    # -------------------------
    # INIT MT5
    # -------------------------
    def init(self):
        if not mt5.initialize():
            print("❌ MT5 no pudo inicializar")
            return False

        mt5.symbol_select(SYMBOL, True)
        acc = mt5.account_info()
        print(f"✅ MT5 conectado | Cuenta {acc.login} | Balance {acc.balance}")
        return True

    # -------------------------
    # GET DATA
    # -------------------------
    def get(self, tf, n=50):
        r = mt5.copy_rates_from_pos(SYMBOL, tf, 0, n)
        if r is None:
            return None
        df = pd.DataFrame(r)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df

    # -------------------------
    # WAIT & SET NY RANGE (15m)
    # -------------------------
    def wait_and_set_range(self):
        print("⏳ Esperando cierre vela 15m de apertura NY (16:30–16:44 MT5)...")

        while True:
            df = self.get(mt5.TIMEFRAME_M15, 3)
            if df is None or len(df) < 2:
                t.sleep(5)
                continue

            c = df.iloc[-2]  # última vela cerrada

            if c.time.time() == NY_RANGE_START_TIME:
                self.range = {"high": c.high, "low": c.low}
                self.range_close_time = c.time + timedelta(minutes=15)
                self.end_time = c.time + timedelta(minutes=OPERATING_WINDOW_MIN)

                print("🔒 Rango M15 fijado correctamente")
                print(c)
                return

            t.sleep(5)

    # -------------------------
    # FVG DETECTION (5m)
    # -------------------------
    def detect_fvg(self, c2, c1, c0):
        rh, rl = self.range["high"], self.range["low"]

        bull_fvg = c2.high < c0.low
        bear_fvg = c2.low > c0.high

        inside = any((c.low <= rh and c.high >= rl) for c in (c2, c1, c0))
        above = any(c.close > rh for c in (c2, c1, c0))
        below = any(c.close < rl for c in (c2, c1, c0))

        if bull_fvg and inside and above:
            bc = next(c for c in (c2, c1, c0) if c.close > rh)
            return {"type": "buy", "break_candle": bc}

        if bear_fvg and inside and below:
            bc = next(c for c in (c2, c1, c0) if c.close < rl)
            return {"type": "sell", "break_candle": bc}

        return None

    # -------------------------
    # LOT CALC
    # -------------------------
    def calc_lot(self, entry, sl, risk_pct):
        acc = mt5.account_info()
        risk = acc.balance * risk_pct
        pts = abs(entry - sl)
        if pts <= 0:
            return 0
        return round(risk / pts, 2)

    # -------------------------
    # OPEN TRADE
    # -------------------------
    def open_trade(self, fvg):
        if self.trades >= MAX_TRADES_PER_DAY:
            return

        tick = mt5.symbol_info_tick(SYMBOL)
        bc = fvg["break_candle"]

        range_break = abs(bc.high - bc.low)

        if range_break <= 60:
            risk_pct = 0.003
        elif range_break <= 90:
            risk_pct = 0.002
        else:
            return

        if fvg["type"] == "buy":
            entry = tick.ask
            sl = bc.low - BUFFER_POINTS
            tp = entry + (entry - sl) * RR
            typ = mt5.ORDER_TYPE_BUY
        else:
            entry = tick.bid
            sl = bc.high + BUFFER_POINTS
            tp = entry - (sl - entry) * RR
            typ = mt5.ORDER_TYPE_SELL

        vol = self.calc_lot(entry, sl, risk_pct)
        if vol <= 0:
            return

        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": vol,
            "type": typ,
            "price": entry,
            "sl": sl,
            "tp": tp,
            "deviation": 30,
            "magic": MAGIC,
            "comment": "FVG_15m_5m",
            "type_filling": mt5.ORDER_FILLING_IOC
        }

        r = mt5.order_send(req)
        if r.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"🚀 Trade abierto | SL {sl} | TP {tp}")
            self.trades += 1

    # -------------------------
    # MAIN LOOP
    # -------------------------
    def run(self):
        self.wait_and_set_range()
        print("👀 Buscando FVG en M5...\n")

        while datetime.now() < self.end_time and self.trades < MAX_TRADES_PER_DAY:
            df = self.get(mt5.TIMEFRAME_M5, 10)
            if df is None or len(df) < 3:
                t.sleep(5)
                continue

            c = df.iloc[-2]

            if self.last_candle_time != c.time:
                if c.time >= self.range_close_time:
                    self.buffer.append(c)

                    if self.pending_fvg and c.time > self.pending_fvg["confirm_after"]:
                        rh, rl = self.range["high"], self.range["low"]

                        if self.pending_fvg["type"] == "buy" and c.close < rh:
                            self.pending_fvg = None
                        elif self.pending_fvg["type"] == "sell" and c.close > rl:
                            self.pending_fvg = None
                        else:
                            self.open_trade(self.pending_fvg)
                            self.pending_fvg = None

                    if len(self.buffer) >= 3 and not self.pending_fvg:
                        fvg = self.detect_fvg(
                            self.buffer[-3],
                            self.buffer[-2],
                            self.buffer[-1]
                        )
                        if fvg:
                            fvg["confirm_after"] = c.time
                            self.pending_fvg = fvg

                self.last_candle_time = c.time

            t.sleep(5)

        print("⏹️ Fin sesión NY")
        mt5.shutdown()

# =========================
# RUN
# =========================
if __name__ == "__main__":
    bot = FVG15m5mBot()
    if bot.init():
        bot.run()
