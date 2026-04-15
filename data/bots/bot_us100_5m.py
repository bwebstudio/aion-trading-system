import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta, time
import time as t

# =========================
# CONFIG
# =========================
SYMBOL               = "US100.cash"
MAGIC                = 163000
RR                   = 2.0
RISK_PCT             = 0.0025
BUFFER_POINTS        = 10.0
OR_TIME              = time(16, 30)
MT5_OFFSET_HOURS     = 2

# ── PARÁMETROS OPTIMIZADOS (backtest 2025-01 → 2026-03) ──
MIN_OR_RANGE              = 30.0   # era 10.0  → filtra OR de rango estrecho (WR 0% en <30 pts)
MAX_FVG_GAP               = 50.0   # nuevo     → filtra gaps enormes (WR 0% en >50 pts)
EMA_PERIOD                = 20     # nuevo     → EMA20 sobre velas M5 para filtro de tendencia
WINDOW_WITH_TREND_MIN     = 45     # era 90    → ventana operativa cuando el trade va con EMA20
WINDOW_COUNTER_TREND_MIN  = 20     # nuevo     → ventana operativa cuando el trade va contra EMA20


class FVGBot1630:

    def __init__(self):
        self.range            = None
        self.end_time         = None
        self.range_close_time = None
        self.last_candle_time = None
        self.buffer           = []
        self.pending_fvg      = None
        self.traded           = False
        self.ema20_at_or      = None   # EMA20 de la vela M5 de las 16:30

    def init(self):
        if not mt5.initialize():
            print("❌ MT5 init failed")
            return False
        mt5.symbol_select(SYMBOL, True)
        acc = mt5.account_info()
        print(f"✅ Conectado | Balance: {acc.balance:.2f}")
        return True

    def get(self, tf, n=50):
        r = mt5.copy_rates_from_pos(SYMBOL, tf, 0, n)
        if r is None:
            return None
        df = pd.DataFrame(r)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df

    def calc_ema20(self):
        """Calcula la EMA20 sobre las últimas velas M5 en el momento del OR."""
        df = self.get(mt5.TIMEFRAME_M5, EMA_PERIOD + 5)
        if df is None or len(df) < EMA_PERIOD:
            print("⚠️  No hay suficientes velas M5 para calcular EMA20")
            return None
        # La vela OR es iloc[-2] (ya cerrada), usamos el close de esa vela como referencia
        closes = df["close"].values
        # EMA con factor alpha = 2/(n+1)
        alpha = 2.0 / (EMA_PERIOD + 1)
        ema = closes[0]
        for c in closes[1:]:
            ema = alpha * c + (1 - alpha) * ema
        return round(ema, 2)

    def wait_and_set_range(self):
        print(f"\n⏳ Esperando OR de las {OR_TIME} (hora MT5) ...")

        while True:
            df = self.get(mt5.TIMEFRAME_M5, 5)
            if df is None or len(df) < 2:
                t.sleep(5)
                continue

            c = df.iloc[-2]

            if c.time.time() == OR_TIME:
                or_range = float(c.high) - float(c.low)

                print(f"📊 Vela OR capturada → {c.time.strftime('%Y-%m-%d %H:%M')} MT5")
                print(f"   Open:  {float(c.open):.2f}")
                print(f"   High:  {float(c.high):.2f}")
                print(f"   Low:   {float(c.low):.2f}")
                print(f"   Close: {float(c.close):.2f}")
                print(f"   Rango: {or_range:.2f} pts")

                # ── FILTRO 1: OR range mínimo ──
                if or_range < MIN_OR_RANGE:
                    print(f"⛔ OR demasiado pequeño ({or_range:.1f} pts < {MIN_OR_RANGE} pts) → sin operación hoy")
                    return

                # ── EMA20 en el momento del OR ──
                self.ema20_at_or = self.calc_ema20()
                or_close = float(c.close)
                if self.ema20_at_or is not None:
                    trend = "ALCISTA 📈" if or_close > self.ema20_at_or else "BAJISTA 📉"
                    print(f"   EMA20: {self.ema20_at_or:.2f} | OR close: {or_close:.2f} → Tendencia M5: {trend}")

                self.range = {
                    "high":  float(c.high),
                    "low":   float(c.low),
                    "close": or_close
                }
                self.range_close_time = c.time + timedelta(minutes=5)
                self.buffer           = []
                self.pending_fvg      = None
                self.traded           = False
                self.last_candle_time = None

                # end_time se calculará dinámicamente según tendencia en run_session
                # Usamos el máximo posible como límite de seguridad
                self.end_time = c.time + timedelta(minutes=max(WINDOW_WITH_TREND_MIN, WINDOW_COUNTER_TREND_MIN))

                print(f"🔒 OR fijado → High: {self.range['high']:.2f} | Low: {self.range['low']:.2f} | Rango: {or_range:.1f} pts ✅")
                return

            t.sleep(5)

    def is_within_window(self, candle_time, fvg_type):
        """
        Comprueba si la vela está dentro de la ventana operativa.
        La ventana depende de si el FVG va con o contra la EMA20.
        """
        if self.ema20_at_or is None:
            # Sin EMA disponible: usar ventana conservadora
            window = WINDOW_COUNTER_TREND_MIN
        else:
            or_close = self.range["close"]
            if fvg_type == "buy":
                with_trend = or_close > self.ema20_at_or
            else:
                with_trend = or_close < self.ema20_at_or

            window = WINDOW_WITH_TREND_MIN if with_trend else WINDOW_COUNTER_TREND_MIN

        or_start = self.range_close_time - timedelta(minutes=5)  # hora de la vela OR
        limit = or_start + timedelta(minutes=window)
        return candle_time < limit, limit, window

    def detect_fvg(self, c2, c1, c0):
        rh, rl = self.range["high"], self.range["low"]

        print(f"  🔍 Triplete evaluado:")
        print(f"     c2: {c2.time.strftime('%Y-%m-%d %H:%M')} MT5 | "
              f"O:{c2.open:.2f} H:{c2.high:.2f} L:{c2.low:.2f} C:{c2.close:.2f} | "
              f"rango:{c2.high-c2.low:.1f}pts")
        print(f"     c1: {c1.time.strftime('%Y-%m-%d %H:%M')} MT5 | "
              f"O:{c1.open:.2f} H:{c1.high:.2f} L:{c1.low:.2f} C:{c1.close:.2f} | "
              f"rango:{c1.high-c1.low:.1f}pts")
        print(f"     c0: {c0.time.strftime('%Y-%m-%d %H:%M')} MT5 | "
              f"O:{c0.open:.2f} H:{c0.high:.2f} L:{c0.low:.2f} C:{c0.close:.2f} | "
              f"rango:{c0.high-c0.low:.1f}pts")
        print(f"     OR: H:{rh:.2f} L:{rl:.2f} | "
              f"gap_bull:{c0.low-c2.high:.1f} gap_bear:{c2.low-c0.high:.1f}")

        bull_fvg = c2.high < c0.low
        bear_fvg = c2.low  > c0.high

        if bull_fvg:
            breaking = [c for c in (c2, c1, c0) if c.open <= rh and c.close > rh]
            inside   = any(c.low <= rh and c.high >= rl for c in (c2, c1, c0))
            if not breaking or not inside:
                print(f"     ❌ BULL gap pero sin BC válida o fuera del OR")
                return None
            bc  = breaking[0]
            gap = c0.low - c2.high

            # ── FILTRO 2: FVG gap máximo ──
            if gap > MAX_FVG_GAP:
                print(f"     ❌ BULL gap demasiado grande ({gap:.1f} pts > {MAX_FVG_GAP} pts) → skip")
                return None

            print(
                f"\n🟢 BULL FVG\n"
                f"  c2: {c2.time.strftime('%H:%M')} H:{c2.high:.2f} L:{c2.low:.2f}\n"
                f"  c1: {c1.time.strftime('%H:%M')} H:{c1.high:.2f} L:{c1.low:.2f}\n"
                f"  c0: {c0.time.strftime('%H:%M')} H:{c0.high:.2f} L:{c0.low:.2f}\n"
                f"  Gap: {gap:.2f} pts | Break: {bc.time.strftime('%H:%M')} "
                f"open:{bc.open:.2f}≤{rh:.2f} close:{bc.close:.2f}>{rh:.2f}\n"
                f"  BC rango: {bc.high-bc.low:.1f} pts | bc.low={bc.low:.2f} → SL={bc.low-BUFFER_POINTS:.2f}\n"
                f"  → Esperando confirmación..."
            )
            return {"type": "buy", "break_candle": bc, "gap": gap}

        if bear_fvg:
            breaking = [c for c in (c2, c1, c0) if c.open >= rl and c.close < rl]
            inside   = any(c.low <= rh and c.high >= rl for c in (c2, c1, c0))
            if not breaking or not inside:
                print(f"     ❌ BEAR gap pero sin BC válida o fuera del OR")
                return None
            bc  = breaking[0]
            gap = c2.low - c0.high

            # ── FILTRO 2: FVG gap máximo ──
            if gap > MAX_FVG_GAP:
                print(f"     ❌ BEAR gap demasiado grande ({gap:.1f} pts > {MAX_FVG_GAP} pts) → skip")
                return None

            print(
                f"\n🔴 BEAR FVG\n"
                f"  c2: {c2.time.strftime('%H:%M')} H:{c2.high:.2f} L:{c2.low:.2f}\n"
                f"  c1: {c1.time.strftime('%H:%M')} H:{c1.high:.2f} L:{c1.low:.2f}\n"
                f"  c0: {c0.time.strftime('%H:%M')} H:{c0.high:.2f} L:{c0.low:.2f}\n"
                f"  Gap: {gap:.2f} pts | Break: {bc.time.strftime('%H:%M')} "
                f"open:{bc.open:.2f}≥{rl:.2f} close:{bc.close:.2f}<{rl:.2f}\n"
                f"  BC rango: {bc.high-bc.low:.1f} pts | bc.high={bc.high:.2f} → SL={bc.high+BUFFER_POINTS:.2f}\n"
                f"  → Esperando confirmación..."
            )
            return {"type": "sell", "break_candle": bc, "gap": gap}

        print(f"     ❌ Sin gap estructural")
        return None

    def calc_lot(self, entry, sl):
        acc  = mt5.account_info()
        risk = acc.balance * RISK_PCT
        pts  = abs(entry - sl)
        if pts <= 0:
            return 0
        return round(risk / pts, 2)

    def has_open_position(self):
        pos = mt5.positions_get(symbol=SYMBOL)
        return pos is not None and len(pos) > 0

    def open_trade(self, fvg, confirm_candle):
        if self.has_open_position():
            print("⛔ Posición abierta → skip")
            return False

        tick = mt5.symbol_info_tick(SYMBOL)
        bc   = fvg["break_candle"]

        if fvg["type"] == "buy":
            entry  = tick.ask
            sl     = bc.low - BUFFER_POINTS
            tp     = entry + (entry - sl) * RR
            typ    = mt5.ORDER_TYPE_BUY
            sl_log = f"bc.low {bc.low:.2f} - {BUFFER_POINTS}"
        else:
            entry  = tick.bid
            sl     = bc.high + BUFFER_POINTS
            tp     = entry - (sl - entry) * RR
            typ    = mt5.ORDER_TYPE_SELL
            sl_log = f"bc.high {bc.high:.2f} + {BUFFER_POINTS}"

        vol = self.calc_lot(entry, sl)
        if vol <= 0:
            print("⚠️ Lotaje inválido")
            return False

        req = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       SYMBOL,
            "volume":       vol,
            "type":         typ,
            "price":        entry,
            "sl":           round(sl, 2),
            "tp":           round(tp, 2),
            "deviation":    30,
            "magic":        MAGIC,
            "comment":      "FVG_1630_v2",
            "type_filling": mt5.ORDER_FILLING_IOC
        }

        r = mt5.order_send(req)

        if r.retcode == mt5.TRADE_RETCODE_DONE:
            print(
                f"🚀 TRADE EJECUTADO\n"
                f"  Dir:   {'BUY' if fvg['type'] == 'buy' else 'SELL'}\n"
                f"  Entry: {entry:.2f}\n"
                f"  SL:    {sl:.2f} ({sl_log})\n"
                f"  TP:    {tp:.2f}\n"
                f"  Vol:   {vol}\n"
                f"  Confirmación: {confirm_candle.time.strftime('%H:%M')} "
                f"close:{confirm_candle.close:.2f}"
            )
            return True
        else:
            print(f"❌ Error orden: {r.retcode}")
            return False

    def run_session(self):
        print("\n👀 Buscando FVG en M1...\n")

        while True:
            if self.traded:
                print("✅ Trade ejecutado → fin sesión")
                return

            df = self.get(mt5.TIMEFRAME_M1, 10)
            if df is None or len(df) < 3:
                t.sleep(3)
                continue

            c = df.iloc[-2]  # última vela M1 cerrada

            # ── DEBUG: mostrar todas las velas que devuelve get() ──
            print(f"\n--- Nueva vela M1 cerrada: {c.time.strftime('%Y-%m-%d %H:%M')} MT5 ---")
            print(f"  get(M1,10) devuelve {len(df)} velas:")
            for _, row in df.iterrows():
                marker = " ◄ c (iloc[-2])" if row["time"] == c.time else ""
                print(f"    {row['time'].strftime('%H:%M')} MT5 | "
                      f"O:{row['open']:.2f} H:{row['high']:.2f} "
                      f"L:{row['low']:.2f} C:{row['close']:.2f} | "
                      f"rango:{row['high']-row['low']:.1f}pts{marker}")

            if self.last_candle_time == c.time:
                t.sleep(1)
                continue

            self.last_candle_time = c.time

            if c.time < self.range_close_time:
                print(f"  ⏭️  Vela anterior al cierre del OR ({self.range_close_time.strftime('%H:%M')}) → skip")
                t.sleep(1)
                continue

            self.buffer.append(c)
            print(f"  Buffer ({len(self.buffer)} velas): "
                  f"{[x.time.strftime('%H:%M') for x in self.buffer]}")

            rh, rl = self.range["high"], self.range["low"]

            # PASO 2: confirmar FVG pendiente
            if self.pending_fvg:
                ftype = self.pending_fvg["type"]

                # ── FILTRO 3: ventana adaptativa según tendencia ──
                in_window, window_limit, window_min = self.is_within_window(c.time, ftype)
                if not in_window:
                    trend_label = "con-tendencia" if window_min == WINDOW_WITH_TREND_MIN else "contra-tendencia"
                    print(f"⏹️  Fuera de ventana {trend_label} ({window_min} min → límite {window_limit.strftime('%H:%M')}) → fin sesión")
                    return

                if ftype == "buy" and c.close > rh:
                    print(f"✅ Confirmación BULL → {c.time.strftime('%H:%M')} "
                          f"close:{c.close:.2f} > {rh:.2f}")
                    if self.open_trade(self.pending_fvg, c):
                        self.traded = True
                    self.pending_fvg = None

                elif ftype == "sell" and c.close < rl:
                    print(f"✅ Confirmación BEAR → {c.time.strftime('%H:%M')} "
                          f"close:{c.close:.2f} < {rl:.2f}")
                    if self.open_trade(self.pending_fvg, c):
                        self.traded = True
                    self.pending_fvg = None

                else:
                    print(f"❌ FVG invalidado → {c.time.strftime('%H:%M')} "
                          f"close:{c.close:.2f} no confirma")
                    self.pending_fvg = None

                t.sleep(1)
                continue

            # PASO 3: detectar nuevo FVG
            if len(self.buffer) >= 3:
                fvg = self.detect_fvg(
                    self.buffer[-3],
                    self.buffer[-2],
                    self.buffer[-1]
                )
                if fvg:
                    # ── Verificar ventana ANTES de registrar el FVG ──
                    in_window, window_limit, window_min = self.is_within_window(c.time, fvg["type"])
                    if not in_window:
                        trend_label = "con-tendencia" if window_min == WINDOW_WITH_TREND_MIN else "contra-tendencia"
                        print(f"⏹️  FVG detectado pero fuera de ventana {trend_label} ({window_min} min) → fin sesión")
                        return
                    self.pending_fvg = fvg

            # Comprobación de tiempo máximo absoluto (ventana con tendencia)
            now_mt5 = datetime.now() + timedelta(hours=MT5_OFFSET_HOURS)
            if now_mt5 >= self.end_time:
                print("⏹️  Fin ventana operativa sin trade")
                return

            t.sleep(1)

    def run(self):
        print("🤖 Bot v2 iniciado — OR 16:30 | FVG + EMA20 + Ventana Adaptativa\n")
        print(f"   MIN_OR_RANGE:             {MIN_OR_RANGE} pts")
        print(f"   MAX_FVG_GAP:              {MAX_FVG_GAP} pts")
        print(f"   EMA_PERIOD (M5):          {EMA_PERIOD}")
        print(f"   Ventana con-tendencia:    {WINDOW_WITH_TREND_MIN} min")
        print(f"   Ventana contra-tendencia: {WINDOW_COUNTER_TREND_MIN} min\n")
        self.wait_and_set_range()
        if self.range is None:
            print("⏹️  Sin sesión hoy por filtro OR")
        else:
            self.run_session()
        print("\n⏹️  Fin del día")
        mt5.shutdown()


if __name__ == "__main__":
    bot = FVGBot1630()
    if bot.init():
        bot.run()