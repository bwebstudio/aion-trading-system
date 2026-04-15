import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta, time
import time as t

# =========================
# CONFIG
# =========================
SYMBOL           = "US100.cash"
MAGIC            = 164000
RR               = 2.0
RISK_PCT         = 0.0025        # 0.25% riesgo por trade
OR_TIME          = time(16, 30)  # 9:30 EST = 16:30 MT5 (UTC+2)
MT5_OFFSET_HOURS = 2
MIN_OR_RANGE     = 5.0

# ══════════════════════════════════════════════════════════
# LÓGICA (Stupid Simple 2R) — v3, backtest 51.3% WR / PF 2.09
# ══════════════════════════════════════════════════════════
# 1. OR = vela M5 de las 16:30 MT5
#    → OR_high, OR_low, midpoint = (high+low)/2
#
# 2. BREAK CANDLE: vela M5 que cruza el nivel desde dentro:
#      BULL: open <= OR_high  Y  close > OR_high
#      BEAR: open >= OR_low   Y  close < OR_low
#
# 3. RETEST: vela M5 posterior que:
#      a) Toque el nivel roto
#      b) Cierre fuera del rango (confirmación)
#    FAKE OUT: toca pero cierra dentro → reset
#    ⚠️  Si el fake out cumple condición de break en dirección contraria,
#        se activa el breakout inverso sin esperar una nueva vela
#
# 4. ENTRY = OR_high (bull) / OR_low (bear) — nivel fijo
#    SL     = midpoint del OR
#    TP     = entry ± (entry - midpoint) × RR
#    Lotaje = riesgo € / risk_pts
#
# 5. 1 trade por día · SL/TP gestionados por MT5
# ══════════════════════════════════════════════════════════


class StupidSimpleBot:

    def __init__(self):
        self.or_high          = None
        self.or_low           = None
        self.midpoint         = None
        self.or_close_dt      = None
        self.breakout_dir     = None   # "bull" | "bear" | None
        self.last_candle_time = None
        self.traded           = False

    # ──────────────────────────────────────────────────────
    # INIT MT5
    # ──────────────────────────────────────────────────────
    def init(self):
        if not mt5.initialize():
            print("❌ MT5 init failed")
            return False
        mt5.symbol_select(SYMBOL, True)
        acc = mt5.account_info()
        print(f"✅ Conectado | Cuenta: {acc.login} | Balance: {acc.balance:.2f} {acc.currency}")
        return True

    # ──────────────────────────────────────────────────────
    # OBTENER VELAS M5
    # ──────────────────────────────────────────────────────
    def get_m5(self, n=10):
        r = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M5, 0, n)
        if r is None:
            return None
        df = pd.DataFrame(r)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        return df

    # ──────────────────────────────────────────────────────
    # CALCULAR LOTAJE
    # ──────────────────────────────────────────────────────
    def calc_lot(self, entry, sl):
        acc      = mt5.account_info()
        risk_amt = acc.balance * RISK_PCT
        risk_pts = abs(entry - sl)
        if risk_pts <= 0:
            return 0
        lot  = risk_amt / risk_pts
        info = mt5.symbol_info(SYMBOL)
        lot  = max(info.volume_min, round(lot / info.volume_step) * info.volume_step)
        lot  = min(lot, info.volume_max)
        return round(lot, 2)

    # ──────────────────────────────────────────────────────
    # POSICIÓN ABIERTA
    # ──────────────────────────────────────────────────────
    def has_open_position(self):
        pos = mt5.positions_get(symbol=SYMBOL)
        return pos is not None and len(pos) > 0

    # ──────────────────────────────────────────────────────
    # ABRIR TRADE
    # ──────────────────────────────────────────────────────
    def open_trade(self, direction, retest_candle):
        if self.has_open_position():
            print("⛔ Posición ya abierta → skip")
            return False

        tick = mt5.symbol_info_tick(SYMBOL)

        if direction == "bull":
            entry      = self.or_high
            sl         = self.midpoint
            risk_pts   = abs(entry - sl)
            tp         = entry + risk_pts * RR
            px         = tick.ask
            order_type = mt5.ORDER_TYPE_BUY
        else:
            entry      = self.or_low
            sl         = self.midpoint
            risk_pts   = abs(entry - sl)
            tp         = entry - risk_pts * RR
            px         = tick.bid
            order_type = mt5.ORDER_TYPE_SELL

        slippage = abs(px - entry)
        rr_real  = abs(tp - px) / abs(px - sl) if abs(px - sl) > 0 else 0

        lot = self.calc_lot(entry, sl)
        if lot <= 0:
            print("⚠️  Lotaje inválido → skip")
            return False

        req = {
            "action"      : mt5.TRADE_ACTION_DEAL,
            "symbol"      : SYMBOL,
            "volume"      : lot,
            "type"        : order_type,
            "price"       : px,
            "sl"          : round(sl, 2),
            "tp"          : round(tp, 2),
            "deviation"   : 30,
            "magic"       : MAGIC,
            "comment"     : "SS_2R",
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        r = mt5.order_send(req)

        if r.retcode == mt5.TRADE_RETCODE_DONE:
            acc = mt5.account_info()
            print(
                f"\n{'═'*52}\n"
                f"🚀 TRADE EJECUTADO — {'BUY' if direction == 'bull' else 'SELL'}\n"
                f"   Retest vela : {retest_candle['time'].strftime('%H:%M')} MT5\n"
                f"   Entry nivel : {entry:.2f}  (OR {'high' if direction=='bull' else 'low'})\n"
                f"   Precio real : {px:.2f}  (slippage: {slippage:.1f} pts)\n"
                f"   SL          : {sl:.2f}  (midpoint OR, {risk_pts:.1f} pts)\n"
                f"   TP          : {tp:.2f}  ({RR}R teórico | {rr_real:.2f}R real)\n"
                f"   Lotes       : {lot}\n"
                f"   Riesgo €    : {acc.balance * RISK_PCT:.2f} {acc.currency}\n"
                f"{'═'*52}"
            )
            return True
        else:
            print(f"❌ Error orden: {r.retcode} — {r.comment}")
            return False

    # ──────────────────────────────────────────────────────
    # PASO 1: ESPERAR Y FIJAR OR
    # ──────────────────────────────────────────────────────
    def wait_and_set_or(self):
        print(f"\n⏳ Esperando vela OR de las {OR_TIME} (hora MT5)...")

        while True:
            df = self.get_m5(5)
            if df is None or len(df) < 2:
                t.sleep(5)
                continue

            c = df.iloc[-2]

            if c["time"].time() == OR_TIME:
                or_range = float(c["high"]) - float(c["low"])

                print(f"\n📊 Vela OR → {c['time'].strftime('%Y-%m-%d %H:%M')} MT5")
                print(f"   O:{c['open']:.2f}  H:{c['high']:.2f}  L:{c['low']:.2f}  C:{c['close']:.2f}")
                print(f"   Rango: {or_range:.1f} pts")

                if or_range < MIN_OR_RANGE:
                    print(f"⛔ OR demasiado pequeño ({or_range:.1f} pts) → sin operación hoy")
                    return False

                self.or_high      = float(c["high"])
                self.or_low       = float(c["low"])
                self.midpoint     = round((self.or_high + self.or_low) / 2, 2)
                self.or_close_dt  = c["time"] + timedelta(minutes=5)
                self.breakout_dir = None
                self.traded       = False
                self.last_candle_time = None

                print(f"🔒 OR fijado:")
                print(f"   OR high  : {self.or_high:.2f}")
                print(f"   OR low   : {self.or_low:.2f}")
                print(f"   Midpoint : {self.midpoint:.2f}  ← SL")
                print(f"   Rango    : {or_range:.1f} pts  → risk = {or_range/2:.1f} pts | TP dist = {or_range:.1f} pts")
                return True

            t.sleep(5)

    # ──────────────────────────────────────────────────────
    # HELPER: evaluar si una vela es break candle
    # ──────────────────────────────────────────────────────
    def _check_break(self, open_, close):
        """Devuelve 'bull', 'bear' o None según si la vela es break candle."""
        if open_ <= self.or_high and close > self.or_high:
            return "bull"
        if open_ >= self.or_low and close < self.or_low:
            return "bear"
        return None

    # ──────────────────────────────────────────────────────
    # PASO 2+3: SESIÓN (breakout + retest en M5)
    # ──────────────────────────────────────────────────────
    def run_session(self):
        print(f"\n👀 Monitorizando M5 desde las {self.or_close_dt.strftime('%H:%M')} MT5...")
        print(f"   Break válido: body cruza OR_high/OR_low desde dentro")
        print(f"   Fake out: si la misma vela rompe en dirección contraria → break inverso")

        while True:
            if self.traded:
                print("✅ Trade ejecutado → fin sesión")
                return

            df = self.get_m5(10)
            if df is None or len(df) < 2:
                t.sleep(5)
                continue

            c = df.iloc[-2]

            if self.last_candle_time == c["time"]:
                t.sleep(3)
                continue
            self.last_candle_time = c["time"]

            if c["time"] < self.or_close_dt:
                t.sleep(3)
                continue

            close = float(c["close"])
            high  = float(c["high"])
            low   = float(c["low"])
            open_ = float(c["open"])

            print(f"\n--- M5 {c['time'].strftime('%H:%M')} | "
                  f"O:{open_:.2f} H:{high:.2f} L:{low:.2f} C:{close:.2f} ---")

            # ── MODO RETEST ────────────────────────────────────────────
            if self.breakout_dir:

                if self.breakout_dir == "bull":
                    touches  = low  <= self.or_high
                    confirms = close > self.or_high

                    if touches and confirms:
                        print(f"✅ RETEST BULL confirmado")
                        print(f"   low:{low:.2f} ≤ OR_high:{self.or_high:.2f} "
                              f"y close:{close:.2f} > OR_high:{self.or_high:.2f}")
                        if self.open_trade("bull", c):
                            self.traded = True
                        self.breakout_dir = None

                    elif touches and not confirms:
                        # Fake out — ¿la misma vela rompe en dirección contraria?
                        inverse = self._check_break(open_, close)
                        if inverse == "bear":
                            print(f"⚠️  FAKE OUT bull — cerró dentro ({close:.2f} ≤ {self.or_high:.2f})")
                            print(f"   ↩️  Misma vela es BREAK BEAR → esperando retest OR_low ({self.or_low:.2f})...")
                            self.breakout_dir = "bear"
                        else:
                            print(f"⚠️  FAKE OUT bull — cerró dentro ({close:.2f} ≤ {self.or_high:.2f}) "
                                  f"→ breakout invalidado, esperando nueva señal")
                            self.breakout_dir = None

                    else:
                        print(f"   ⏳ Esperando retest OR_high ({self.or_high:.2f})...")

                else:  # bear
                    touches  = high >= self.or_low
                    confirms = close < self.or_low

                    if touches and confirms:
                        print(f"✅ RETEST BEAR confirmado")
                        print(f"   high:{high:.2f} ≥ OR_low:{self.or_low:.2f} "
                              f"y close:{close:.2f} < OR_low:{self.or_low:.2f}")
                        if self.open_trade("bear", c):
                            self.traded = True
                        self.breakout_dir = None

                    elif touches and not confirms:
                        # Fake out — ¿la misma vela rompe en dirección contraria?
                        inverse = self._check_break(open_, close)
                        if inverse == "bull":
                            print(f"⚠️  FAKE OUT bear — cerró dentro ({close:.2f} ≥ {self.or_low:.2f})")
                            print(f"   ↩️  Misma vela es BREAK BULL → esperando retest OR_high ({self.or_high:.2f})...")
                            self.breakout_dir = "bull"
                        else:
                            print(f"⚠️  FAKE OUT bear — cerró dentro ({close:.2f} ≥ {self.or_low:.2f}) "
                                  f"→ breakout invalidado, esperando nueva señal")
                            self.breakout_dir = None

                    else:
                        print(f"   ⏳ Esperando retest OR_low ({self.or_low:.2f})...")

            # ── MODO BREAKOUT ──────────────────────────────────────────
            else:
                brk = self._check_break(open_, close)
                if brk == "bull":
                    print(f"🟢 BREAK CANDLE BULL — abrió dentro ({open_:.2f} ≤ {self.or_high:.2f}) "
                          f"y cerró fuera ({close:.2f})")
                    print(f"   Esperando retest del OR_high ({self.or_high:.2f})...")
                    self.breakout_dir = "bull"
                elif brk == "bear":
                    print(f"🔴 BREAK CANDLE BEAR — abrió dentro ({open_:.2f} ≥ {self.or_low:.2f}) "
                          f"y cerró fuera ({close:.2f})")
                    print(f"   Esperando retest del OR_low ({self.or_low:.2f})...")
                    self.breakout_dir = "bear"
                else:
                    print(f"   ⏳ Sin breakout válido — rango [{self.or_low:.2f} – {self.or_high:.2f}]")

            t.sleep(3)

    # ──────────────────────────────────────────────────────
    # RUN PRINCIPAL
    # ──────────────────────────────────────────────────────
    def run(self):
        print("=" * 52)
        print("🤖 STUPID SIMPLE BOT — SS 2R v3 | US100.cash")
        print(f"   OR        : vela M5 de las {OR_TIME} (MT5 UTC+2)")
        print(f"   Break     : body cruza OR_high/OR_low desde dentro")
        print(f"   Retest    : toca nivel y cierra fuera")
        print(f"   Fake out  : si rompe en dirección contraria → break inverso")
        print(f"   Entry     : OR_high (bull) / OR_low (bear) — nivel fijo")
        print(f"   SL        : midpoint del OR")
        print(f"   TP        : {RR}R desde entry nivel fijo")
        print(f"   Riesgo    : {RISK_PCT*100:.2f}% | MIN_OR: {MIN_OR_RANGE} pts")
        print("=" * 52)

        ok = self.wait_and_set_or()
        if not ok:
            mt5.shutdown()
            return

        self.run_session()

        if self.traded:
            print("\n✅ Sesión completada con trade ejecutado")
            print("   SL/TP ya fijados en MT5")
        else:
            print("\n⏹️  Sin trade hoy — sin señal válida")

        mt5.shutdown()
        print("🔌 MT5 desconectado")


# ══════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════
if __name__ == "__main__":
    bot = StupidSimpleBot()
    if bot.init():
        bot.run()