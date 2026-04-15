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
OR_END_TIME      = time(16, 34)  # OR = velas M1 16:30–16:34 (5 velas)
MT5_OFFSET_HOURS = 2
MIN_OR_RANGE     = 5.0

# ══════════════════════════════════════════════════════════
# LÓGICA (Stupid Simple 2R) — v4 M1
# ══════════════════════════════════════════════════════════
# 1. OR = max high / min low velas M1 16:30–16:34 MT5
#    Si el bot arranca tarde (>16:34), recupera el OR del
#    histórico reciente sin esperar a mañana.
#
# 2. BREAK CANDLE M1:
#      BULL: open <= OR_high  Y  close > OR_high
#      BEAR: open >= OR_low   Y  close < OR_low
#
# 3. RETEST M1: toca nivel + cierra fuera → entry
#    FAKE OUT: si misma vela rompe en dirección contraria
#              → break inverso, si no → reset
#
# 4. ENTRY = OR_high (bull) / OR_low (bear) — nivel fijo
#    SL     = midpoint del OR
#    TP     = entry ± (entry - midpoint) × RR
#
# 5. 1 trade por día · SL/TP gestionados por MT5
# ══════════════════════════════════════════════════════════


class StupidSimpleBot:

    def __init__(self):
        self.or_high          = None
        self.or_low           = None
        self.midpoint         = None
        self.or_close_dt      = None
        self.breakout_dir     = None
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
    # OBTENER VELAS M1
    # ──────────────────────────────────────────────────────
    def get_m1(self, n=10):
        r = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M1, 0, n)
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
    # HELPER: evaluar si una vela M1 es break candle
    # ──────────────────────────────────────────────────────
    def _check_break(self, open_, close):
        if open_ <= self.or_high and close > self.or_high:
            return "bull"
        if open_ >= self.or_low and close < self.or_low:
            return "bear"
        return None

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
            "comment"     : "SS_2R_M1",
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        r = mt5.order_send(req)

        if r.retcode == mt5.TRADE_RETCODE_DONE:
            acc = mt5.account_info()
            print(
                f"\n{'═'*52}\n"
                f"🚀 TRADE EJECUTADO — {'BUY' if direction == 'bull' else 'SELL'}\n"
                f"   Retest vela : {retest_candle['time'].strftime('%H:%M')} MT5 (M1)\n"
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
    # PASO 1: FIJAR OR
    # Funciona tanto si arranca antes de las 16:30 (espera)
    # como si arranca tarde (recupera OR del histórico)
    # ──────────────────────────────────────────────────────
    def set_or(self):
        print(f"\n⏳ Buscando OR del día (velas M1 {OR_TIME}–{OR_END_TIME} MT5)...")

        while True:
            # Pedir suficientes velas para cubrir la sesión de hoy
            df = self.get_m1(120)
            if df is None or len(df) < 6:
                t.sleep(5)
                continue

            now_time = df.iloc[-1]["time"].time()

            # ── Caso 1: todavía no son las 16:34 → esperar ──────────
            if now_time < OR_END_TIME:
                print(f"   Hora actual: {now_time.strftime('%H:%M')} — esperando cierre del OR ({OR_END_TIME})...")
                t.sleep(10)
                continue

            # ── Caso 2: ya pasó la hora → buscar OR en histórico ────
            or_candles = df[
                (df["time"].dt.time >= OR_TIME) &
                (df["time"].dt.time <= OR_END_TIME)
            ]

            if len(or_candles) < 3:
                print(f"   No se encontraron suficientes velas del OR ({len(or_candles)}/5) — reintentando...")
                t.sleep(5)
                continue

            or_high  = float(or_candles["high"].max())
            or_low   = float(or_candles["low"].min())
            or_range = or_high - or_low

            print(f"\n📊 OR fijado (velas M1 {OR_TIME.strftime('%H:%M')}–{OR_END_TIME.strftime('%H:%M')})")
            print(f"   Velas OR encontradas: {len(or_candles)}/5")
            print(f"   OR high  : {or_high:.2f}  |  OR low: {or_low:.2f}")
            print(f"   Rango    : {or_range:.1f} pts")

            if or_range < MIN_OR_RANGE:
                print(f"⛔ OR demasiado pequeño ({or_range:.1f} pts) → sin operación hoy")
                return False

            self.or_high      = or_high
            self.or_low       = or_low
            self.midpoint     = round((or_high + or_low) / 2, 2)
            # Sesión desde 16:35 — si ya pasó, monitorizamos desde ahora
            self.or_close_dt  = or_candles["time"].iloc[-1] + timedelta(minutes=1)
            self.breakout_dir = None
            self.traded       = False
            self.last_candle_time = None

            print(f"   Midpoint : {self.midpoint:.2f}  ← SL")
            print(f"   Risk     : {or_range/2:.1f} pts | TP dist: {or_range:.1f} pts")

            if now_time > OR_END_TIME:
                print(f"   ⚡ Bot arrancado tarde ({now_time.strftime('%H:%M')}) — "
                      f"OR recuperado del histórico, monitorizando desde ahora")

            return True

    # ──────────────────────────────────────────────────────
    # PASO 2+3: SESIÓN (breakout + retest en M1)
    # ──────────────────────────────────────────────────────
    def run_session(self):
        print(f"\n👀 Monitorizando M1 desde las {self.or_close_dt.strftime('%H:%M')} MT5...")
        print(f"   Break válido : body cruza OR_high/OR_low desde dentro")
        print(f"   Fake out     : si misma vela rompe al contrario → break inverso")

        while True:
            if self.traded:
                print("✅ Trade ejecutado → fin sesión")
                return

            df = self.get_m1(10)
            if df is None or len(df) < 2:
                t.sleep(2)
                continue

            c = df.iloc[-2]   # última vela M1 cerrada

            if self.last_candle_time == c["time"]:
                t.sleep(1)
                continue
            self.last_candle_time = c["time"]

            # Si arrancó tarde, ignorar velas anteriores al OR
            if c["time"] < self.or_close_dt:
                t.sleep(1)
                continue

            close = float(c["close"])
            high  = float(c["high"])
            low   = float(c["low"])
            open_ = float(c["open"])

            print(f"\n--- M1 {c['time'].strftime('%H:%M')} | "
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
                        inverse = self._check_break(open_, close)
                        if inverse == "bear":
                            print(f"⚠️  FAKE OUT bull — cerró dentro ({close:.2f})")
                            print(f"   ↩️  Misma vela es BREAK BEAR → retest OR_low ({self.or_low:.2f})...")
                            self.breakout_dir = "bear"
                        else:
                            print(f"⚠️  FAKE OUT bull — cerró dentro ({close:.2f}) → reset")
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
                        inverse = self._check_break(open_, close)
                        if inverse == "bull":
                            print(f"⚠️  FAKE OUT bear — cerró dentro ({close:.2f})")
                            print(f"   ↩️  Misma vela es BREAK BULL → retest OR_high ({self.or_high:.2f})...")
                            self.breakout_dir = "bull"
                        else:
                            print(f"⚠️  FAKE OUT bear — cerró dentro ({close:.2f}) → reset")
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
                    print(f"   ⏳ Sin breakout — rango [{self.or_low:.2f} – {self.or_high:.2f}]")

            t.sleep(1)

    # ──────────────────────────────────────────────────────
    # RUN PRINCIPAL
    # ──────────────────────────────────────────────────────
    def run(self):
        print("=" * 52)
        print("🤖 STUPID SIMPLE BOT — SS 2R v4 M1 | US100.cash")
        print(f"   OR        : velas M1 {OR_TIME}–{OR_END_TIME} (MT5 UTC+2)")
        print(f"   Break     : body cruza OR_high/OR_low desde dentro")
        print(f"   Retest    : toca nivel y cierra fuera (M1)")
        print(f"   Fake out  : break inverso si aplica")
        print(f"   Entry     : OR_high (bull) / OR_low (bear) — nivel fijo")
        print(f"   SL        : midpoint del OR")
        print(f"   TP        : {RR}R desde entry nivel fijo")
        print(f"   Riesgo    : {RISK_PCT*100:.2f}% | MIN_OR: {MIN_OR_RANGE} pts")
        print(f"   ⚡ Arranque tardío: OR recuperado automáticamente del histórico")
        print("=" * 52)

        ok = self.set_or()
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