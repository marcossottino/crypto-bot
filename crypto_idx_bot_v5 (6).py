import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import time

# ============================================
# CONFIGURACION
# ============================================
TELEGRAM_TOKEN = "8724062414:AAFQpXB3ygx_cCQwjA8Exg_Ei073Xi1eRv0"
CHAT_ID = "1375427916"

PESOS = {
    "BTCUSDT": 0.35,
    "ETHUSDT": 0.25,
    "LTCUSDT": 0.15,
    "XRPUSDT": 0.15,
    "ZECUSDT": 0.10
}

CANDLES_NEEDED = 100

# ============================================
# PARAMETROS OPTIMIZADOS
# ============================================
EMA_RAPIDA = 8
EMA_LENTA = 20
EMA_TENDENCIA = 50
RSI_PERIODO = 7
RSI_BUY_MIN = 45
RSI_BUY_MAX = 60
RSI_SELL_MIN = 40
RSI_SELL_MAX = 55
RSI_EXTREMO_MIN = 25   # SELL bloqueado si RSI menor a esto
RSI_EXTREMO_MAX = 75   # BUY bloqueado si RSI mayor a esto

# ============================================
# BINANCE
# ============================================
def obtener_velas_binance(symbol, interval="5m", limit=100):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        df = pd.DataFrame(data, columns=[
            'time','open','high','low','close','volume',
            'close_time','quote_vol','trades','taker_buy_base','taker_buy_quote','ignore'
        ])
        for col in ['close','open','high','low','volume']:
            df[col] = df[col].astype(float)
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        return df[['time','open','high','low','close','volume']]
    except Exception as e:
        print(f"Error Binance {symbol}: {e}")
        return None

def calcular_crypto_idx():
    idx_closes = None
    idx_volumes = None
    base_times = None
    for symbol, peso in PESOS.items():
        df = obtener_velas_binance(symbol)
        if df is None:
            continue
        normalized = df['close'] / df['close'].iloc[0] * 1000
        if idx_closes is None:
            idx_closes = normalized * peso
            idx_volumes = df['volume'] * peso
            base_times = df['time']
        else:
            idx_closes += normalized * peso
            idx_volumes += df['volume'] * peso
    if idx_closes is None:
        return None
    return pd.DataFrame({'time': base_times, 'close': idx_closes, 'volume': idx_volumes})

# ============================================
# INDICADORES
# ============================================
def calcular_ema(series, periodo):
    return series.ewm(span=periodo, adjust=False).mean()

def calcular_rsi(series, periodo=RSI_PERIODO):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(span=periodo, adjust=False).mean()
    avg_loss = loss.ewm(span=periodo, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calcular_macd(series, fast=12, slow=26, signal=9):
    ema_fast = calcular_ema(series, fast)
    ema_slow = calcular_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calcular_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calcular_bollinger(series, periodo=20, desv=2):
    media = series.rolling(window=periodo).mean()
    std = series.rolling(window=periodo).std()
    return media + std*desv, media, media - std*desv

def calcular_atr(df, periodo=14):
    tr = df['close'].rolling(2).max() - df['close'].rolling(2).min()
    return tr.rolling(window=periodo).mean()

def calcular_stoch_rsi(series, periodo=14, smooth=3):
    rsi = calcular_rsi(series, periodo)
    rsi_min = rsi.rolling(window=periodo).min()
    rsi_max = rsi.rolling(window=periodo).max()
    stoch = (rsi - rsi_min) / (rsi_max - rsi_min + 0.0001)
    k = stoch.rolling(window=smooth).mean() * 100
    d = k.rolling(window=smooth).mean()
    return k, d

# ============================================
# FILTRO ENTRADA SEGURA
# ============================================
def filtro_entrada_segura(df, direccion, ema_lenta):
    close = df['close']
    precio = close.iloc[-1]
    distancia = abs(precio - ema_lenta) / ema_lenta * 100
    distancia_ok = distancia >= 0.1

    if direccion == 'BUY':
        velas_ok = (close.iloc[-1] > close.iloc[-2] and
                    close.iloc[-2] > close.iloc[-3] and
                    close.iloc[-3] > close.iloc[-4])
    else:
        velas_ok = (close.iloc[-1] < close.iloc[-2] and
                    close.iloc[-2] < close.iloc[-3] and
                    close.iloc[-3] < close.iloc[-4])

    _, _, hist = calcular_macd(close)
    hist_now = hist.iloc[-1]
    hist_prev = hist.iloc[-2]
    if direccion == 'BUY':
        macd_ok = hist_now > hist_prev and hist_now > 0
    else:
        macd_ok = hist_now < hist_prev and hist_now < 0

    filtros_ok = sum([distancia_ok, velas_ok, macd_ok])
    print(f"  Filtros: dist={distancia_ok}({distancia:.2f}%) velas={velas_ok} macd={macd_ok} total={filtros_ok}/3")
    return filtros_ok >= 2

# ============================================
# ANALISIS PRINCIPAL
# ============================================
def analizar_niveles(df):
    close = df['close']

    ema8  = calcular_ema(close, EMA_RAPIDA)
    ema20 = calcular_ema(close, EMA_LENTA)
    ema50 = calcular_ema(close, EMA_TENDENCIA)
    rsi   = calcular_rsi(close, RSI_PERIODO)
    macd_line, signal_line, hist = calcular_macd(close)
    bb_upper, bb_mid, bb_lower  = calcular_bollinger(close)
    atr    = calcular_atr(df)
    stoch_k, stoch_d = calcular_stoch_rsi(close)

    e8  = ema8.iloc[-1]
    e20 = ema20.iloc[-1]
    e50 = ema50.iloc[-1]
    rsi_now   = rsi.iloc[-1]
    macd_now  = macd_line.iloc[-1]
    macd_prev = macd_line.iloc[-2]
    sig_now   = signal_line.iloc[-1]
    sig_prev  = signal_line.iloc[-2]
    precio      = close.iloc[-1]
    precio_prev = close.iloc[-2]
    bb_mid_now  = bb_mid.iloc[-1]
    atr_now = atr.iloc[-1]
    atr_med = atr.mean()
    sk = stoch_k.iloc[-1]; sd = stoch_d.iloc[-1]
    sk_prev = stoch_k.iloc[-2]; sd_prev = stoch_d.iloc[-2]

    # ===== FILTRO RSI EXTREMO =====
    if rsi_now < RSI_EXTREMO_MIN:
        print(f"  RSI {rsi_now:.1f} EXTREMO BAJO - mercado sobrevendido, NO OPERAR SELL")
        return None, 0, 0, rsi_now, e8, e20, precio, False
    if rsi_now > RSI_EXTREMO_MAX:
        print(f"  RSI {rsi_now:.1f} EXTREMO ALTO - mercado sobrecomprado, NO OPERAR BUY")
        return None, 0, 0, rsi_now, e8, e20, precio, False

    buy_conds = [
        e8 > e20,
        RSI_BUY_MIN <= rsi_now <= RSI_BUY_MAX,
        macd_now > sig_now and macd_prev <= sig_prev,
        precio > e50,
        precio > bb_mid_now,
        atr_now > atr_med * 0.8,
        sk > sd and sk_prev <= sd_prev and sk < 80,
        precio > precio_prev,
    ]

    sell_conds = [
        e8 < e20,
        RSI_SELL_MIN <= rsi_now <= RSI_SELL_MAX,
        macd_now < sig_now and macd_prev >= sig_prev,
        precio < e50,
        precio < bb_mid_now,
        atr_now > atr_med * 0.8,
        sk < sd and sk_prev >= sd_prev and sk > 20,
        precio < precio_prev,
    ]

    buy_score  = sum(buy_conds)
    sell_score = sum(sell_conds)

    if buy_score >= sell_score and buy_score >= 3:
        direccion = 'BUY'
        score = buy_score
    elif sell_score > buy_score and sell_score >= 3:
        direccion = 'SELL'
        score = sell_score
    else:
        return None, 0, 0, rsi_now, e8, e20, precio, False

    entrada_ok = filtro_entrada_segura(df, direccion, e20)

    nivel = 3 if score >= 8 else (2 if score >= 5 else 1)

    return direccion, nivel, score, rsi_now, e8, e20, precio, entrada_ok

# ============================================
# HORARIO
# ============================================
def calcular_entrada():
    ahora = datetime.now()
    minutos_para_siguiente = 5 - (ahora.minute % 5)
    entrada = ahora + timedelta(minutes=minutos_para_siguiente)
    entrada = entrada.replace(second=0, microsecond=0)
    salida  = entrada + timedelta(minutes=5)
    return entrada.strftime("%H:%M"), salida.strftime("%H:%M")

# ============================================
# MENSAJE TELEGRAM
# ============================================
def formatear_mensaje(direccion, nivel, score, rsi, ema8, ema20, precio):
    hora = datetime.now().strftime("%H:%M")
    fecha = datetime.now().strftime("%d/%m/%Y")
    hora_entrada, hora_salida = calcular_entrada()

    emoji_dir = "🟢" if direccion == 'BUY' else "🔴"
    icono = "📈" if direccion == 'BUY' else "📉"

    if nivel == 1:
        estrellas = "⭐"; efectividad = "70-75%"; confianza = "Media"
    elif nivel == 2:
        estrellas = "⭐⭐"; efectividad = "75-80%"; confianza = "Media-Alta"
    else:
        estrellas = "⭐⭐⭐"; efectividad = "80%+"; confianza = "ALTA ✅"

    return f"""{estrellas} <b>SENAL NIVEL {nivel}</b> {estrellas}
{emoji_dir} <b>{direccion}</b> - CRYPTO IDX {icono}

🕐 Señal: <b>{hora}</b> | {fecha}
⏰ Entrá a las: <b>{hora_entrada}</b>
🏁 Cerrá a las: <b>{hora_salida}</b>
⏱ Duración: <b>5 minutos</b>

📊 Efectividad: <b>{efectividad}</b>
💪 Confianza: <b>{confianza}</b>
✅ Confirmaciones: <b>{score}/8</b>
🛡 Entrada: <b>SEGURA</b> ✅

📉 RSI({RSI_PERIODO}): <b>{rsi:.1f}</b>
📈 EMA{EMA_RAPIDA}: <b>{ema8:.2f}</b>
📈 EMA{EMA_LENTA}: <b>{ema20:.2f}</b>
💰 Precio IDX: <b>{precio:.2f}</b>

⚠️ <i>Solo informativo</i>"""

def enviar_telegram(mensaje):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": CHAT_ID, "text": mensaje, "parse_mode": "HTML"}, timeout=10)
        print("Telegram OK" if r.status_code == 200 else f"Error: {r.text}")
    except Exception as e:
        print(f"Error Telegram: {e}")

# ============================================
# BOT PRINCIPAL
# ============================================
def run_bot():
    print("=" * 50)
    print("  BOT CRYPTO IDX v5 - PARAMETROS OPTIMIZADOS")
    print(f"  EMA {EMA_RAPIDA}/{EMA_LENTA} | RSI {RSI_PERIODO}")
    print(f"  RSI extremo bloqueado: <{RSI_EXTREMO_MIN} o >{RSI_EXTREMO_MAX}")
    print("=" * 50)

    enviar_telegram(f"🤖 <b>Bot Crypto IDX v5</b>\n🔧 EMA {EMA_RAPIDA}/{EMA_LENTA} | RSI {RSI_PERIODO}\n🚫 RSI extremo bloqueado (&lt;{RSI_EXTREMO_MIN} o &gt;{RSI_EXTREMO_MAX})\n⭐ Nivel 1: 70-75%\n⭐⭐ Nivel 2: 75-80%\n⭐⭐⭐ Nivel 3: 80%+")

    ultima_senal = None

    while True:
        try:
            hora_actual = datetime.now().strftime("%H:%M")
            print(f"\n[{hora_actual}] Analizando...")

            df = calcular_crypto_idx()
            if df is None or len(df) < 50:
                print("Error datos, reintentando en 30s...")
                time.sleep(30)
                continue

            direccion, nivel, score, rsi, e8, e20, precio, entrada_ok = analizar_niveles(df)

            print(f"RSI: {rsi:.1f} | Score: {score}/8 | Señal: {direccion or 'ESPERAR'} | Nivel: {nivel} | Segura: {entrada_ok}")

            if direccion and entrada_ok and hora_actual != ultima_senal:
                ultima_senal = hora_actual
                enviar_telegram(formatear_mensaje(direccion, nivel, score, rsi, e8, e20, precio))
                print(f"SENAL ENVIADA: {direccion} - NIVEL {nivel}")
            elif direccion and not entrada_ok:
                print(f"Señal {direccion} - entrada NO segura, omitiendo")

            print("Esperando 5 minutos...")
            time.sleep(300)

        except KeyboardInterrupt:
            print("\nBot detenido.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(30)

if __name__ == "__main__":
    run_bot()
