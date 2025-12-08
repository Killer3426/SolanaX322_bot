import logging, requests, json, time, threading, schedule, os
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
MORALIS_API_KEY = os.getenv('MORALIS_API_KEY')

# Временные переменные (сбрасываются при рестарте — но это нормально)
subscribers = set()
last_check = datetime.now() - timedelta(minutes=5)

def get_new_tokens():
    url = f"https://solana-gateway.moralis.io/token/mainnet/exchange/pumpfun/new?limit=30"
    headers = {"X-API-Key": MORALIS_API_KEY}
    r = requests.get(url, headers=headers)
    return r.json() if r.status_code == 200 else []

def get_metadata(addr):
    url = f"https://solana-gateway.moralis.io/token/mainnet/{addr}"
    headers = {"X-API-Key": MORALIS_API_KEY}
    r = requests.get(url, headers=headers)
    return r.json() if r.status_code == 200 else None

def is_good_token(token, meta):
    s = meta.get('security', {})
    return (not s.get('is_honeypot', True) and
            float(s.get('buy_tax', 99)) < 8 and
            float(s.get('sell_tax', 99)) < 8 and
            s.get('is_open_source', False) and
            token.get('holders_count', 0) > 80 and
            float(token.get('volume_24h', 0)) > 15000)

def find_and_send():
    global last_check
    tokens = get_new_tokens()
    good = []
    for t in tokens:
        addr = t.get('address')
        if not addr: continue
        meta = get_metadata(addr)
        if meta and is_good_token(t, meta):
            created = datetime.fromtimestamp(t.get('created_timestamp',0)/1000)
            if created > last_check:
                good.append(t)
    if good:
        msg = "НОВЫЕ ТОПЫ PUMP.FUN\n\n"
        for t in good[:7]:
            msg += f"{t.get('name','?')} (${t.get('symbol','?')})\n"
            msg += f"Цена: ${t.get('usd_price','?')}\n"
            msg += f"Холдеры: {t.get('holders_count',0)}\n"
            msg += f"DexScreener: https://dexscreener.com/solana/{t['address']}\n\n"
        for chat_id in subscribers:
            try: application.bot.send_message(chat_id, msg)
            except: pass
    last_check = datetime.now()

async def start(update: Update, context):
    await update.message.reply_text('Бот живой! /find — свежие токены\n/subscribe — алерты каждые 5 мин')

async def find(update: Update, context):
    tokens = get_new_tokens()
    good = []
    for t in tokens:
        meta = get_metadata(t.get('address'))
        if meta and is_good_token(t, meta):
            good.append(t)
    msg = "Текущие топы:\n\n" if good else "Пока тихо..."
    for t in good[:10]:
        msg += f"{t.get('name','?')} (${t.get('symbol','?')})\nДекс: https://dexscreener.com/solana/{t['address']}\n\n"
    await update.message.reply_text(msg)

async def subscribe(update: Update, context):
    subscribers.add(update.message.chat_id)
    await update.message.reply_text('Подписан! Буду присылать новые токены автоматически.')

def auto_check(app):
    schedule.every(5).minutes.do(find_and_send)
    while True:
        schedule.run_pending()
        time.sleep(1)

application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
application.add_handler(CommandHandler("start", start))
application.add_handler(CommandHandler("find", find))
application.add_handler(CommandHandler("subscribe", subscribe))

threading.Thread(target=auto_check, args=(application,), daemon=True).start()

logger.info("Бот запущен — максимальная простота и надёжность")
application.run_polling() 
