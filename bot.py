import logging
import requests
import json
import time
import threading
import schedule
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
from datetime import datetime, timedelta
import os  # Для env vars

# Настройка логирования
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Ключи (из env vars)
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN', 'YOUR_TELEGRAM_BOT_TOKEN_HERE')
MORALIS_API_KEY = os.getenv('MORALIS_API_KEY', 'YOUR_MORALIS_API_KEY_HERE')

# Файлы для persistence
SUBSCRIBERS_FILE = 'subscribers.txt'

# Получение новых токенов с pump.fun
def get_new_pumpfun_tokens(limit=10):
    url = f"https://solana-gateway.moralis.io/token/mainnet/exchange/pumpfun/new?limit={limit}"
    headers = {"X-API-Key": MORALIS_API_KEY}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        logger.error(f"Error: {response.text}")
        return []

# Получение metadata и security для токена
def get_token_metadata(address):
    url = f"https://solana-gateway.moralis.io/token/mainnet/{address}"
    headers = {"X-API-Key": MORALIS_API_KEY}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    return None

# Фильтрация потенциального топа (только security + базовые метрики, без ML)
def is_potential_top_token(token_data, metadata):
    security = metadata.get('security', {})
    # Базовые фильтры на скам
    if (
        not security.get('is_honeypot', False) and
        float(security.get('buy_tax', 0)) < 0.1 and
        float(security.get('sell_tax', 0)) < 0.1 and
        not security.get('cannot_sell_all', False) and
        security.get('is_open_source', True)
    ):
        # Дополнительно: базовый "score" по метрикам (адаптация под тренды)
        holders = token_data.get('holders_count', 0)
        volume = float(token_data.get('volume_24h', 0))
        if holders > 50 and volume > 10000:  # Простой фильтр на активность
            return True
    return False

# Основная функция поиска
def find_tokens():
    tokens = get_new_pumpfun_tokens(limit=20)  # Больше для фильтра
    filtered = []
    
    for token in tokens:
        address = token.get('address', '')
        if not address:
            continue
        metadata = get_token_metadata(address)
        if metadata and is_potential_top_token(token, metadata):
            filtered.append((token, metadata))
    
    return filtered

# Форматирование сообщения
def format_tokens(filtered):
    message = "Топ свежие потенциальные мемкоины на pump.fun (отфильтрованы от скама):\n\n"
    for token, metadata in filtered[:10]:
        name = token.get('name', 'Unknown')
        symbol = token.get('symbol', 'N/A')
        price = token.get('usd_price', 'N/A')
        created = token.get('created_timestamp', 'N/A')
        address = token['address']
        message += f"📈 {name} ({symbol})\nЦена: ${price}\nСоздан: {created}\nАдрес: {address}\nDexScreener: https://dexscreener.com/solana/{address}\n\n"
    return message if filtered else "Нет подходящих токенов сейчас. Попробуй позже!"

# Команда /start
async def start(update: Update, context):
    await update.message.reply_text('Привет! Бот для свежих мемкоинов на pump.fun. /find - поиск, /subscribe - уведомления.')

# Команда /find
async def find(update: Update, context):
    filtered = find_tokens()
    message = format_tokens(filtered)
    await update.message.reply_text(message)

# Подписка /subscribe
async def subscribe(update: Update, context):
    chat_id = update.message.chat_id
    subscribers = load_subscribers()
    if chat_id not in subscribers:
        subscribers.append(chat_id)
        save_subscribers(subscribers)
        await update.message.reply_text('Вы подписаны на уведомления о новых топовых токенах!')
    else:
        await update.message.reply_text('Вы уже подписаны.')

# Загрузка/сохранение подписчиков
def load_subscribers():
    try:
        with open(SUBSCRIBERS_FILE, 'r') as f:
            return [int(line.strip()) for line in f]
    except FileNotFoundError:
        return []

def save_subscribers(subs):
    with open(SUBSCRIBERS_FILE, 'w') as f:
        for s in subs:
            f.write(f"{s}\n")

# Фоновая задача: Проверка новых и уведомления
last_check = datetime.now() - timedelta(minutes=5)
def check_and_notify(application):
    global last_check
    filtered = find_tokens()
    recent_filtered = [t for t in filtered if datetime.fromtimestamp(t[0].get('created_timestamp', 0)/1000) > last_check]
    
    if recent_filtered:
        message = "Новые потенциальные топы на pump.fun:\n\n" + format_tokens(recent_filtered)
        subscribers = load_subscribers()
        for chat_id in subscribers:
            try:
                application.bot.send_message(chat_id=chat_id, text=message)
            except Exception as e:
                logger.error(f"Error sending to {chat_id}: {e}")
    
    last_check = datetime.now()

# Schedule задачи
def run_schedule(application):
    schedule.every(5).minutes.do(lambda: check_and_notify(application))
    while True:
        schedule.run_pending()
        time.sleep(1)

# Основная функция
def main():
    if 'YOUR_' in TELEGRAM_TOKEN:
        logger.error("TELEGRAM_TOKEN not set! Add it in Render Environment.")
        return
    
    if 'YOUR_' in MORALIS_API_KEY:
        logger.error("MORALIS_API_KEY not set! Add it in Render Environment.")
        return
    
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("find", find))
    application.add_handler(CommandHandler("subscribe", subscribe))
    
    # Неизвестные
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, lambda update, context: update.message.reply_text('Используй /find или /subscribe.')))
    
    # Запуск schedule в фоне
    threading.Thread(target=run_schedule, args=(application,), daemon=True).start()
    
    logger.info("Bot started")
    application.run_polling()

if __name__ == '__main__':
    main()
