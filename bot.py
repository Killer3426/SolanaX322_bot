import logging
import requests
import json
import time
import threading
import schedule
import torch
import torch.nn as nn
import torch.optim as optim
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
from datetime import datetime, timedelta
import asyncio  # Добавлено для v21

# Настройка логирования
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Ключи (из env vars)
TELEGRAM_TOKEN = 'YOUR_TELEGRAM_BOT_TOKEN_HERE'  # Render подставит из env
MORALIS_API_KEY = 'YOUR_MORALIS_API_KEY_HERE'

# Файлы для persistence
SUBSCRIBERS_FILE = 'subscribers.txt'
HISTORICAL_FILE = 'historical_tokens.json'

# ML модель: Простая NN для scoring (features: liquidity, holders, price_change)
class TokenScorer(nn.Module):
    def __init__(self):
        super(TokenScorer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Score 0-1
        )
    
    def forward(self, x):
        return self.fc(x)

model = TokenScorer()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Загрузка/сохранение исторических данных и модели
def load_historical():
    try:
        with open(HISTORICAL_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_historical(data):
    with open(HISTORICAL_FILE, 'w') as f:
        json.dump(data, f)

# Тренировка модели на исторических (предполагаем labels: 1 если price_change > 10%, else 0)
def train_model():
    historical = load_historical()
    if len(historical) < 10:
        logger.info("Not enough data for training.")
        return
    
    features = []
    labels = []
    for token in historical:
        liq = float(token.get('usd_liquidity', 0))
        holders = token.get('holders_count', 0)
        change = float(token.get('price_change', 0))
        features.append([liq, holders, change])
        labels.append(1 if change > 0.1 else 0)  # Пример: >10% рост = топ
    
    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    
    logger.info("Model trained.")

# Score токена с ML
def get_ml_score(token):
    liq = float(token.get('usd_liquidity', 0))
    holders = token.get('holders_count', 0)
    change = float(token.get('price_change', 0))
    input_tensor = torch.tensor([[liq, holders, change]], dtype=torch.float32)
    with torch.no_grad():
        score = model(input_tensor).item()
    return score

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

# Фильтрация потенциального топа (security + ML score)
def is_potential_top_token(token_data, metadata):
    security = metadata.get('security', {})
    if (
        not security.get('is_honeypot', False) and
        float(security.get('buy_tax', 0)) < 0.1 and
        float(security.get('sell_tax', 0)) < 0.1 and
        not security.get('cannot_sell_all', False) and
        security.get('is_open_source', True)
    ):
        score = get_ml_score(token_data)
        return score > 0.7  # ML-фильтр
    return False

# Основная функция поиска
def find_tokens():
    tokens = get_new_pumpfun_tokens(limit=20)  # Больше для фильтра
    historical = load_historical()
    new_historical = historical.copy()
    filtered = []
    
    for token in tokens:
        address = token['address']
        metadata = get_token_metadata(address)
        if metadata and is_potential_top_token(token, metadata):
            filtered.append((token, metadata))
        
        # Добавляем в historical для ML
        if address not in [t.get('address', '') for t in historical]:
            new_historical.append(token)  # Добавляем price_change etc. later if needed
    
    save_historical(new_historical[:500])  # Limit history
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
    return message if filtered else "Нет подходящих токенов сейчас."

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
        message = "Новые потенциальные топы на pump.fun:\n\n" + format_tokens([list(t) for t in recent_filtered])  # Адаптация для list/tuple
        subscribers = load_subscribers()
        for chat_id in subscribers:
            try:
                asyncio.run_coroutine_threadsafe(application.bot.send_message(chat_id=chat_id, text=message), application.loop)
            except Exception as e:
                logger.error(f"Error sending to {chat_id}: {e}")
    
    last_check = datetime.now()

# Schedule задачи (адаптировано для v21)
def run_schedule(application):
    schedule.every(5).minutes.do(check_and_notify, application=application)
    schedule.every(30).minutes.do(train_model)
    
    while True:
        schedule.run_pending()
        time.sleep(1)

# Основная функция (обновлено для v21: без Updater)
def main():
    global TELEGRAM_TOKEN, MORALIS_API_KEY  # Для env vars
    # Подставляем env vars (Render их экспортирует)
    import os
    TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN', TELEGRAM_TOKEN)
    MORALIS_API_KEY = os.getenv('MORALIS_API_KEY', MORALIS_API_KEY)
    
    if 'YOUR_' in TELEGRAM_TOKEN:
        logger.error("TELEGRAM_TOKEN not set!")
        return
    
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("find", find))
    application.add_handler(CommandHandler("subscribe", subscribe))
    
    # Неизвестные
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, lambda update, context: update.message.reply_text('Используй /find или /subscribe.')))
    
    # Запуск schedule в фоне
    threading.Thread(target=run_schedule, args=(application,), daemon=True).start()
    
    # Начальная тренировка
    train_model()
    
    # Запуск polling (v21 стиль)
    logger.info("Bot started")
    application.run_polling()

if __name__ == '__main__':
    main()
