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
import os
import tweepy  # Для X sentiment

# Настройка логирования
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Ключи из env
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
MORALIS_API_KEY = os.getenv('MORALIS_API_KEY')
TWITTER_CONSUMER_KEY = os.getenv('TWITTER_CONSUMER_KEY')
TWITTER_CONSUMER_SECRET = os.getenv('TWITTER_CONSUMER_SECRET')
TWITTER_ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN')
TWITTER_ACCESS_SECRET = os.getenv('TWITTER_ACCESS_SECRET')

# Проверка ключей
if not all([TELEGRAM_TOKEN, MORALIS_API_KEY, TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET]):
    logger.error("Missing API keys! Add them in Render Environment.")
    exit(1)

# Twitter auth
auth = tweepy.OAuth1UserHandler(TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET)
api = tweepy.API(auth)

# Файлы (на Render Disk /data)
SUBSCRIBERS_FILE = '/data/subscribers.txt'
HISTORICAL_FILE = '/data/historical_tokens.json'
SETTINGS_FILE = '/data/settings.json'  # Для кастом фильтров

# ML модель
class TokenScorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 128),  # Features: liquidity, holders, volume, sentiment_score
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

model = TokenScorer()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.BCELoss()

# Загрузка/сохранение
def load_historical():
    try:
        with open(HISTORICAL_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_historical(data):
    with open(HISTORICAL_FILE, 'w') as f:
        json.dump(data, f)

def train_model():
    historical = load_historical()
    if len(historical) < 20:
        logger.info("Not enough data for training.")
        return

    features = []
    labels = []
    for token in historical:
        liq = float(token.get('usd_liquidity', 0))
        holders = token.get('holders_count', 0)
        volume = float(token.get('volume_24h', 0))
        sentiment = token.get('sentiment_score', 0)
        change = float(token.get('price_change', 0))
        features.append([liq, holders, volume, sentiment])
        labels.append(1 if change > 0.2 else 0)  # >20% рост = топ

    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    for epoch in range(200):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    logger.info("Model re-trained.")

# Sentiment с X
def get_sentiment_score(symbol, address):
    try:
        tweets = api.search_tweets(q=f"{symbol} OR {address} filter:safe", count=50, tweet_mode='extended')
        positive = sum(1 for tweet in tweets if tweet.favorite_count > 50 or 'pump' in tweet.full_text.lower())
        total = len(tweets)
        return positive / total if total > 0 else 0
    except Exception as e:
        logger.error(f"X API error: {e}")
        return 0

# Получение токенов
def get_new_pumpfun_tokens(limit=30):
    url = f"https://solana-gateway.moralis.io/token/mainnet/exchange/pumpfun/new?limit={limit}"
    headers = {"X-API-Key": MORALIS_API_KEY}
    response = requests.get(url, headers=headers)
    return response.json() if response.status_code == 200 else []

# Metadata
def get_token_metadata(address):
    url = f"https://solana-gateway.moralis.io/token/mainnet/{address}"
    headers = {"X-API-Key": MORALIS_API_KEY}
    response = requests.get(url, headers=headers)
    return response.json() if response.status_code == 200 else None

# Фильтрация
def is_potential_top_token(token_data, metadata, settings):
    security = metadata.get('security', {})
    if (
        not security.get('is_honeypot', False) and
        float(security.get('buy_tax', 0)) < 0.05 and  # Ужесточили
        float(security.get('sell_tax', 0)) < 0.05 and
        not security.get('cannot_sell_all', False) and
        security.get('is_open_source', True) and
        security.get('liquidity_locked', True)  # Новый чек
    ):
        holders = token_data.get('holders_count', 0)
        volume = float(token_data.get('volume_24h', 0))
        sentiment = get_sentiment_score(token_data.get('symbol'), token_data['address'])
        if holders > settings['min_holders'] and volume > settings['min_volume']:
            input_tensor = torch.tensor([[float(token_data.get('usd_liquidity', 0)), holders, volume, sentiment]], dtype=torch.float32)
            with torch.no_grad():
                score = model(input_tensor).item()
            return score > 0.75, sentiment
    return False, 0

# Поиск
def find_tokens(settings):
    tokens = get_new_pumpfun_tokens()
    historical = load_historical()
    new_historical = historical.copy()
    filtered = []

    for token in tokens:
        address = token.get('address')
        metadata = get_token_metadata(address)
        if metadata:
            is_top, sentiment = is_potential_top_token(token, metadata, settings)
            if is_top:
                token['sentiment_score'] = sentiment
                filtered.append((token, metadata))
            if address not in [t.get('address') for t in historical]:
                new_historical.append(token)

    save_historical(new_historical[:1000])
    return filtered

# Формат
def format_tokens(filtered):
    message = "🚀 Топ свежие мемкоины на pump.fun (максимум апгрейд: ML + X sentiment):\n\n"
    for token, metadata in filtered:
        name = token.get('name', 'Unknown')
        symbol = token.get('symbol', 'N/A')
        price = token.get('usd_price', 'N/A')
        created = datetime.fromtimestamp(token.get('created_timestamp', 0)/1000).strftime('%Y-%m-%d %H:%M')
        address = token['address']
        sentiment = token.get('sentiment_score', 0)
        message += f"📈 {name} ({symbol}) | Sentiment: {sentiment*100:.1f}%\nЦена: ${price}\nСоздан: {created}\nBirdeye: https://birdeye.so/token/{address}\nDexScreener: https://dexscreener.com/solana/{address}\nPhoton: https://photon-sol.tinyastro.io/en/r/{address}\n\n"
    return message if filtered else "Нет топов сейчас. Жди уведомлений!"

# Загрузка/сохранение настроек
def load_settings():
    try:
        with open(SETTINGS_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {'min_holders': 100, 'min_volume': 20000}

def save_settings(settings):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f)

# Команды
async def start(update: Update, context):
    await update.message.reply_text('🚀 Максимальный апгрейд бота! /find - поиск, /subscribe - алерты, /settings - настройка.')

async def find(update: Update, context):
    settings = load_settings()
    filtered = find_tokens(settings)
    message = format_tokens(filtered)
    await update.message.reply_text(message)

async def subscribe(update: Update, context):
    chat_id = update.message.chat_id
    subscribers = load_subscribers()
    if chat_id not in subscribers:
        subscribers.append(chat_id)
        save_subscribers(subscribers)
        await update.message.reply_text('Подписан на супер-алерты!')
    else:
        await update.message.reply_text('Уже подписан.')

async def settings(update: Update, context):
    args = context.args
    if len(args) == 2:
        key, value = args
        s = load_settings()
        if key in s:
            s[key] = int(value)
            save_settings(s)
            await update.message.reply_text(f'Настройка {key} = {value}')
        else:
            await update.message.reply_text('Ключи: min_holders, min_volume')
    else:
        s = load_settings()
        await update.message.reply_text(f'Текущие: {s}\nИспользуй /settings min_holders 200')

async def stats(update: Update, context):
    historical = load_historical()
    await update.message.reply_text(f'Поймано топов: {len(historical)} | ML обучен: Да')

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

# Уведомления
last_check = datetime.now() - timedelta(minutes=3)
def check_and_notify(application):
    global last_check
    settings = load_settings()
    filtered = find_tokens(settings)
    recent_filtered = [t for t in filtered if datetime.fromtimestamp(t[0].get('created_timestamp', 0)/1000) > last_check and float(t[0].get('price_change', 0)) > 2]  # >2x рост

    if recent_filtered:
        message = "🔥 Супер-алерт: Новый памп!\n\n" + format_tokens(recent_filtered)
        subscribers = load_subscribers()
        for chat_id in subscribers:
            try:
                application.bot.send_message(chat_id=chat_id, text=message)
            except Exception as e:
                logger.error(f"Error: {e}")

    last_check = datetime.now()

# Schedule
def run_schedule(application):
    schedule.every(3).minutes.do(lambda: check_and_notify(application))
    schedule.every(15).minutes.do(train_model)

    while True:
        schedule.run_pending()
        time.sleep(1)

# Main
def main():
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("find", find))
    application.add_handler(CommandHandler("subscribe", subscribe))
    application.add_handler(CommandHandler("settings", settings))
    application.add_handler(CommandHandler("stats", stats))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, lambda update, context: update.message.reply_text('Команды: /find, /subscribe, /settings')))

    threading.Thread(target=run_schedule, args=(application,), daemon=True).start()

    train_model()  # Начальная

    logger.info("Bot started - max upgrade!")
    application.run_polling()

if __name__ == '__main__':
    main()
