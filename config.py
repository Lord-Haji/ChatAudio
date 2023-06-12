import os

from dotenv import load_dotenv, find_dotenv

# Check if a .env file exists
if os.path.exists('.env'):
    load_dotenv(find_dotenv())


API_KEY = os.getenv("OPENAI_API_KEY")

print(API_KEY)

DATABASE_FILE = 'transcriptions.db'