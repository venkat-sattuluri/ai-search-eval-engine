import os
from dotenv import load_dotenv

load_dotenv()

anthropic_key = os.getenv("ANTHROPIC_API_KEY")
langfuse_pk = os.getenv("LANGFUSE_PUBLIC_KEY")
langfuse_sk = os.getenv("LANGFUSE_SECRET_KEY")

print("--- Environment Variable Check ---")
if anthropic_key and (anthropic_key.startswith("sk-ant-api") or anthropic_key.startswith("sk-ant-usr")):
    print("✅ ANTHROPIC_API_KEY is loaded and looks valid!")
else:
    print("❌ ANTHROPIC_API_KEY is missing or formatted incorrectly.")

if langfuse_pk and langfuse_pk.startswith("pk-lf-"):
    print("✅ LANGFUSE_PUBLIC_KEY is loaded and looks valid!")
else:
    print("❌ LANGFUSE_PUBLIC_KEY is missing or formatted incorrectly.")

if langfuse_sk and langfuse_sk.startswith("sk-lf-"):
    print("✅ LANGFUSE_SECRET_KEY is loaded and looks valid!")
else:
    print("❌ LANGFUSE_SECRET_KEY is missing or formatted incorrectly.")
