from huggingface_hub import login

print("Starting Hugging Face Login...")
print("Please paste your token when prompted (it will not show on screen).")
try:
    login()
    print("✅ Login successful!")
except Exception as e:
    print(f"❌ Login failed: {e}")
