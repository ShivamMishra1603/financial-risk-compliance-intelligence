import os
from sec_edgar_downloader import Downloader
from tqdm import tqdm

# Configuration
# Ideally these should be in env vars, but for a resume project script hardcoding placeholders is fine or better yet, asking user.
# The user needs to provide a valid User-Agent string as per SEC requirements: "Name email@address.com"
USER_AGENT_NAME = "Shivam Mishra"
USER_AGENT_EMAIL = "shivam.mishra.1@stonybrook.edu"

# List of 50 diverse S&P 500 companies (Tech, Finance, Health, Retail, Energy)
TICKERS = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "CSCO", "CRM",
    # Finance
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "C", "AXP", "V", "MA",
    # Healthcare
    "UNH", "JNJ", "LLY", "MRK", "ABBV", "PFE", "TMO", "DHR", "BMY", "AMGN",
    # Retail/Consumer
    "WMT", "PG", "COST", "HD", "KO", "PEP", "MCD", "NKE", "SBUX", "TGT",
    # Energy/Industrial
    "XOM", "CVX", "GE", "CAT", "DE", "HON", "UNP", "UPS", "LMT", "RTX"
]

DATA_DIR = os.path.abspath("data/raw")

def download_filings():
    if not USER_AGENT_EMAIL or "email" in USER_AGENT_EMAIL:
        print("WARNING: Please set a valid User-Agent email in the script before running.")
        
    dl = Downloader(USER_AGENT_NAME, USER_AGENT_EMAIL, DATA_DIR)
    
    print(f"Downloading 10-K filings for {len(TICKERS)} companies to {DATA_DIR}...")
    
    success_count = 0
    
    for ticker in tqdm(TICKERS):
        try:
            # Download the latest 2 10-K filings (covers last ~2-3 years)
            dl.get("10-K", ticker, limit=2)
            success_count += 1
        except Exception as e:
            print(f"Failed to download {ticker}: {e}")
            
    print(f"Finished. Successfully downloaded {success_count}/{len(TICKERS)} companies.")

if __name__ == "__main__":
    download_filings()
