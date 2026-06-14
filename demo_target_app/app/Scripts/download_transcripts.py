import sys
from pathlib import Path
from sec_edgar_downloader import Downloader

current_filepath = Path(__file__).resolve()
data_path = current_filepath.parent.parent / "data"


def fetch_sec_corpus():
    dl = Downloader('RagRadarCorp' , 'retro@ragradar.com',data_path)

    tickers = ['NVDA', 'MSFT', 'TSLA', 'GOOGL', 'AMZN', 'META', 'AAPL', 'NFLX', 'INTC', 'AMD']

    for ticker in tickers:
        try:
            dl.get("8-K", ticker, limit=5)
        except Exception as e:
            print("Error: ", str(e), "while fetching for: " , ticker)

    print("Completed fetching data for all tickers")

if __name__ == "__main__":
    fetch_sec_corpus()
