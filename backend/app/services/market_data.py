from typing import List, Dict, Any
import pandas as pd
from app.exchange.adapters import BaseExchangeAdapter, BybitAdapter

class MarketDataService:
    """
    A service class dedicated to fetching market data from an exchange.
    """
    def __init__(self, adapter: BaseExchangeAdapter = None):
        """
        Initializes the service with a specific exchange adapter.
        Defaults to the Bybit adapter in paper trading mode.
        """
        self.adapter = adapter or BybitAdapter(paper_mode=True)

    def get_live_data(self, symbol: str) -> Dict[str, Any]:
        """
        Fetches live ticker data for a given symbol.
        This is an alias for get_instrument_details.
        """
        return self.get_instrument_details(symbol)

    def get_full_options_chain(self, underlying: str) -> pd.DataFrame:
        """
        Fetches the complete options chain for a given underlying asset.
        """
        print(f"[*] MarketDataService: Fetching options chain for {underlying}...")
        chain_df = self.adapter.get_options_chain(underlying)
        if chain_df.empty:
            print(f"[!] MarketDataService: No options chain data returned for {underlying}.")
        return chain_df

    def get_instrument_details(self, symbol: str) -> Dict[str, Any]:
        """
        Fetches detailed ticker information for a single option.
        """
        print(f"[*] MarketDataService: Fetching ticker for {symbol}...")
        ticker = self.adapter.get_ticker(symbol)
        if not ticker:
            print(f"[!] MarketDataService: No ticker data returned for {symbol}.")
        return ticker