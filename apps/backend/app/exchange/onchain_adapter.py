# In app/exchange/onchain_adapter.py

import requests

class OnChainAdapter:
    """Adapter for fetching data from various on-chain metric providers."""

    def get_whale_alert_data(self, symbol: str = "btc", api_key: str = "demo"):
        """Fetches whale transaction data from Whale Alert API."""
        print(f"[*] OnChain Adapter: Fetching Whale Alert data for {symbol}...")
        try:
            # Note: The free/demo key is very limited.
            r = requests.get(
                "https://api.whale-alert.io/v1/transactions",
                params={"api_key": api_key, "symbol": symbol, "min_value": 500000}
            )
            r.raise_for_status()
            transactions = r.json().get("transactions", [])
            count = len(transactions)
            volume = sum(tx.get("amount_usd", 0) for tx in transactions)
            print(f"[+] OnChain Adapter: Found {count} whale transactions totaling ${volume:,.2f} USD.")
            return count, volume
        except requests.RequestException as e:
            print(f"[!] OnChain Adapter: Whale Alert API error: {e}")
            return 0, 0.0

    def get_coinmetrics_data(self, symbol: str = "btc"):
        """Fetches Exchange Net Flow, Active Addresses, and Tx Count from CoinMetrics."""
        print(f"[*] OnChain Adapter: Fetching CoinMetrics data for {symbol}...")
        try:
            url = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
            params = {
                "assets": symbol,
                "metrics": "ExchangeNetFlowUSD,AdrActCnt,TxTfrCnt",
                "frequency": "1d",
                "page_size": 1 # Only need the latest
            }
            r = requests.get(url, params=params)
            r.raise_for_status()
            latest_data = r.json()['data'][-1]

            net_flow = float(latest_data.get('ExchangeNetFlowUSD', 0.0))
            active_addr = int(latest_data.get('AdrActCnt', 0))
            tx_count = int(latest_data.get('TxTfrCnt', 0))
            print(f"[+] OnChain Adapter: Fetched CoinMetrics data for {symbol}.")
            return net_flow, active_addr, tx_count
        except requests.RequestException as e:
            print(f"[!] OnChain Adapter: CoinMetrics API error: {e}")
            return 0.0, 0, 0