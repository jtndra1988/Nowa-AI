from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseExchangeAdapter(ABC):
    """
    Abstract Base Class for all exchange adapters.
    It defines the standard interface (the "contract") that every
    exchange-specific implementation must adhere to.
    """

    @abstractmethod
    def get_options_chain(self, underlying_symbol: str) -> List[Dict[str, Any]]:
        """
        Should fetch the entire options chain for a given underlying currency.
        
        Args:
            underlying_symbol (str): The symbol of the underlying asset (e.g., 'BTC').

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                                  represents a single option instrument in the chain.
        """
        pass

    @abstractmethod
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Should fetch the latest ticker information for a specific instrument.
        This includes mark price, greeks, volume, open interest, etc.

        Args:
            symbol (str): The full symbol of the instrument 
                          (e.g., 'BTC-28MAR25-80000-C').

        Returns:
            Dict[str, Any]: A dictionary containing the ticker data.
        """
        pass

    # You can add more abstract methods here as the application grows,
    # for example, for placing orders, checking positions, etc.

    # @abstractmethod
    # def create_order(self, symbol: str, side: str, amount: float, price: float = None, order_type: str = 'market'):
    #     pass

    # @abstractmethod
    # def get_open_positions(self) -> List[Dict[str, Any]]:
    #     pass
