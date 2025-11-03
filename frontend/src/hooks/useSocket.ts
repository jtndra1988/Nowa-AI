import { useState, useEffect, useRef } from 'react';

const WEBSOCKET_URL = "ws://localhost:8000/api/v1/ws"; // WebSocket URL of our FastAPI backend

export const useSocket = () => {
  const [messages, setMessages] = useState<any[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    // Function to establish connection
    const connect = () => {
      console.log("Attempting to connect to WebSocket...");
      const socket = new WebSocket(WEBSOCKET_URL);

      socket.onopen = () => {
        console.log("WebSocket connection established.");
        setIsConnected(true);
      };

      socket.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          console.log("Received message:", message);
          setMessages((prevMessages: any[]) => [...prevMessages, message]);
        } catch (error) {
          console.error("Error parsing message:", error);
        }
      };

      socket.onclose = () => {
        console.log("WebSocket connection closed. Reconnecting...");
        setIsConnected(false);
        // Implement a retry mechanism
        setTimeout(() => {
          connect();
        }, 5000); // Attempt to reconnect every 5 seconds
      };

      socket.onerror = (error) => {
        console.error("WebSocket error:", error);
        socket.close(); // This will trigger the onclose event and reconnection logic
      };

      ws.current = socket;
    };

    connect();

    // Cleanup function to close the connection when the component unmounts
    return () => {
      if (ws.current) {
        ws.current.close();
      }
    };
  }, []); // Empty dependency array ensures this runs only once on mount

  // Function to send messages (optional, but useful for future features)
  const sendMessage = (message: any) => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(message));
    } else {
      console.error("WebSocket is not connected.");
    }
  };

  return { messages, isConnected, sendMessage };
};
