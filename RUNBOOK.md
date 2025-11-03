AI Crypto Options - Runbook
This document provides step-by-step instructions for setting up and running the application, including how to get API keys for paper trading on a reliable platform.

1. Getting Bybit Testnet API Keys (for Paper Trading)
This application is configured to use the Bybit Testnet by default. You will trade with fake money, which is the only safe way to test a trading bot. The Bybit Testnet is known for being stable and developer-friendly.

Create a Bybit Testnet Account:

Go to the official Bybit Testnet website: https://testnet.bybit.com/

Sign up for a new account. This is completely separate from a real Bybit account.

Navigate to API Management:

Once logged in, click your profile icon in the top-right corner and select "API".

Create a New Key:

Click the "Create New Key" button.

Select "System-generated API Keys".

Give your key a name, for example: ai-options-bot.

For permissions, select "Read-Write".

Ensure the box for "Unified Trading" is checked. This is required for the latest APIs to access options and futures.

You do not need to set an IP restriction for the testnet, making the setup simpler.

Click "Submit".

Copy Your Keys:

Bybit will show you your API Key and API Secret.

IMPORTANT: Copy both immediately and save them somewhere safe. The Secret Key will only be shown once.

Update Your .env file:

Create a .env file in the root of the project (by copying .env.example).

Copy these new keys into your .env file for the BYBIT_TESTNET_API_KEY and BYBIT_TESTNET_API_SECRET variables.

2. Running the Application
Follow the setup instructions in README.md to run the application using Docker. The application will automatically use your new Bybit testnet keys from the .env file.

The primary command to launch the entire stack is:

make up

Or, if you don't have make installed:

docker-compose up --build

3. Running the AI Model Training
From the root ai-crypto-options directory, you can run the training script inside the running backend container. Open a new terminal and run:

docker-compose exec backend poetry run python app/ml/train.py
