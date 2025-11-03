# Use a Node.js base image, which includes all the tools we need for development
FROM node:18-alpine

# Set the working directory inside the container
WORKDIR /app

# Copy the "shopping list" for dependencies
# Using package*.json will copy both package.json and package-lock.json if it exists
COPY package*.json ./

# Install all the necessary libraries
# This is the standard command for development.
RUN npm install

# Copy the rest of the frontend source code into the container
COPY . .

# Expose the port for the Vite development server (usually 5173 or 3000)
# We will use 3000 to match our docker-compose.yml file.
EXPOSE 3000

# The definitive command to start the Vite development server.
# The '--host' flag is crucial to make it accessible from outside the Docker container.
CMD ["npm", "run", "dev", "--", "--host"]

# docker-compose down --remove-orphans && docker-compose up --build
    

