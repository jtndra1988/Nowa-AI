# ---- Dev image for Next.js 16 ----
FROM node:20-alpine

# Helpful for some native deps & Next on Alpine
RUN apk add --no-cache libc6-compat

WORKDIR /usr/src/app

# Improve DX inside containers
ENV NODE_ENV=development
ENV WATCHPACK_POLLING=true
ENV CHOKIDAR_USEPOLLING=true
ENV NEXT_TELEMETRY_DISABLED=1
ENV PORT=3000

# Install deps first for better caching
COPY package*.json ./
# If you use npm:
RUN npm ci || npm install

# Copy source (compose will also mount it as a volume for live reload)
COPY . .

# Expose dev port
EXPOSE 3000

# Ensure Next listens on all interfaces in a container
CMD ["npm", "run", "dev", "--", "-H", "0.0.0.0", "-p", "3000"]
