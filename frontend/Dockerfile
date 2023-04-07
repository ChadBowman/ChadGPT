FROM node:16-alpine

COPY . /app/
WORKDIR /app

RUN npm install && npm run build && npm install -g serve

CMD ["serve", "-s", "build"]
