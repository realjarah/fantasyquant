FROM python:3.11-slim

WORKDIR /app

# Install the package.
COPY . .
RUN pip install --no-cache-dir .

# Cloud platforms set PORT; default to 8000.
ENV PORT=8000

EXPOSE ${PORT}

CMD fantasyquant serve --port ${PORT}
