# 🐳 Base image
FROM python:3.10-slim

# 📂 Set workdir
WORKDIR /app

# 🧱 Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 📁 Copy application files
COPY . .

# 🛠 Entrypoint setup
RUN chmod +x entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]
