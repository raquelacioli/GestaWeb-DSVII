# 1. Use uma imagem Python oficial como base
FROM python:3.10-slim

# 2. Defina o diretório de trabalho dentro do contêiner
WORKDIR /app

# 3. Copie o arquivo de dependências e instale as bibliotecas
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copie todo o código da sua aplicação para dentro do contêiner
COPY . .

# 5. Exponha a porta que o Streamlit usa (padrão é 8501)
EXPOSE 8501

# 6. Defina o comando para iniciar a aplicação quando o contêiner rodar
# As flags --server.* garantem que o app funcione corretamente dentro do Docker
CMD ["streamlit", "run", "ESUSSifilis.py", "--server.port=8501", "--server.address=0.0.0.0"]