FROM python:3.9-slim

WORKDIR /app

COPY . .

RUN pip install -r requirements.txt

EXPOSE 8000

# Either of these commands should work:
CMD ["python", "app.py"]
# Or alternative: CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]