FROM python:3.11
MAINTAINER Chad Bowman <chad.bowman0+github@gmail.com>

COPY . /app/
WORKDIR /app
RUN pip install .

CMD ["uvicorn", "--host", "0.0.0.0", "--port", "8000", "backend.api:app"]
