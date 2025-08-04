FROM python:3.12.3-slim

WORKDIR /app

# copy dependency files
COPY Pipfile Pipfile.lock ./

# install pipenv and dependencies
RUN pip install pipenv && pipenv install --system --deploy

# copy the serving script
COPY scripts/serve_model_simple.py ./serve.py

EXPOSE 5000

# run with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "serve:app"]