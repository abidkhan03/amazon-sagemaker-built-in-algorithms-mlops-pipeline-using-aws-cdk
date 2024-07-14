FROM public.ecr.aws/lambda/python:3.11

WORKDIR ${LAMBDA_TASK_ROOT}
# Copy requirements.txt
COPY requirements.txt .

# Install the specified packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy function code
COPY handler.py .

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "handler.handle" ]