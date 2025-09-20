# Use a lightweight Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy requirements.txt into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy scripts and data files into the container
COPY pretty_print_csv.py .
COPY test_serper.py .
COPY test_simple_serper.py .
COPY test_one_record.py .
COPY test_100_records.py .
COPY fix_simple_qa_urls.py .
COPY simple_qa_test_set.csv .

# Set the default command to start a Bash shell
CMD ["bash"]

