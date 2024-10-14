import openai
import pandas as pd

# Replace with your OpenAI API key
OPENAI_API_KEY = "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"
openai.api_key = OPENAI_API_KEY

# Load CSV data from the specified file path
def load_csv_data():
    csv_file_path = r'F:\Project\End-End T2SQL\CountRecords.csv'
    try:
        count_records = pd.read_csv(csv_file_path)
        return count_records
    except FileNotFoundError:
        return "CSV file not found."

# The prompt to guide OpenAI for generating proper responses based on CSV data
admin_prompt = """
You are an expert in detecting bad marks on microchips and providing insights based on the following database. The database consists of several columns from the CSV file that tracks microchips being processed through a machine:

Columns:
1. Count ID: Tray number used for counting.
2. LOT ID: A unique identifier for each tray.
3. In/Out: Indicates whether the tray is entering ('in') or exiting ('out') the machine.
4. Timestamp: The time and date the tray was recorded.
5. Machine ID: The ID of the machine processing the tray.
6. CDB: Bad marks on the chip that look like a star.
7. BMS: Bad marks on the chip that look like "//".
8. Maker: Bad marks on the chip that look like a dot.
9. Badmarks total: The total number of bad marks on the chip.
10. Good chips: The number of good chips on the tray.

Objective:
Our primary objective is to count the number of good chips, analyze the bad marks from various sources (CDB, BMS, Maker), and report on the overall quality of the chips. The data helps us improve manufacturing quality by tracking defects and analyzing trends based on machine and time.

Examples:
- Question: "How many good chips are there on tray 1?"
  Response: Look at the Good chips column for tray number 1 in the CSV.

- Question: "What is the most common type of bad mark in the last recorded batch?"
  Response: Analyze the CDB, BMS, and Maker columns to see which has the highest value.

Instructions:
- Case Sensitivity: Column names are case-sensitive, so make sure to use the exact column names when generating queries.
- If the user asks about 'bad marks', reference the CDB, BMS, and Maker columns.
- If the user asks about 'good chips', reference the Good chips column.
- If they ask about total bad marks, reference the 'Badmarks total' column.
"""

# Function to interact with OpenAI and generate a response
def get_openai_response(question, admin_prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # You can change to a different model if needed
            messages=[
                {"role": "system", "content": admin_prompt},
                {"role": "user", "content": question}
            ],
            max_tokens=150,
            temperature=0.5,
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Error: {e}")
        return "Error: Unable to generate response."

# Function to simulate query on CSV data
def read_csv_query(query, data):
    # For demonstration, handle simple query parsing from AI response
    if "COUNT(*) FROM countrecords" in query:
        return len(data)
    if "Good chips" in query and "tray" in query:
        # Extract the tray number from the query
        tray_number = int(query.split("tray")[1].strip().split()[0])
        return data[data['Count ID'] == tray_number]['Good chips'].values[0]
    if "badmark" in query and "ORDER BY badmark DESC" in query:
        return data[['Timestamp', 'Badmarks total']].sort_values(by='Badmarks total', ascending=False).head(1)

    # Default response if query parsing doesn't match
    return "Query parsing not implemented yet."

# Main loop to handle user questions
while True:
    question = input("Ask me anything: ")
    if question.lower() == "exit":
        break

    # Send the question to OpenAI model
    openai_response = get_openai_response(question, admin_prompt)
    
    if openai_response.startswith("####"):
        query = openai_response[4:].strip()
        print("Generated query: " + query)

        # Load CSV data
        data = load_csv_data()
        if isinstance(data, str):
            print(data)  # CSV file not found error
        else:
            # Execute the simulated SQL query on the CSV data
            query_response = read_csv_query(query, data)
            print(query_response)

    else:
        print(openai_response)
