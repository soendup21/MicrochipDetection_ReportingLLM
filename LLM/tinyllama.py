import mysql.connector
import requests

url = "http://localhost:11434/api/generate"

admin_prompt = [
"""

"""
]

# Creating connection object
mydb = mysql.connector.connect(
    host = "localhost",
    user = "root",
    password = "",
    database = ""
)

def get_gemini_response(question, admin_prompt):
    try:
        model: "tinyllama" # type: ignore
        message = admin_prompt[0] + "\n" + question
        response = model.generate_content(message)
        return response.text
    except Exception as e:
        print(f"Error: {e}")
        return "Error: Unable to generate response."

def read_sql_query(query, db):
    # db.connect()
    cursor = db.cursor()
    cursor.execute(query)
    # db.commit()
    # db.close()
    results = []
    for x in cursor:
        results.append(x)
    return results

while True:
    question = input("Ask me anything: ")
    if (question == "exit"):
        break
    gemini_response = get_gemini_response(question, admin_prompt)
    if gemini_response.startswith("####"):
        query = gemini_response[4:].strip()
        print("sending query: " + query)
        query_response = read_sql_query(query, mydb)
        for row in query_response:
            print(row)

    else:
        print(gemini_response)