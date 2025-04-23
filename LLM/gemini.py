import mysql.connector
import google.generativeai as genai

GOOGLE_API_KEY=""
genai.configure(api_key=GOOGLE_API_KEY)

admin_prompt = [
"""
"""
]

# Creating connection object
import mysql.connector

import mysql.connector

mydb = mysql.connector.connect(
    host="localhost",        # Docker exposes MySQL on localhost
    user="root",             # MySQL root user
    password="my-secret-pw",  # Your MySQL root password
    database="MYSQL_DATABASE" # The database that was created
)



def get_gemini_response(question, admin_prompt):
    try:
        model = genai.GenerativeModel('gemini-pro')  
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