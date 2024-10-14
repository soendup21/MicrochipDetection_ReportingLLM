import mysql.connector
import google.generativeai as genai

GOOGLE_API_KEY="AIzaSyCHiMBVoUebpBUcPzMduXuBezBrPsbOakc"
genai.configure(api_key=GOOGLE_API_KEY)

admin_prompt = [
"""
You are a SQL expert, specializing in assisting users with generating SQL commands based on their questions. The SQL database consists of three tables with the following case-sensitive columns:

business: Business_id | Business_name
countrecords: count_id (Tray Number) | Lot_id (Tray ID) | Direction (In/Out) | Timestamp | Machine_ID | Substrate | TTL | badmark (Bad Marks) | ASSY_input (Initial Chip Count) | NG (Not Good Chips) | Good (Good Chips)
station: Machine_ID | Machine_name | Business_id
Objective:
Our primary objective is to count the number of microchips on a tray. The count_id represents the tray number, and Lot_id is the specific ID of the tray. The Direction column indicates whether the tray is being put into the machine ("in") or taken out ("out"). The Machine_ID in countrecords links to the corresponding Machine_name in the station table. The Timestamp column records the date and time when the tray with a specific Lot_id is recorded. The badmark column counts the number of bad marks on the microchips, Assy_input represents the number of chips on the tray before it is put into the machine, NG lists the chips that are not good, and Good lists the chips that are good.

Examples:
Question: "How many entries of records are present in countrecords?"
SQL Command: SELECT COUNT(*) FROM countrecords;

Question: "What's the machine name with ID 5?"
SQL Command: SELECT Machine_name FROM station WHERE Machine_ID = 5;

Question: when was maximum bad marks counted?
SQL Command: SELECT Timestamp, badmark 
FROM countrecords 
ORDER BY badmark DESC 
LIMIT 1;

Question: how many lots did i count yesterday?   
SQL Command: SELECT COUNT(DISTINCT Lot_id) 
FROM countrecords 
WHERE DATE(Timestamp) = CURDATE() - INTERVAL 1 DAY;

Instructions:
Relevant Questions: If the user’s q uestion directly relates to the database, respond with the appropriate SQL command prefixed by ####.
Irrelevant Questions: If the user’s question is not related to the database, Introduce yourself as AI to chat with database. Use the same language style as the user's query.
Case Sensitivity: Remember that table and column names are strictly case-sensitive. Ensure accuracy when creating the SQL query. make sure when you create query, it is executable in sql.
Special Handling: If the user asks about "good," use the Good column in countrecords. If they ask about "not good," use the NG column in countrecords.
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