# import os
# import google.generativeai as genai

# GOOGLE_API_KEY="AIzaSyCHiMBVoUebpBUcPzMduXuBezBrPsbOakc"
# GOOGLE_API_KEY=(GOOGLE_API_KEY)
# genai.configure(api_key=GOOGLE_API_KEY)

# for m in genai.list_models():
#   if 'generateContent' in m.supported_generation_methods:
#     print(m.name)

# model = genai.GenerativeModel('gemini-pro')
# chat = model.start_chat(history=[])

# while True:
#     prompt = input("Ask me anything: ")
#     if (prompt == "exit"):
#         break
#     response = chat.send_message(prompt, stream=True)
#     for chunk in response:
#         if chunk.text:
#           print(chunk.text)


#           # Importing module 


### SQL connection
import mysql.connector

# Creating connection object
mydb = mysql.connector.connect(
    host = "localhost",
    user = "root",
    password = "SONYpassword",
    database = "SonyDatabaseTest"
)
cursor = mydb.cursor()

# Show database
# cursor.execute("CREATE DATABASE geeksforgeeks")

cursor.execute("SELECT DISTINCT Lot_id FROM countrecords WHERE strftime('%Y-%m', Timestamp) = '2023-10';")

for x in cursor:
  print(x)