import requests
import json
import mysql.connector
from mysql.connector import Error
import re
import time
from decimal import Decimal
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False
    
DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "sony",       
    "password": "sony",   
    "database": "sony"    
}

OLLAMA_BASE_URL = "http://localhost:11434/api/chat"
SQL_MODEL_NAME = "llama3.1:latest"
ANALYSIS_MODEL_NAME = SQL_MODEL_NAME

MAX_RESULTS_TO_ANALYZE = 20
MAX_HISTORY_TURNS = 6

TABLE_NAME = "object_counts"
TABLE_CONTEXT = f"""
**Table Purpose:**
This table, named '{TABLE_NAME}', logs inspection data for microchip trays. It records counts of specific types of defects ('bad marks') found on each tray and includes identifiers for the tray, batch, and operator.

**Table Schema and Column Descriptions:**

"""

def connect_to_db(config):
    try:
        connection = mysql.connector.connect(**config)
        if connection.is_connected():
            return connection
    except Error as e:
        print(f"Error connecting to MySQL database: {e}")
        return None

def check_and_reconnect_db(connection, config):
    retries = 1
    for attempt in range(retries + 1):
        try:
            if connection is None or not connection.is_connected() or not connection.ping(reconnect=True, attempts=1, delay=1):
                print("Database connection issue. Attempting to reconnect...")
                if connection:
                    try:
                        connection.close()
                    except Error:
                        pass
                connection = connect_to_db(config)
                if connection:
                    print("Successfully reconnected to MySQL database.")
                    return connection
                else:
                    print(f"Reconnect attempt {attempt + 1} failed.")
                    if attempt < retries:
                        time.sleep(2)
                    else:
                        print("Giving up on reconnection.")
                        return None
            else:
                return connection
        except Error as e:
            print(f"Error during connection check/ping: {e}")
            connection = None
            if attempt >= retries:
                print("Giving up on reconnection after ping error.")
                return None
    return connection

def execute_sql_query(connection, query):
    if not connection:
         print("Cannot execute query: No database connection.")
         return {"error": "No database connection."}
    normalized_query = query.strip().upper()
    blocked_commands = ["DROP ", "ALTER ", "TRUNCATE ", "DELETE FROM "]
    if any(normalized_query.startswith(cmd) for cmd in blocked_commands):
        print(f"\n--- Blocked Potentially Destructive Query ---")
        print(f"Query: {query}")
        print("Reason: Command blocked for safety.")
        print("----------------------------------------------\n")
        return {"error": f"Query blocked for safety: {query}"}
    results = None
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute(query)
        if cursor.description is None:
            connection.commit()
            rows_affected = cursor.rowcount
            cursor.close()
            if normalized_query.startswith("UPDATE ") or normalized_query.startswith("INSERT "):
                 return {"status": "success", "rows_affected": rows_affected}
            else:
                 return {"status": "executed (non-select)", "rows_affected": rows_affected}
        else:
            results = cursor.fetchall()
            cursor.close()
            processed_results = []
            for row in results:
                processed_row = {}
                for key, value in row.items():
                    if isinstance(value, Decimal):
                        processed_row[key] = float(value)
                    elif isinstance(value, (bytes, bytearray)):
                        try:
                            processed_row[key] = value.decode('utf-8')
                        except UnicodeDecodeError:
                            processed_row[key] = repr(value)
                    elif hasattr(value, 'isoformat'):
                        processed_row[key] = value.isoformat()
                    else:
                        processed_row[key] = value
                processed_results.append(processed_row)
            return {"data": processed_results}
    except Error as e:
        error_message = f"MySQL Error: {e}"
        print(f"\n--- Error executing SQL ---")
        print(f"Query: {query}")
        print(f"Error: {e}")
        print("---------------------------\n")
        try:
            connection.rollback()
        except Error as rb_e:
            print(f"Rollback failed: {rb_e}")
        return {"error": error_message}
    except Exception as ex:
        error_message = f"Python Error during query execution/processing: {ex}"
        print(f"\n--- Python Error During Query Handling ---")
        print(f"Query: {query}")
        print(f"Error: {ex}")
        print("-----------------------------------------\n")
        return {"error": error_message}

def ollama_send_message(messages, model_name):
    payload = {
        "model": model_name,
        "messages": messages,
        "options": {
            "temperature": 0.1 
        },
        "stream": True
    }
    full_response_content = ""
    try:
        response = requests.post(OLLAMA_BASE_URL, json=payload, stream=True, timeout=90)
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        full_response_content += data["message"]["content"]
                    if data.get("done", False): break
                except json.JSONDecodeError:
                    print(f"\nWarning: Failed to parse JSON line: {line}")
                except Exception as e:
                    print(f"\nError processing stream line: {e} - Line: {line}")
    except requests.exceptions.Timeout:
         print(f"\n--- Request Error ---")
         print(f"Error: Request to {model_name} timed out.")
         return None
    except requests.exceptions.RequestException as e:
        print(f"\n--- Request Error --- Error sending request to {model_name}: {e}")
        return None
    except Exception as e:
        print(f"\n--- Unexpected Error during API call to {model_name} --- Error: {e}")
        return None
    return full_response_content.strip()

def extract_sql_from_response(response_text):
    if not response_text:
        return None
    refusal_messages = [
        "SELECT 'Ambiguous request. Please provide more specific details.';",
        "SELECT 'Please ask a question about the object_counts table.';",
        "SELECT 'Schema modification commands like ALTER or DROP are not allowed.';"
    ]
    if response_text.strip() in refusal_messages:
        return response_text.strip()
    code_block_match = re.search(r"```sql\s*(.*?)\s*```", response_text, re.DOTALL | re.IGNORECASE)
    if code_block_match:
        sql = code_block_match.group(1).strip()
        sql = re.sub(r"--.*$", "", sql, flags=re.MULTILINE).strip()
        sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL).strip()
        if sql:
            if not sql.endswith(';'): sql += ';'
            sql = re.sub(r'"(\w+)"', r'`\1`', sql)
            if re.match(r"^(SELECT|INSERT|UPDATE|DELETE|WITH)\s+", sql, re.IGNORECASE):
                 return sql
    direct_sql_match = re.match(r"^(SELECT|INSERT|UPDATE|DELETE|WITH)\s+.*?;$", response_text.strip(), re.IGNORECASE | re.DOTALL)
    if direct_sql_match:
        sql = response_text.strip()
        sql = re.sub(r'"(\w+)"', r'`\1`', sql)
        sql = re.sub(r"--.*$", "", sql, flags=re.MULTILINE).strip()
        sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL).strip()
        if sql:
             return sql
    print(f"\n--- Could not extract SQL --- Expected SQL or specific SELECT '...' message.")
    print(f"Raw Response:\n{response_text}")
    print(f"----------------------------\n")
    return None

def analyze_results_with_llm(original_question, sql_query, sql_result):
    analysis_prompt_messages = []
    result_data = sql_result.get("data", None)
    result_error = sql_result.get("error", None)
    result_status = sql_result.get("status", None)
    rows_affected = sql_result.get("rows_affected", None)
    system_message = (
        f"You are a data analysis assistant. Your task is to interpret the results of a SQL query "
        f"in the context of the user's original question and the provided table schema. Provide a concise, natural language answer "
        f"based only on the query results and the table context. Interpret the results briefly and directly.\n\n"
        f"**Table Context for Interpretation:**\n{TABLE_CONTEXT}"
    )
    analysis_prompt_messages.append({"role": "system", "content": system_message})
    user_prompt_content = f"Original question: \"{original_question}\"\n\n"
    user_prompt_content += f"Executed SQL query: `{sql_query}`\n\n"
    if result_error:
        user_prompt_content += f"The query resulted in an error: {result_error}\n\n"
        user_prompt_content += f"Explain this error to the user in simple terms."
    elif result_status == "success":
        command_match = re.match(r"^\s*(UPDATE|INSERT)\s+", sql_query, re.IGNORECASE)
        command_type = command_match.group(1).upper() if command_match else "Operation"
        user_prompt_content += f"The {command_type} query executed successfully."
        if rows_affected is not None:
             user_prompt_content += f" {rows_affected} row(s) were affected.\n\n"
        else:
             user_prompt_content += "\n\n"
        user_prompt_content += f"Briefly inform the user about the successful {command_type} operation on the '{TABLE_NAME}' table."
    elif result_data is not None:
        if not result_data:
            user_prompt_content += f"Query Results: The query ran successfully but returned no data.\n\n"
            user_prompt_content += f"Explain that no records matching the criteria were found."
        else:
            limited_data = result_data[:MAX_RESULTS_TO_ANALYZE]
            if HAS_TABULATE and len(limited_data) > 0:
                 try:
                     headers = limited_data[0].keys()
                     table_str = tabulate(limited_data, headers="keys", tablefmt="grid")
                     user_prompt_content += f"Query Results (first {len(limited_data)} rows):\n```\n{table_str}\n```\n\n"
                 except Exception:
                     data_as_json = json.dumps(limited_data, indent=2)
                     user_prompt_content += f"Query Results (first {len(limited_data)} rows):\n```json\n{data_as_json}\n```\n\n"
            else:
                 data_as_json = json.dumps(limited_data, indent=2)
                 user_prompt_content += f"Query Results (first {len(limited_data)} rows):\n```json\n{data_as_json}\n```\n\n"
            if len(result_data) > MAX_RESULTS_TO_ANALYZE:
                 user_prompt_content += f"(Note: Query returned {len(result_data)} total rows, but only showing the first {MAX_RESULTS_TO_ANALYZE} for analysis.)\n\n"
            user_prompt_content += f"Based only on the provided query results, the original question, and the table context, answer the user's question in a natural and concise way. Do not just repeat the raw data unless asked for; instead, summarize the key points."
    else:
         user_prompt_content += "Query Results: The execution status is unclear.\n\n"
         user_prompt_content += f"Inform the user that the query was executed against the '{TABLE_NAME}' table but the outcome details are missing."
    analysis_prompt_messages.append({"role": "user", "content": user_prompt_content})
    print(f"🤖 Analyzing results (using {ANALYSIS_MODEL_NAME})...", end="\r")
    analysis_response = ollama_send_message(analysis_prompt_messages, ANALYSIS_MODEL_NAME)
    print("                                                     ", end="\r")
    if analysis_response:
        return analysis_response
    else:
        return "[Analysis failed: Could not get response from the analysis model.]"

def interactive_chatbot():
    print("Interactive SQL Chatbot")
    print(f"SQL Model: {SQL_MODEL_NAME}, Analysis Model: {ANALYSIS_MODEL_NAME}")
    print(f"Table context: {TABLE_NAME}")
    print("Type 'exit' to quit.")
    sql_system_prompt = f"""
You are a specialized Text-to-SQL assistant for MySQL. Your ONLY task is to convert natural language questions into VALID and EXECUTABLE **MySQL** SQL queries based on the provided table context.

VERY STRICT INSTRUCTIONS:
1.  OUTPUT ONLY THE SQL QUERY. Generate absolutely nothing else.
2.  ENSURE MYSQL SYNTAX. Use backticks (`) for identifiers if needed.
3.  TERMINATE WITH SEMICOLON.
4.  HANDLE AMBIGUITY.
5.  IGNORE NON-SQL REQUESTS.
6.  NO SCHEMA MODIFICATION.
7.  USE PROVIDED CONTEXT ONLY.
8.  PAY ATTENTION TO FORMATS.
**Table Context:**
{TABLE_CONTEXT}
"""
    sql_conversation_history = [{"role": "system", "content": sql_system_prompt}]
    connection = None
    while True:
        if connection is None:
             connection = connect_to_db(DB_CONFIG)
             if not connection:
                  user_input = input("Fatal: DB connection failed. Press Enter to retry, or type 'exit': ")
                  if user_input.lower() == 'exit': break
                  else: continue
        connection = check_and_reconnect_db(connection, DB_CONFIG)
        if not connection:
             user_input = input("DB connection lost. Press Enter to retry, or type 'exit': ")
             if user_input.lower() == 'exit': break
             else: continue
        try:
            user_input = input("You: ")
        except EOFError:
             print("\nInput stream ended. Goodbye!")
             break
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        if not user_input.strip(): continue
        sql_conversation_history.append({"role": "user", "content": user_input})
        print(f"🤖 Generating SQL (using {SQL_MODEL_NAME})...", end="\r")
        raw_sql_response = ollama_send_message(sql_conversation_history, SQL_MODEL_NAME)
        print("                                                ", end="\r")
        if raw_sql_response is None:
            print("Assistant: Failed to get response from the SQL generation model.")
            sql_conversation_history.pop()
            continue
        # Print raw LLM response to ensure visibility of generated SQL attempt
        print(f"Raw SQL Response: {raw_sql_response}")
        extracted_sql = extract_sql_from_response(raw_sql_response)
        if not extracted_sql:
            print("Assistant: Sorry, I am an AI assistant to help you chat with database only. ")
            continue
        if extracted_sql.startswith("SELECT '"):
             message = extracted_sql[8:-2]
             print(f"Assistant: {message}")
             sql_conversation_history.append({"role": "assistant", "content": extracted_sql})
             continue 
        print(f"Generated SQL: {extracted_sql}")
        sql_conversation_history.append({"role": "assistant", "content": extracted_sql})
        print("Executing query...", end="\r")
        sql_result = execute_sql_query(connection, extracted_sql)
        print("                  ", end="\r")
        print(f"Database Output: {json.dumps(sql_result, indent=2, default=str)}")
        if sql_result: 
            final_answer = analyze_results_with_llm(user_input, extracted_sql, sql_result)
            print(f"Assistant: {final_answer}")
        else:
            print("Assistant: There was an unexpected issue executing the query or processing its results.")
        if len(sql_conversation_history) > (1 + 2 * MAX_HISTORY_TURNS):
             keep_from_index = len(sql_conversation_history) - (2 * MAX_HISTORY_TURNS)
             sql_conversation_history = [sql_conversation_history[0]] + sql_conversation_history[keep_from_index:]
    if connection and connection.is_connected():
        try: 
            connection.close()
            print("Database connection closed.")
        except Error as e: 
            print(f"Error closing connection: {e}")

if __name__ == "__main__":
    interactive_chatbot()