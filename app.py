from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
import openpyxl
import os
import socket
from datetime import datetime
import re
import mysql.connector
from mysql.connector import Error
import requests
from urllib.parse import urlparse, parse_qs
import logging
from pdfminer.high_level import extract_text

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Cerebras Configuration
CEREBRAS_API_KEY = "csk-jvhewc4w4y4mjc99hkewv29ryvw9khkmr46wecdr5w5y9n39"
CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1/"

def extract_text_with_pdfminer(pdf_path):
    """
    Extracts text from a PDF file using pdfminer.
    """
    try:
        logging.info(f"Extracting text from PDF: {pdf_path}")
        
        # Check if file exists and has content
        if not os.path.exists(pdf_path):
            logging.error(f"PDF file does not exist: {pdf_path}")
            return None
            
        file_size = os.path.getsize(pdf_path)
        if file_size == 0:
            logging.error(f"PDF file is empty (0 bytes): {pdf_path}")
            return None
            
        logging.info(f"PDF file size: {file_size} bytes")
        
        # Try to extract text
        text = extract_text(pdf_path)
        
        # Remove null bytes and clean up
        text = text.replace('\x00', '').strip()
        
        if len(text) < 10:
            logging.warning(f"PDF text extraction resulted in very short text: {len(text)} characters")
            return None
            
        logging.info(f"Successfully extracted {len(text)} characters from PDF")
        return text
        
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {str(e)}")
        logging.error(f"This PDF might be corrupted, password-protected, or not a valid PDF file")
        return None

class CerebrasLLM:
    """
    Cerebras LLM client for PDF summarization
    """
    def __init__(self, api_key, system_prompt):
        self.api_key = api_key
        self.system_prompt = system_prompt
        self.base_url = CEREBRAS_BASE_URL
        self.conversation_history = []
        self.reset_conversation()

    def reset_conversation(self):
        self.conversation_history = [
            {"role": "system", "content": self.system_prompt}
        ]

    def get_headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

    def get_available_models(self):
        try:
            response = requests.get(
                f"{self.base_url}models", 
                headers=self.get_headers(),
                timeout=10
            )
            return [model["id"] for model in response.json().get("data", [])]
        except Exception as e:
            logging.error(f"Model fetch error: {str(e)}")
            return []

    def get_response(self, user_input):
        """
        Get a complete response from the LLM (non-streaming version for summarization)
        """
        self.conversation_history.append({"role": "user", "content": user_input})
        try:
            models = self.get_available_models()
            if not models:
                logging.error("No models available")
                return None
            
            # Prefer models with larger context windows for PDF summarization
            preferred_models = ['llama3.1-70b', 'llama-3.3-70b', 'llama3.1-8b']
            selected_model = None
            
            for preferred in preferred_models:
                if preferred in models:
                    selected_model = preferred
                    break
            
            # If no preferred model found, use the first available
            if not selected_model:
                selected_model = models[0]
                
            logging.info(f"Using model: {selected_model}")
            
            response = requests.post(
                f"{self.base_url}chat/completions",
                headers=self.get_headers(),
                json={
                    "model": selected_model,
                    "messages": self.conversation_history,
                    "temperature": 0.3,  # Lower temperature for more focused summaries
                    "max_tokens": 1000,  # Enough for a good summary
                    "stream": False
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                full_response = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                # Extract actual response (filter out thinking parts if any)
                actual_response = self.extract_actual_response(full_response)
                
                logging.info(f"Received response: {len(actual_response)} characters")
                self.conversation_history.append({"role": "assistant", "content": actual_response})
                return actual_response
            else:
                logging.error(f"API Error: {response.status_code} - {response.text}")
                return None
            
        except Exception as e:
            logging.error(f"LLM Error: {str(e)}")
            return None
    
    def extract_actual_response(self, full_response):
        """
        Extract only the actual response from the LLM, filtering out thinking information.
        The thinking parts always end with </think> tag.
        """
        try:
            # Check if the response contains </think>
            if '</think>' in full_response:
                # Find the position of </think> and get everything after it
                think_end_pos = full_response.find('</think>')
                actual_response = full_response[think_end_pos + 8:].strip()  # 8 is length of '</think>'
                return actual_response
            else:
                # No thinking block found, return the original response
                return full_response
        except Exception as e:
            logging.error(f"Error extracting actual response: {str(e)}")
            return full_response

def summarize_pdf_with_cerebras(pdf_text, max_words=100):
    """
    Summarizes PDF content using Cerebras LLM with prompt-based instructions.
    Works with any language input and returns English summary.
    """
    if not pdf_text or len(pdf_text.strip()) < 50:
        logging.error("PDF text is too short or empty")
        return None
    
    # Truncate PDF text to first 8000 characters to avoid context length issues
    max_chars = 8000
    if len(pdf_text) > max_chars:
        pdf_text = pdf_text[:max_chars]
        logging.info(f"PDF text truncated to {max_chars} characters")
    
    # Create system prompt for summarization
    system_prompt = """You are a helpful AI assistant specialized in document summarization.
Your task is to read documents in any language and provide clear, concise summaries in English.
Focus on capturing the main points, key ideas, and important details."""

    # Create the LLM instance
    llm = CerebrasLLM(CEREBRAS_API_KEY, system_prompt)
    
    # Create the user prompt with the PDF content (truncated if necessary)
    user_prompt = f"""Please read the following document carefully and provide a comprehensive summary in English.

The summary should:
- Be approximately {max_words} words
- Capture all the main points and key ideas
- Be written in clear, professional English
- Include important details and context

DOCUMENT:
{pdf_text}

Please provide the summary in English:"""

    logging.info("=" * 60)
    logging.info("Sending entire PDF content to Cerebras LLM for summarization...")
    logging.info(f"PDF text length: {len(pdf_text)} characters")
    logging.info("=" * 60)
    
    # Get the summary
    summary = llm.get_response(user_prompt)
    
    if summary:
        # Truncate to word count if needed
        words = summary.split()
        if len(words) > max_words:
            summary = ' '.join(words[:max_words])
        
        logging.info("✓ Successfully generated English summary!")
        return summary
    else:
        logging.error("Failed to generate summary")
        return None

def registerUser(name, affiliation, email, password, role, confirmation):
    try:
        # Connect to the MySQL database
        connection = mysql.connector.connect(
            host='localhost',         # Host where MySQL is running
            port=3307,                # MySQL port
            user='root',              # MySQL username: dhrabdco_root
            password='',              # MySQL password (set as per your configuration): dhra#2025
            database='dhra'  # Replace with your database name
        )

        if connection.is_connected():
            print("Connected to MySQL database")

            # Prepare an SQL query
            query = """INSERT INTO Client (name, affiliation, email, password, role, confirmation) 
                       VALUES (%s, %s, %s, %s, %s, %s)"""
            
            # Replace with the actual values you want to insert
            values = (name, affiliation, email, password, role, confirmation)

            # Create a cursor and execute the query
            cursor = connection.cursor()
            cursor.execute(query, values)

            # Commit the transaction
            connection.commit()
            print(f"{cursor.rowcount} record(s) inserted successfully.")
            r = 1

    except Error as e:
        print(f"Error: {e}")
        r = 0
    finally:
        # Close the database connection
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")
            return r
        
def acceptArticle(id):
    try:
        # Connect to the MySQL database
        connection = mysql.connector.connect(
            host='localhost',         # Host where MySQL is running
            port=3307,                # MySQL port
            user='root',              # MySQL username: dhrabdco_root
            password='',              # MySQL password (set as per your configuration): dhra#2025
            database='dhra'  # Replace with your database name
        )

        if connection.is_connected():
            print("Connected to MySQL database")

            # Prepare an SQL query
            query = "UPDATE Post SET confirmation = 1 where id = %s"
            
            # Replace with the actual values you want to insert
            values = (id,)

            # Create a cursor and execute the query
            cursor = connection.cursor()
            cursor.execute(query, values)

            # Commit the transaction
            connection.commit()
            print(f"{cursor.rowcount} record(s) inserted successfully.")
            r = 1

    except Error as e:
        print(f"Error: {e}")
        r = 0
    finally:
        # Close the database connection
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")
            return r
def returnArticle(id, comments):
    try:
        # Connect to the MySQL database
        connection = mysql.connector.connect(
            host='localhost',         # Host where MySQL is running
            port=3307,                # MySQL port
            user='root',              # MySQL username: dhrabdco_root
            password='',              # MySQL password (set as per your configuration): dhra#2025
            database='dhra'  # Replace with your database name
        )

        if connection.is_connected():
            print("Connected to MySQL database")

            # Prepare an SQL query
            query = "UPDATE Post SET confirmation = 2, r_comments = %s where id = %s"
            
            # Replace with the actual values you want to insert
            values = (comments, id)

            # Create a cursor and execute the query
            cursor = connection.cursor()
            cursor.execute(query, values)

            # Commit the transaction
            connection.commit()
            print(f"{cursor.rowcount} record(s) inserted successfully.")
            r = 1

    except Error as e:
        print(f"Error: {e}")
        r = 0
    finally:
        # Close the database connection
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")
            return r
        
def resubmitArticle(id, title, start_date, end_date, location, abstract, co_authors, references, dLink):
    try:
        # Connect to the MySQL database
        connection = mysql.connector.connect(
            host='localhost',         # Host where MySQL is running
            port=3307,                # MySQL port
            user='root',              # MySQL username: dhrabdco_root
            password='',              # MySQL password (set as per your configuration): dhra#2025
            database='dhra'  # Replace with your database name
        )

        if connection.is_connected():
            print("Connected to MySQL database")

            # Prepare an SQL query
            query = "UPDATE Post SET title=%s, startDate=%s, endDate=%s, location=%s, abstract=%s, coAuthors=%s, reference=%s, dLink=%s, confirmation = 0 where id = %s"
            
            # Replace with the actual values you want to insert
            values = (title, start_date, end_date, location, abstract, co_authors, references, dLink, id)

            # Create a cursor and execute the query
            cursor = connection.cursor()
            cursor.execute(query, values)

            # Commit the transaction
            connection.commit()
            print(f"{cursor.rowcount} record(s) inserted successfully.")
            r = 1

    except Error as e:
        print(f"Error: {e}")
        r = 0
    finally:
        # Close the database connection
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")
            return r





def createNewPost(email, category, typ, title, startDate, endDate, location, mLink, dLink, abstract, coAuthors, reference, confirmation):
    try:
        # Connect to the MySQL database
        connection = mysql.connector.connect(
            host='localhost',         # Host where MySQL is running
            port=3307,                # MySQL port
            user='root',              # MySQL username: dhrabdco_root
            password='',              # MySQL password (set as per your configuration): dhra#2025
            database='dhra'  # Replace with your database name
        )

        if connection.is_connected():
            print("Connected to MySQL database")

            # Prepare an SQL query
            query = """INSERT INTO Post (email, category, type, title, startDate, endDate, location, mLink, dLink, abstract, coAuthors, reference, confirmation) 
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
            
            # Replace with the actual values you want to insert
            values = (email, category, typ, title, startDate, endDate, location, mLink, dLink, abstract, coAuthors, reference, confirmation)

            # Create a cursor and execute the query
            cursor = connection.cursor()
            cursor.execute(query, values)

            # Commit the transaction
            connection.commit()
            print(f"{cursor.rowcount} record(s) inserted successfully.")
            download_pdf(dLink, email, title)
            r = 1

    except Error as e:
        print(f"Error: {e}")
        r = 0
    finally:
        # Close the database connection
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")
            return r

def clientLogin(email, password, role):
    try:
        # Connect to the MySQL database
        connection = mysql.connector.connect(
            host='localhost',         # Host where MySQL is running
            port=3307,                # MySQL port
            user='root',              # MySQL username: dhrabdco_root
            password='',              # MySQL password (set as per your configuration): dhra#2025
            database='dhra'  # Replace with your database name
        )
        r = 0
        if connection.is_connected():
            print("Connected to MySQL database")

            # Prepare an SQL query
            query = "SELECT * FROM Client WHERE email = %s AND password = %s AND role = %s"
            
            # Replace with the actual values you want to insert
            values = (email, password, role)

            # Create a cursor and execute the query
            cursor = connection.cursor()
            cursor.execute(query, values)
            account = cursor.fetchone()
            if account:
                r = 1
            else:
                r = 0
    except Error as e:
        print(f"Error: {e}")
        r = 0
    finally:
        # Close the database connection
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")
            return r

def create_user_folder(email):
    try:
        # Define the parent folder
        parent_folder = "researchers"
        
        # Construct the full path of the new folder
        new_folder_path = os.path.join(parent_folder, email)
        
        # Create the folder
        os.makedirs(new_folder_path, exist_ok=True)
        
        print(f"Folder '{email}' created successfully in '{parent_folder}'.")
    except Exception as e:
        print(f"Error creating folder: {e}")

def download_pdf(drive_link, email, title):
    try:
        # Extract file ID from the Google Drive link
        parsed_url = urlparse(drive_link)
        file_id = parse_qs(parsed_url.query).get('id', [None])[0]
        if not file_id:
            # If the link doesn't contain ?id=, check for /d/ format
            file_id = parsed_url.path.split('/')[3] if '/d/' in parsed_url.path else None

        if not file_id:
            print("Invalid Google Drive link.")
            return

        # Construct the export link for direct download
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

        # Send a GET request to download the file
        response = requests.get(download_url, stream=True)
        if response.status_code != 200:
            print("Failed to download the file. Check the Google Drive link.")
            return

        # Define the target directory
        parent_folder = f"researchers/{email}"
        os.makedirs(parent_folder, exist_ok=True)

        # Define the new file name and full path
        new_filename = f"{title}.pdf"
        target_path = os.path.join(parent_folder, new_filename)

        # Check if the file already exists and delete it if necessary
        if os.path.exists(target_path):
            print(f"File '{new_filename}' already exists. Deleting the old file.")
            os.remove(target_path)

        # Save the file to the target directory
        with open(target_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)

        print(f"File downloaded and stored as: {target_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


@app.route('/')
def index():
    #return redirect(url_for('login'))
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        role = request.form['role']
        r = clientLogin(email,password,role)
        if r:
            if role == 'reader':
                return render_template('profile.html')
            if role == 'researcher':
                return render_template('researcher.html', role=role, email=email)
            if role == 'expert':
                return render_template('expert.html', role=role, email=email)
        else:
            message = 'Invalid Credentials'
            return render_template('login.html', message=message)
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        affiliation = request.form['affiliation']
        email = request.form['email']
        password = request.form['password']
        retype_password = request.form['confirm_password']
        role = request.form['role']

        if password != retype_password:
            flash('Passwords do not match!', 'danger')
            return redirect(url_for('register'))

        confirmation = 1 if role == 'reader' else 0
        r = registerUser(name, affiliation, email, password, role, confirmation)
        if r:
            if role == 'researcher':
                create_user_folder(email)
            message = 'Registered Successfully'
            return render_template('login.html', message=message)
    return render_template('register.html')

@app.route('/accept_article', methods=['POST'])
def accept_article():
    eemail = request.form.get('eemail')
    erole = request.form.get('erole')
    id = request.form.get('mid')
    r = acceptArticle(id)
    if r:
        if erole == 'expert':
            return render_template('expert.html', role=erole, email=eemail)
        else:
            return jsonify({"message": "Accepted article but no template for this role"}), 200
    else:
        return jsonify({"message": "Failed to accept the article", "id": id}), 400

@app.route('/return_article', methods=['POST'])
def return_article():
    remail = request.form.get('remail')
    rrole = request.form.get('rrole')
    comments = request.form.get('comments')
    id = request.form.get('rid')
    r = returnArticle(id, comments)
    if r:
        if rrole == 'expert':
            return render_template('expert.html', role=rrole, email=remail)
        else:
            return jsonify({"message": "Returned article but no template for this role"}), 200
    else:
        return jsonify({"message": "Failed to return the article", "id": id}), 400

@app.route('/resubmit_article', methods=['POST'])
def resubmit_article():
    # Get form data from the request
    id = request.form.get('mid')
    title = request.form.get('mtitle')
    field_of_article = request.form.get('mfield')
    start_date = request.form.get('mstartDate')
    end_date = request.form.get('mendDate')
    location = request.form.get('mlocation')
    abstract = request.form.get('mabstract')
    co_authors = request.form.get('mcoAuthors')
    references = request.form.get('mreferences')
    dLink = request.form.get('articleLink')
    email = request.form.get('memail')
    role = request.form.get('mrole')

    r = resubmitArticle(id, title, start_date, end_date, location, abstract, co_authors, references, dLink)
    if r:
        if role == 'researcher':
            download_pdf(dLink, email, title)
            return render_template('researcher.html', role=role, email=email)
        else:
            return jsonify({"message": "Resubmitted article but no template for this role"}), 200
    else:
        return jsonify({"message": "Failed to resubmit the article", "id": id}), 400

@app.route('/createPost', methods=['GET', 'POST'])
def createPost():
    if request.method == 'POST':
        # Retrieve form data
        email = request.form.get('emailInput')
        category = request.form.get('category')
        role = request.form.get('roleInput')
        typ = request.form.get('field')
        title = request.form.get('title')
        startDate = request.form.get('start_date')
        endDate = request.form.get('end_date')
        location = request.form.get('location')
        mLink = request.form.get('multimedia_drive_link')
        dLink = request.form.get('document_drive_link')
        abstract = request.form.get('abstract')
        coAuthors = request.form.get('co_authors')
        reference = request.form.get('references')
        confirmation = 0

        r = createNewPost(email, category, typ, title, startDate, endDate, location, mLink, dLink, abstract, coAuthors, reference, confirmation)
        if r:
            return render_template('researcher.html', email=email, role=role)
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    return "Welcome to the dashboard!"  # Replace with actual dashboard implementation


@app.route('/api/researcher_data', methods=['POST'])
def researcher_data():
    data = request.json
    email = data.get('email')
    role = data.get('role')

    if not email or not role:
        return jsonify({'error': 'Missing parameters'}), 400

    try:
        # Connect to the MySQL database
        connection = mysql.connector.connect(
            host='localhost',         # Host where MySQL is running
            port=3307,                # MySQL port
            user='root',              # MySQL username: dhrabdco_root
            password='',              # MySQL password (set as per your configuration): dhra#2025
            database='dhra'  # Replace with your database name
        )
        if connection.is_connected():
            print("Connected to MySQL database")

            # Prepare an SQL query
            query = "SELECT name, affiliation, email, role FROM Client WHERE email=%s AND role=%s"
            
            # Replace with the actual values you want to insert
            values = (email, role)

            # Create a cursor and execute the query
            cursor = connection.cursor()
            cursor.execute(query, values)
            researcher = cursor.fetchone()
            if researcher:
                response = {
                    'name': researcher[0],
                    'affiliation': researcher[1],
                    'email': researcher[2],
                    'role': researcher[3],
                }
                return jsonify(response), 200
            else:
                return jsonify({'error': 'Researcher not found'}), 404
    except Error as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Close the database connection
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")

# Endpoint to fetch all posts for a given email
@app.route('/get_posts', methods=['GET'])
def get_posts():
    try:
        email = request.args.get('email')
        # Connect to the MySQL database
        connection = mysql.connector.connect(
            host='localhost',         # Host where MySQL is running
            port=3307,                # MySQL port
            user='root',              # MySQL username: dhrabdco_root
            password='',              # MySQL password (set as per your configuration): dhra#2025
            database='dhra'  # Replace with your database name
        )
        if connection.is_connected():
            print("Connected to MySQL database")

        # Prepare an SQL query
        query = "SELECT id, title, confirmation FROM Post WHERE email = %s"
            
        # Replace with the actual values you want to insert
        values = (email,)
        # Create a cursor and execute the query
        cursor = connection.cursor()
        cursor.execute(query, values)
        posts = cursor.fetchall()

        # Convert fetched data to a list of dictionaries
        posts_data = [{"id": row[0], "title": row[1], "confirmation": row[2]} for row in posts]

        cursor.close()
        connection.close()
        return jsonify(posts_data), 200
    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": "Failed to fetch posts."}), 500

@app.route('/get_posts_expert', methods=['GET'])
def get_posts_expert():
    try:
        connection = mysql.connector.connect(
            host='localhost',         # Host where MySQL is running
            port=3307,                # MySQL port
            user='root',              # MySQL username: dhrabdco_root
            password='',              # MySQL password (set as per your configuration): dhra#2025
            database='dhra'  # Replace with your database name
        )
        if connection.is_connected():
            print("Connected to MySQL database")

        # Prepare an SQL query
        query = "SELECT id, title, confirmation FROM Post WHERE confirmation = 0 or confirmation = 2"
            
        # Replace with the actual values you want to insert
        values = (0,2)
        # Create a cursor and execute the query
        cursor = connection.cursor()
        cursor.execute(query)
        posts = cursor.fetchall()

        # Convert fetched data to a list of dictionaries
        posts_data = [{"id": row[0], "title": row[1], "confirmation": row[2]} for row in posts]

        cursor.close()
        connection.close()
        return jsonify(posts_data), 200
    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": "Failed to fetch posts."}), 500
    
@app.route('/get_post_details/<int:post_id>', methods=['GET'])
def get_post_details(post_id):
    try:
        # Connect to the MySQL database
        connection = mysql.connector.connect(
            host='localhost',         # Host where MySQL is running
            port=3307,                # MySQL port
            user='root',              # MySQL username: dhrabdco_root
            password='',              # MySQL password (set as per your configuration): dhra#2025
            database='dhra'  # Replace with your database name
        )
        if connection.is_connected():
            print("Connected to MySQL database")

        # Prepare an SQL query
        query = "SELECT type, startDate, endDate, location, mLink, dLink, abstract, coAuthors, reference, id, r_comments, confirmation, title, email FROM Post WHERE id = %s"
            
        # Replace with the actual values you want to insert
        values = (post_id,)
        # Create a cursor and execute the query
        cursor = connection.cursor()
        cursor.execute(query, values)

        post_details = cursor.fetchone()
        cursor.close()
        connection.close()
        if not post_details:
            return jsonify({"error": "Post not found."}), 404
        
        # Convert fetched data to a list of dictionaries
        post_details = {
                            "type": post_details[0], 
                            "startDate": post_details[1], 
                            "endDate": post_details[2], 
                            "location": post_details[3], 
                            "dLink": post_details[5], 
                            "abstract": post_details[6], 
                            "coAuthors": post_details[7], 
                            "reference": post_details[8], 
                            "id": post_details[9],
                            "comments": post_details[10],
                            "confirmation": post_details[11],
                            "title": post_details[12],
                            "email": post_details[13]
                    }

        return jsonify(post_details), 200
    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": "Failed to fetch post details."}), 500

@app.route('/get_ai_summary/<int:post_id>', methods=['GET'])
def get_ai_summary(post_id):
    """
    Generate AI summary for a specific post's PDF
    """
    try:
        # Check if Cerebras API key is available
        if not CEREBRAS_API_KEY:
            return jsonify({"error": "AI service not configured. Please set CEREBRAS_API_KEY."}), 500
        
        # Connect to the MySQL database to get post details
        connection = mysql.connector.connect(
            host='localhost',
            port=3307,
            user='root',
            password='',
            database='dhra'
        )
        if connection.is_connected():
            logging.info(f"Connected to MySQL database for post {post_id}")

            # Get post details including email and title
            query = "SELECT email, title FROM Post WHERE id = %s"
            values = (post_id,)
            cursor = connection.cursor()
            cursor.execute(query, values)
            post = cursor.fetchone()
            cursor.close()
            connection.close()

            if not post:
                return jsonify({"error": "Post not found."}), 404
            
            email = post[0]
            title = post[1]
            
            # Construct PDF path
            pdf_path = os.path.join("researchers", email, f"{title}.pdf")
            
            logging.info(f"Looking for PDF at: {pdf_path}")
            
            # Check if PDF exists
            if not os.path.exists(pdf_path):
                return jsonify({"error": "PDF file not found."}), 404
            
            # Extract text from PDF
            pdf_text = extract_text_with_pdfminer(pdf_path)
            
            if not pdf_text:
                return jsonify({"error": "Failed to extract text from PDF. The file may be corrupted, password-protected, or image-based (scanned)."}), 500
            
            # Generate summary
            logging.info(f"Generating AI summary for post {post_id}")
            summary = summarize_pdf_with_cerebras(pdf_text, max_words=150)
            
            if summary:
                return jsonify({
                    "success": True,
                    "summary": summary,
                    "title": title
                }), 200
            else:
                return jsonify({"error": "Failed to generate AI summary."}), 500
                
    except Error as e:
        logging.error(f"Database Error: {str(e)}")
        return jsonify({"error": f"Database error: {str(e)}"}), 500
    except Exception as e:
        logging.error(f"Error generating AI summary: {str(e)}")
        return jsonify({"error": f"Failed to generate AI summary: {str(e)}"}), 500

@app.route('/adminDashboard', methods=['GET', 'POST'])
def admin_dashboard_login():
    if request.method == 'POST':
        admin_id = request.form.get('admin_id')
        password = request.form.get('password')
        
        # Fixed credentials for admin
        if admin_id == 'Admin' and password == 'incorrect':
            return redirect(url_for('admin_dashboard'))
        else:
            message = 'Invalid Credentials'
            return render_template('admin_login.html', message=message)
    
    return render_template('admin_login.html')

@app.route('/adminDashboard/home')
def admin_dashboard():
    # Fetch users with confirmation=0
    try:
        connection = mysql.connector.connect(
            host='localhost',         # Host where MySQL is running
            port=3307,                # MySQL port
            user='root',              # MySQL username: dhrabdco_root
            password='',              # MySQL password (set as per your configuration): dhra#2025
            database='dhra'  # Replace with your database name
        )
        if connection.is_connected():
            cursor = connection.cursor()
            query = "SELECT name, email, role FROM Client WHERE confirmation = 0"
            cursor.execute(query)
            users = cursor.fetchall()
            cursor.close()
            connection.close()
            users_data = [{"name": row[0], "email": row[1], "role": row[2]} for row in users]

        connection = mysql.connector.connect(
            host='localhost',         # Host where MySQL is running
            port=3306,                # MySQL port
            user='dhrabdco_root',              # MySQL username
            password='dhra#2025',              # MySQL password (set as per your configuration)
            database='dhrabdco_dhra'  # Replace with your database name
        )
        if connection.is_connected():
            cursor = connection.cursor()
            query = "SELECT id, email,type, title from Post WHERE confirmation = 1"
            cursor.execute(query)
            posts = cursor.fetchall()
            cursor.close()
            connection.close()

            posts_data = [{"id": row[0], "email": row[1], "type": row[2], "title": row[3]} for row in posts]
            
            return render_template('dashboard.html', users=users_data, posts = posts_data)
    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": "Failed to fetch users."}), 500

@app.route('/adminDashboard/approve_user', methods=['POST'])
def approve_user():
    user_email = request.form.get('user_email')
    
    try:
        connection = mysql.connector.connect(
            host='localhost',         # Host where MySQL is running
            port=3307,                # MySQL port
            user='root',              # MySQL username: dhrabdco_root
            password='',              # MySQL password (set as per your configuration): dhra#2025
            database='dhra'  # Replace with your database name
        )
        if connection.is_connected():
            cursor = connection.cursor()
            query = "UPDATE Client SET confirmation = 1 WHERE email = %s"
            cursor.execute(query, (user_email,))
            connection.commit()
            cursor.close()
            connection.close()
            
            return jsonify({"message": "User approved successfully"}), 200
    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": "Failed to approve user."}), 500
    



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
