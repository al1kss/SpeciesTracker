import datetime
import os
import secrets  # https://docs.python.org/3/library/secrets.html
from random import random

from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import sqlite3
import bcrypt
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS


class DatabaseManager:
    def __init__(self, name: str):
        self.connection = sqlite3.connect(name)
        self.cursor = self.connection.cursor()

    def search(self, query, values):
        result = self.cursor.execute(query, values).fetchall()
        return result

    def close(self):
        self.connection.close()

    def run_save(self, query, values):
        self.cursor.execute(query, values)
        self.connection.commit()

    def create_db(self):
        query = '''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'student'
        )
        '''
        self.cursor.execute(query)

        query_images = '''
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL,
                    latitude REAL,
                    longitude REAL,
                    username TEXT,
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    species_name TEXT,
                    species_type TEXT,
                    icon_class TEXT
                    --FOREIGN KEY(username) REFERENCES users(username) ON DELETE CASCADE if I want to link images to users, delete cascade means if user is deleted, their images are also deleted
                )
                '''
        self.cursor.execute(query_images)
        self.connection.commit()

    def add_user(self, username, email, password, role='student'):
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        query = 'INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)'
        self.run_save(query, (username, email, hashed, role))

    def check_user_exists(self, username, email):
        query = 'SELECT username, email FROM users WHERE username = ? OR email = ?'
        result = self.search(query, (username, email))
        return result

    def add_image(self, file_path, latitude, longitude, username=None, species_name=None, species_type=None,
                  icon_class=None):
        query = 'INSERT INTO images (file_path, latitude, longitude, username, species_name, species_type, icon_class) VALUES (?,?,?,?,?,?,?)'
        self.run_save(query, (file_path, latitude, longitude, username, species_name, species_type, icon_class))


app = Flask(__name__, template_folder='templates')
secret = secrets.token_hex(16)  # Generate a random secret key
print(secret)
app.secret_key = str(secret)  # Replace with a secure, random string
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'HEIF'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def get_exif_location(image_path):
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()
        if not exif_data:
            return None, None

        gps_info = None
        for tag, value in exif_data.items():
            if TAGS.get(tag) == 'GPSInfo':
                gps_info = value
                break

        if not gps_info:
            return None, None

        gps_data = {}
        for tag, value in gps_info.items():
            gps_tag = GPSTAGS.get(tag, tag)
            gps_data[gps_tag] = value

        def convert_to_degrees(value):
            d, m, s = value
            return d + (m / 60.0) + (s / 3600.0)

        latitude = None
        longitude = None  # give citation to the website
        if 'GPSLatitude' in gps_data and 'GPSLongitude' in gps_data:
            lat = convert_to_degrees(gps_data['GPSLatitude'])
            lon = convert_to_degrees(gps_data['GPSLongitude'])
            latitude = lat if gps_data.get('GPSLatitudeRef') == 'N' else -lat
            longitude = lon if gps_data.get('GPSLongitudeRef') == 'E' else -lon

        return latitude, longitude
    except Exception as e:
        print(f"Error extracting EXIF data: {e}")
        return None, None


def init_db():
    print("Initializing database...")
    db = DatabaseManager('users.db')
    db.create_db()
    if not db.search('SELECT username FROM users WHERE username = ?', ('MrsX',)):
        db.add_user('MrsX', 'mrsx@example.com', 'securepassword123', 'admin')
        print("Admin user 'MrsX' created.")
    else:
        print("Admin user 'MrsX' already exists.")
    db.close()


init_db()


@app.route('/sign-in/', methods=['GET', 'POST'])
def sign_in():
    if request.method == 'GET':
        return render_template('auth.html', error_message='', logged_in='logged_in' in session,
                               username=session.get('username'))

    username = request.form.get('username')
    email = request.form.get('email')
    if not username or not email:
        return render_template('auth.html', error_message='Please provide username and email',
                               logged_in='logged_in' in session, username=session.get('username'))

    db = DatabaseManager('users.db')
    existing_user = db.check_user_exists(username, email)
    if existing_user:
        if any(user[0] == username for user in existing_user):
            error_message = 'Username already exists'
        elif any(user[1] == email for user in existing_user):
            error_message = 'Email already exists'
        db.close()
        return render_template('auth.html', error_message=error_message, logged_in='logged_in' in session,
                               username=session.get('username'))

    db.close()
    return render_template('password.html', username=username, email=email, error_message='',
                           logged_in='logged_in' in session)


@app.route('/register/', methods=['POST'])
def register():
    username = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')
    conf_password = request.form.get('conf-password')

    if not all([username, email, password, conf_password]):
        return render_template('password.html', username=username or '', email=email or '',
                               error_message='Please complete all fields', logged_in='logged_in' in session)

    if password != conf_password:
        return render_template('password.html', username=username, email=email, error_message='Passwords do not match',
                               logged_in='logged_in' in session)

    db = DatabaseManager('users.db')
    existing_user = db.check_user_exists(username, email)
    if existing_user:
        db.close()
        if any(user[0] == username for user in existing_user):
            return render_template('auth.html', error_message='Username already exists',
                                   logged_in='logged_in' in session, username=session.get('username'))
        if any(user[1] == email for user in existing_user):
            return render_template('auth.html', error_message='Email already exists', logged_in='logged_in' in session,
                                   username=session.get('username'))

    db.add_user(username, email, password)
    db.close()
    return redirect(url_for('login_screen', success_message='Registration was successful'))


@app.route('/login/', methods=['POST', 'GET'])
def login_screen():
    if 'logged_in' in session:
        return redirect(url_for('index'))  # Redirect logged-in users to index
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if not username or not password:
            return render_template('login.html', error_message='Please enter both username and password',
                                   logged_in=False, username=None)

        db = DatabaseManager('users.db')
        query = 'SELECT password_hash FROM users WHERE username = ?'
        result = db.search(query, (username,))
        db.close()

        if not result:
            return render_template('login.html', error_message='Invalid username or password', logged_in=False,
                                   username=None)

        stored_hash = result[0][0]
        if bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8')):
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error_message='Invalid username or password', logged_in=False,
                                   username=None)
    success_message = request.args.get('success_message', '')
    return render_template('login.html', success_message=success_message, logged_in=False, username=None)


@app.route('/upload/', methods=['POST'])
def upload():
    if 'logged_in' not in session:
        return jsonify({'error': 'Please log in to upload images'}), 401

    # Check if this is the new modal upload format or old simple upload
    if 'files' in request.files and 'images' in request.form:
        # New modal upload format
        import json
        files = request.files.getlist('files')
        image_data = json.loads(request.form.get('images'))

        db = DatabaseManager('users.db')
        response = {'uploaded': [], 'failed': []}

        # Create upload folder if it doesn't exist
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        for i, file in enumerate(files):
            if i >= len(image_data):
                response['failed'].append(f"Missing metadata for file {file.filename}")
                continue

            image_info = image_data[i]

            try:
                if file and allowed_file(file.filename):
                    # Create filename with timestamp and username
                    original_name = file.filename.rsplit('.', 1)[0].lower()
                    extension = file.filename.rsplit('.', 1)[1].lower()
                    final_filename = f"{datetime.date.today()}-{session['username']}-{original_name}.{extension}"
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], final_filename)

                    # Save the file
                    file.save(file_path)

                    # Extract coordinates from metadata or use provided ones
                    extracted_lat, extracted_lng = get_exif_location(file_path)
                    latitude = image_info.get('latitude') or extracted_lat
                    longitude = image_info.get('longitude') or extracted_lng

                    # Save to database
                    db.add_image(
                        file_path,
                        latitude,
                        longitude,
                        username=session['username'],
                        species_name=image_info.get('species_name'),
                        species_type=image_info.get('species_type'),
                        icon_class=image_info.get('icon_class')
                    )

                    response['uploaded'].append({
                        'filename': final_filename,
                        'species_name': image_info.get('species_name'),
                        'species_type': image_info.get('species_type'),
                        'latitude': latitude,
                        'longitude': longitude
                    })
                else:
                    response['failed'].append(f"Invalid file type: {file.filename}")

            except Exception as e:
                print(f"Error processing {file.filename}: {e}")
                response['failed'].append(f"Error processing {file.filename}: {str(e)}")

        db.close()
        return jsonify(response), 200

    elif 'images' in request.files:
        # Old simple upload format (for backward compatibility)
        files = request.files.getlist('images')
        db = DatabaseManager('users.db')
        response = {'uploaded': [], 'failed': []}

        # Create upload folder if it doesn't exist
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        for file in files:
            try:
                if file and allowed_file(file.filename):
                    # Create filename with timestamp and username
                    original_name = file.filename.rsplit('.', 1)[0].lower()
                    extension = file.filename.rsplit('.', 1)[1].lower()
                    final_filename = f"{datetime.date.today()}-{session['username']}-{original_name}.{extension}"
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], final_filename)

                    # Save the file
                    file.save(file_path)

                    # Extract coordinates from EXIF
                    latitude, longitude = get_exif_location(file_path)

                    # Save to database with basic info
                    db.add_image(
                        file_path,
                        latitude,
                        longitude,
                        username=session['username'],
                        species_name=None,  # No species info for simple upload
                        species_type=None,
                        icon_class=None
                    )

                    response['uploaded'].append({
                        'filename': final_filename,
                        'latitude': latitude,
                        'longitude': longitude
                    })
                else:
                    response['failed'].append(f"Invalid file type: {file.filename}")

            except Exception as e:
                print(f"Error processing {file.filename}: {e}")
                response['failed'].append(f"Error processing {file.filename}: {str(e)}")

        db.close()
        return jsonify(response), 200

    else:
        return jsonify({'error': 'No files provided'}), 400


@app.route('/get_images/')
def get_images():
    db = DatabaseManager('users.db')
    query = '''
        SELECT file_path, latitude, longitude, username, upload_date, species_name, species_type, icon_class
        FROM images 
        WHERE latitude IS NOT NULL AND longitude IS NOT NULL
    '''
    images = db.search(query, ())
    db.close()
    result = []
    for image in images:
        result.append({
            'filename': os.path.basename(image[0]),
            'latitude': image[1],
            'longitude': image[2],
            'username': image[3],
            'upload_date': image[4],
            'species_name': image[5],
            'species_type': image[6],
            'icon_class': image[7]
        })
    return jsonify(result)


@app.route('/logout/')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    return redirect(url_for('index'))


@app.route('/about/')
def about():
    return render_template('about.html', logged_in='logged_in' in session, username=session.get('username'))


@app.route('/upload/')
def upload_page():
    return render_template('upload.html', logged_in='logged_in' in session, username=session.get('username'))


@app.route('/')
def index():
    return render_template('index.html', logged_in='logged_in' in session, username=session.get('username'))


if __name__ == '__main__':
    print("Starting...")
    app.run(port=8080, debug=True)  # Set debug=True for development