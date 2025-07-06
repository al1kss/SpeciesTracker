# 🌲 ForestSpeciesTracker

A comprehensive website for tracking and analyzing forest biodiversity through species mapping. Built specifically for ecological research and International Baccalaureate (IB) Environmental Systems and Societies (ESS) classes. Includes information on how to run it below

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Ecological Analysis](#ecological-analysis)
- [Deployment](#deployment)
- [Customization for Different Forests](#customization-for-different-forests)
- [Technical Details](#technical-details)
- [Contributing](#contributing)
- [License](#license)

## 🌟 Overview

ForestSpeciesTracker was created to solve the problem of lost field data in ecological research. Instead of relying on paper logs that get lost or damaged, this application allows students and researchers to:

- Upload species images with automatic GPS coordinate extraction
- Visualize species locations on an interactive map
- Perform quadrant transect sampling analysis
- Generate comprehensive ecological reports with statistical analysis

**Built during the Shipwrecked hackathon for UWC ISAK Japan's forest ecosystem research.**

## ✨ Features

### Core Functionality
- **Image Upload with GPS Extraction**: Automatically extracts coordinates from image EXIF data
- **Interactive Forest Mapping**: Leaflet.js-powered map with custom forest boundaries
- **Species Database**: SQLite database for storing species information and user data
- **User Authentication**: Secure login system with bcrypt password hashing

### Ecological Analysis Tools
- **Quadrant Sampling**: Automated grid-based transect sampling
- **Statistical Analysis**: Mean, standard deviation, population estimation
- **Distribution Graphs**: Species count by type, spatial distribution
- **Kite Diagrams**: Transect-based species abundance visualization
- **Heatmaps**: Quadrant species density mapping
- **Percentage Frequency Analysis**: Species occurrence across quadrats

### Data Visualization
- **Interactive Charts**: Matplotlib-generated graphs with dark theme
- **Download Options**: Export individual graphs as PNG files
- **Real-time Updates**: Live map updates when new species are added
- **Marker Clustering**: Organized display of multiple species locations

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- pip (Python package installer)

### Step 1: Clone the Repository
```bash
git clone https://github.com/al1kss/SpeciesTracker.git
cd SpeciesTracker
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Application
```bash
python app.py
```

The application will start on `http://localhost:8080`

## ⚙️ Configuration

### 1. Admin User Setup
The application creates a default admin user:
- **Username**: `MrsX`
- **Email**: `mrsx@example.com`
- **Password**: `securepassword123`

**Important**: Change the admin password in `app.py` before deployment:
```python
db.add_user('YourUsername', 'your.email@example.com', 'your_secure_password', 'admin')
```

### 2. Map API Key Configuration
Get a free API key from [Thunderforest](https://www.thunderforest.com/):

1. Sign up for a free account
2. Generate an API key
3. Replace the API key in `templates/index.html`:
```javascript
L.tileLayer('https://tile.thunderforest.com/cycle/{z}/{x}/{y}.png?apikey=YOUR_API_KEY_HERE', {
    attribution: '&copy; OpenStreetMap'
}).addTo(map);
```

### 3. Forest Boundary Configuration
The default coordinates are set for Karuizawa forest. To use your own forest:

1. Get coordinates using Google Maps
2. Update the bounds in `app.py`:
```python
FOREST_BOUNDS = {
    'top_left': [YOUR_LAT, YOUR_LNG],
    'top_right': [YOUR_LAT, YOUR_LNG],
    'bottom_left': [YOUR_LAT, YOUR_LNG],
    'bottom_right': [YOUR_LAT, YOUR_LNG]
}
```

## 📱 Usage

### For Students/Researchers

1. **Register an Account**
   - Click "Sign Up" and create your account
   - Login with your credentials

2. **Upload Species Images**
   - Navigate to the Upload page
   - Drag and drop images or click "Select Images"
   - Fill in species information (name, type, icon)
   - Verify GPS coordinates on the map
   - Upload to the database

3. **View Species Map**
   - Explore the interactive map on the homepage
   - Click markers to view species details
   - Toggle quadrant sampling overlay
   - View species clustering

4. **Generate Analysis**
   - Click "Generate Insights" button
   - Wait for analysis processing
   - View comprehensive ecological reports
   - Download individual graphs

### For Teachers/Administrators

1. **Monitor Student Data**
   - Access all uploaded species data
   - View real-time map updates
   - Generate class-wide reports

2. **Ecological Teaching Tools**
   - Use quadrant sampling for field exercises
   - Generate statistical reports for assignments
   - Export data for further analysis

## 📊 Ecological Analysis

### Quadrant Sampling
- **Grid System**: 6×11 quadrats (66 total)
- **Automated Calculation**: Species count per quadrat
- **Visual Representation**: Color-coded heatmap

### Statistical Measures
- **Population Estimation**: `(Total Area ÷ Sampled Area) × Species Count`
- **Percentage Frequency**: `(Quadrats with Species ÷ Total Quadrats) × 100`
- **Diversity Metrics**: Species richness and distribution
- **Spatial Analysis**: Geographic clustering patterns

### Generated Reports Include:
1. Species Distribution Bar Chart
2. Quadrant Density Heatmap
3. Population Estimation Analysis
4. Percentage Frequency Chart
5. Kite Diagrams (Transect Analysis)
6. Species Diversity Pie Chart
7. Statistical Summary
8. Spatial Distribution Scatter Plot
9. Upload Timeline

## 🌐 Deployment

### PythonAnywhere Deployment

1. **Upload Files**
   - Upload all project files to PythonAnywhere
   - Ensure `requirements.txt` is in the root directory

2. **Install Dependencies**
   ```bash
   pip3.9 install --user -r requirements.txt
   ```

3. **Database Setup**
   - The SQLite database will be created automatically on first run
   - Ensure proper file permissions

4. **Storage Considerations**
   - PythonAnywhere free tier: 512MB disk limit
   - Approximately 500+ image uploads before reaching limit
   - Monitor storage usage regularly

5. **WSGI Configuration**
   - Set up WSGI file to point to your `app.py`
   - Configure static files serving

## 🗺️ Customization for Different Forests

### Step 1: Get Forest Coordinates
1. Open Google Maps
2. Right-click on forest corners to get coordinates
3. Copy latitude and longitude values

### Step 2: Update Forest Bounds
Replace coordinates in `app.py`:
```python
FOREST_BOUNDS = {
    'top_left': [36.360569, 138.548037],     # Northwest corner
    'top_right': [36.36286106538286, 138.55092177935643],   # Northeast corner
    'bottom_left': [36.355673, 138.551822],  # Southwest corner
    'bottom_right': [36.35843935596947, 138.55464579115937] # Southeast corner
}
```

### Step 3: Update Map Center
In `templates/index.html`, update the map center:
```javascript
map.setView([YOUR_CENTER_LAT, YOUR_CENTER_LNG], 16);
```

### Step 4: Adjust Quadrat Grid (Optional)
Modify grid size in `app.py`:
```python
V_DIVISIONS = 6  # Vertical divisions
H_DIVISIONS = 11 # Horizontal divisions
```

## 🔧 Technical Details

### Technology Stack
- **Backend**: Flask (Python)
- **Database**: SQLite
- **Frontend**: HTML5, CSS3, JavaScript
- **Mapping**: Leaflet.js with Thunderforest tiles
- **Charts**: Matplotlib with dark theme
- **Authentication**: bcrypt password hashing
- **Image Processing**: Pillow (PIL) for EXIF data extraction

### File Structure
```
SpeciesTracker/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── static/
│   ├── uploads/          # Uploaded images storage
│   └── logo.png         # Application logo
├── templates/
│   ├── index.html       # Main map interface
│   ├── upload.html      # Image upload page
│   ├── output.html      # Analysis results
│   ├── about.html       # About page
│   ├── auth.html        # Registration page
│   ├── login.html       # Login page
│   ├── password.html    # Password setup
│   ├── header.html      # Navigation header
│   ├── footer.html      # Page footer
│   └── divider.html     # Section divider
└── users.db             # SQLite database (created automatically)
```

### Database Schema
```sql
-- Users table
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'student'
);

-- Images table
CREATE TABLE images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL,
    latitude REAL,
    longitude REAL,
    username TEXT,
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    species_name TEXT,
    species_type TEXT,
    icon_class TEXT
);
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

© 2025 ForestSpeciesTracker. Built with passion by Alikhan Abdykaimov for Shipwrecked hackathon.

## 🆘 Support

For support or questions:
- GitHub Issues: [Create an issue](https://github.com/al1kss/SpeciesTracker/issues)
- LinkedIn: [Alikhan Abdykaimov](https://www.linkedin.com/in/alikhan-abdykaimov/)
- Instagram: [@al1k.ss](https://www.instagram.com/al1k.ss/)

---

**Made for students, by a student** 🌿

*Created during Shipwrecked - the world's first island hackathon*
