import datetime
import os
import secrets
from random import random
import base64
import io
import math

from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import sqlite3
import bcrypt
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

# Add matplotlib imports
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from collections import Counter
import seaborn as sns

# Set matplotlib style for dark theme
plt.style.use('dark_background')
sns.set_palette("husl")


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
secret = secrets.token_hex(16)
print(secret)
app.secret_key = str(secret)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'HEIF'}

# Forest boundary coordinates (from your index.html)
FOREST_BOUNDS = {
    'top_left': [36.360569, 138.548037],
    'top_right': [36.36286106538286, 138.55092177935643],
    'bottom_left': [36.355673, 138.551822],
    'bottom_right': [36.35843935596947, 138.55464579115937]
}

V_DIVISIONS = 6
H_DIVISIONS = 11
TOTAL_QUADRATS = V_DIVISIONS * H_DIVISIONS


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
        longitude = None
        if 'GPSLatitude' in gps_data and 'GPSLongitude' in gps_data:
            lat = convert_to_degrees(gps_data['GPSLatitude'])
            lon = convert_to_degrees(gps_data['GPSLongitude'])
            latitude = lat if gps_data.get('GPSLatitudeRef') == 'N' else -lat
            longitude = lon if gps_data.get('GPSLongitudeRef') == 'E' else -lon

        return latitude, longitude
    except Exception as e:
        print(f"Error extracting EXIF data: {e}")
        return None, None


def is_point_in_polygon(point, vertices):
    """Ray-casting algorithm to check if point is inside polygon"""
    x, y = point
    inside = False
    j = len(vertices) - 1

    for i in range(len(vertices)):
        xi, yi = vertices[i]
        xj, yj = vertices[j]

        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i

    return inside


def get_quadrant_for_point(lat, lng):
    """Get quadrant index for a given point using bilinear interpolation"""
    top_left = FOREST_BOUNDS['top_left']
    top_right = FOREST_BOUNDS['top_right']
    bottom_left = FOREST_BOUNDS['bottom_left']
    bottom_right = FOREST_BOUNDS['bottom_right']

    for i in range(V_DIVISIONS):
        for j in range(H_DIVISIONS):
            # Calculate fractions for interpolation
            v_fraction1 = i / V_DIVISIONS
            v_fraction2 = (i + 1) / V_DIVISIONS
            h_fraction1 = j / H_DIVISIONS
            h_fraction2 = (j + 1) / H_DIVISIONS

            # Interpolate quadrilateral vertices
            tl_top_lat = top_left[0] + v_fraction1 * (top_right[0] - top_left[0])
            tl_top_lng = top_left[1] + v_fraction1 * (top_right[1] - top_left[1])
            tl_bottom_lat = bottom_left[0] + v_fraction1 * (bottom_right[0] - bottom_left[0])
            tl_bottom_lng = bottom_left[1] + v_fraction1 * (bottom_right[1] - bottom_left[1])
            top_left_vertex = [
                tl_top_lat + h_fraction1 * (tl_bottom_lat - tl_top_lat),
                tl_top_lng + h_fraction1 * (tl_bottom_lng - tl_top_lng)
            ]

            tr_top_lat = top_left[0] + v_fraction2 * (top_right[0] - top_left[0])
            tr_top_lng = top_left[1] + v_fraction2 * (top_right[1] - top_left[1])
            tr_bottom_lat = bottom_left[0] + v_fraction2 * (bottom_right[0] - bottom_left[0])
            tr_bottom_lng = bottom_left[1] + v_fraction2 * (bottom_right[1] - bottom_left[1])
            top_right_vertex = [
                tr_top_lat + h_fraction1 * (tr_bottom_lat - tr_top_lat),
                tr_top_lng + h_fraction1 * (tr_bottom_lng - tr_top_lng)
            ]

            bottom_right_vertex = [
                tr_top_lat + h_fraction2 * (tr_bottom_lat - tr_top_lat),
                tr_top_lng + h_fraction2 * (tr_bottom_lng - tr_top_lng)
            ]

            bottom_left_vertex = [
                tl_top_lat + h_fraction2 * (tl_bottom_lat - tl_top_lat),
                tl_top_lng + h_fraction2 * (tl_bottom_lng - tl_top_lng)
            ]

            vertices = [top_left_vertex, top_right_vertex, bottom_right_vertex, bottom_left_vertex]

            if is_point_in_polygon([lat, lng], vertices):
                return i * H_DIVISIONS + j, vertices

    return None, None


def calculate_quadrant_data(images):
    """Calculate species distribution across quadrants"""
    quadrant_counts = {}
    species_by_quadrant = {}

    for image in images:
        if image[1] and image[2]:  # lat, lng
            quadrant_idx, vertices = get_quadrant_for_point(image[1], image[2])
            if quadrant_idx is not None:
                species_type = image[6] or 'Unknown'  # species_type

                if quadrant_idx not in quadrant_counts:
                    quadrant_counts[quadrant_idx] = 0
                    species_by_quadrant[quadrant_idx] = []

                quadrant_counts[quadrant_idx] += 1
                species_by_quadrant[quadrant_idx].append(species_type)

    return quadrant_counts, species_by_quadrant


def generate_matplotlib_graphs(images):
    """Generate all ecological analysis graphs"""
    graphs = {}

    # Convert images to more manageable format
    image_data = []
    for img in images:
        image_data.append({
            'filename': os.path.basename(img[0]),
            'latitude': img[1],
            'longitude': img[2],
            'username': img[3],
            'upload_date': img[4],
            'species_name': img[5],
            'species_type': img[6],
            'icon_class': img[7]
        })

    # Filter valid coordinates
    valid_images = [img for img in image_data if img['latitude'] and img['longitude']]

    if not valid_images:
        return {}

    # Calculate quadrant data
    quadrant_counts, species_by_quadrant = calculate_quadrant_data(images)

    # 1. Species Distribution by Type
    graphs['species_distribution'] = create_species_distribution_chart(valid_images)

    # 2. Quadrant Heatmap
    graphs['quadrant_heatmap'] = create_quadrant_heatmap(quadrant_counts)

    # 3. Population Estimation Chart
    graphs['population_estimation'] = create_population_estimation_chart(quadrant_counts, valid_images)

    # 4. Percentage Frequency Chart
    graphs['percentage_frequency'] = create_percentage_frequency_chart(species_by_quadrant, valid_images)

    # 5. Kite Diagram
    graphs['kite_diagram'] = create_kite_diagram(quadrant_counts)

    # 6. Species Diversity Pie Chart
    graphs['diversity_pie'] = create_diversity_pie_chart(valid_images)

    # 7. Statistical Summary
    graphs['statistical_summary'] = create_statistical_summary(quadrant_counts, valid_images)

    # 8. Spatial Distribution Scatter
    graphs['spatial_scatter'] = create_spatial_scatter(valid_images)

    # 9. Upload Timeline
    graphs['upload_timeline'] = create_upload_timeline(valid_images)

    return graphs


def create_species_distribution_chart(images):
    """Species count by type bar chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#1a1a1a')

    species_types = [img['species_type'] or 'Unknown' for img in images]
    type_counts = Counter(species_types)

    bars = ax.bar(type_counts.keys(), type_counts.values(),
                  color=['#fab633', '#f39c12', '#e74c3c', '#9b59b6', '#3498db', '#2ecc71'])

    ax.set_title('Species Distribution by Type', fontsize=16, color='#e0e0e0', pad=20)
    ax.set_xlabel('Species Type', fontsize=12, color='#e0e0e0')
    ax.set_ylabel('Count', fontsize=12, color='#e0e0e0')
    ax.tick_params(colors='#e0e0e0')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom', color='#e0e0e0')

    plt.tight_layout()
    return fig_to_base64(fig)


def create_quadrant_heatmap(quadrant_counts):
    """Quadrant species density heatmap"""
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('#1a1a1a')

    # Create 2D array for heatmap
    heatmap_data = np.zeros((V_DIVISIONS, H_DIVISIONS))

    for quadrant_idx, count in quadrant_counts.items():
        row = quadrant_idx // H_DIVISIONS
        col = quadrant_idx % H_DIVISIONS
        heatmap_data[row, col] = count

    im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Species Count', color='#e0e0e0')
    cbar.ax.yaxis.set_tick_params(color='#e0e0e0')

    # Add text annotations
    for i in range(V_DIVISIONS):
        for j in range(H_DIVISIONS):
            text = ax.text(j, i, int(heatmap_data[i, j]),
                           ha="center", va="center", color="black", fontweight='bold')

    ax.set_title('Quadrant Species Density Heatmap', fontsize=16, color='#e0e0e0', pad=20)
    ax.set_xlabel('Horizontal Divisions', fontsize=12, color='#e0e0e0')
    ax.set_ylabel('Vertical Divisions', fontsize=12, color='#e0e0e0')
    ax.tick_params(colors='#e0e0e0')

    plt.tight_layout()
    return fig_to_base64(fig)


def create_population_estimation_chart(quadrant_counts, images):
    """Population estimation using ecological formula"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor('#1a1a1a')

    # Calculate total area and sampled area
    total_forest_area = 1000  # Assume 1000 m² total forest area
    sampled_quadrats = len(quadrant_counts)
    area_per_quadrat = total_forest_area / TOTAL_QUADRATS
    total_sampled_area = sampled_quadrats * area_per_quadrat

    # Group by species type
    species_counts = {}
    for img in images:
        species_type = img['species_type'] or 'Unknown'
        species_counts[species_type] = species_counts.get(species_type, 0) + 1

    # Calculate estimated populations
    estimated_populations = {}
    for species, count in species_counts.items():
        estimated_pop = (total_forest_area / total_sampled_area) * count
        estimated_populations[species] = estimated_pop

    # Chart 1: Observed vs Estimated
    species_names = list(species_counts.keys())
    observed = list(species_counts.values())
    estimated = list(estimated_populations.values())

    x = np.arange(len(species_names))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, observed, width, label='Observed Count', color='#fab633')
    bars2 = ax1.bar(x + width / 2, estimated, width, label='Estimated Population', color='#3498db')

    ax1.set_title('Population Estimation Analysis', fontsize=14, color='#e0e0e0')
    ax1.set_xlabel('Species Type', fontsize=12, color='#e0e0e0')
    ax1.set_ylabel('Count', fontsize=12, color='#e0e0e0')
    ax1.set_xticks(x)
    ax1.set_xticklabels(species_names)
    ax1.legend()
    ax1.tick_params(colors='#e0e0e0')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                     f'{height:.1f}', ha='center', va='bottom', color='#e0e0e0', fontsize=9)

    # Chart 2: Formula explanation
    ax2.text(0.1, 0.8, 'Population Estimation Formula:', fontsize=14, fontweight='bold',
             color='#fab633', transform=ax2.transAxes)
    ax2.text(0.1, 0.65, 'Estimated Population = (Total Area ÷ Area Sampled) × Total Count',
             fontsize=12, color='#e0e0e0', transform=ax2.transAxes)
    ax2.text(0.1, 0.5, f'Total Forest Area: {total_forest_area} m²',
             fontsize=11, color='#e0e0e0', transform=ax2.transAxes)
    ax2.text(0.1, 0.4, f'Sampled Area: {total_sampled_area:.1f} m²',
             fontsize=11, color='#e0e0e0', transform=ax2.transAxes)
    ax2.text(0.1, 0.3, f'Quadrats with Species: {sampled_quadrats}/{TOTAL_QUADRATS}',
             fontsize=11, color='#e0e0e0', transform=ax2.transAxes)
    ax2.text(0.1, 0.1, f'Multiplication Factor: {total_forest_area / total_sampled_area:.2f}',
             fontsize=11, color='#fab633', fontweight='bold', transform=ax2.transAxes)

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')

    plt.tight_layout()
    return fig_to_base64(fig)


def create_percentage_frequency_chart(species_by_quadrant, images):
    """Percentage frequency analysis"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor('#1a1a1a')

    # Calculate percentage frequency for each species type
    species_frequency = {}
    all_species = set()

    for img in images:
        species_type = img['species_type'] or 'Unknown'
        all_species.add(species_type)

    for species in all_species:
        quadrats_with_species = 0
        for quadrant_species in species_by_quadrant.values():
            if species in quadrant_species:
                quadrats_with_species += 1

        percentage_freq = (quadrats_with_species / TOTAL_QUADRATS) * 100
        species_frequency[species] = percentage_freq

    # Chart 1: Percentage Frequency Bar Chart
    species_names = list(species_frequency.keys())
    frequencies = list(species_frequency.values())

    bars = ax1.bar(species_names, frequencies, color='#fab633')
    ax1.set_title('Percentage Frequency by Species', fontsize=14, color='#e0e0e0')
    ax1.set_xlabel('Species Type', fontsize=12, color='#e0e0e0')
    ax1.set_ylabel('Percentage Frequency (%)', fontsize=12, color='#e0e0e0')
    ax1.tick_params(colors='#e0e0e0')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'{height:.1f}%', ha='center', va='bottom', color='#e0e0e0')

    # Chart 2: Formula explanation
    ax2.text(0.1, 0.8, 'Percentage Frequency Formula:', fontsize=14, fontweight='bold',
             color='#fab633', transform=ax2.transAxes)
    ax2.text(0.1, 0.65, '% Frequency = (Quadrats with Species ÷ Total Quadrats) × 100',
             fontsize=12, color='#e0e0e0', transform=ax2.transAxes)
    ax2.text(0.1, 0.5, f'Total Quadrats: {TOTAL_QUADRATS}',
             fontsize=11, color='#e0e0e0', transform=ax2.transAxes)

    y_pos = 0.35
    for species, freq in species_frequency.items():
        quadrats_with = int(freq * TOTAL_QUADRATS / 100)
        ax2.text(0.1, y_pos, f'{species}: {quadrats_with}/{TOTAL_QUADRATS} = {freq:.1f}%',
                 fontsize=10, color='#e0e0e0', transform=ax2.transAxes)
        y_pos -= 0.05

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')

    plt.tight_layout()
    return fig_to_base64(fig)


def create_kite_diagram(quadrant_counts):
    """Kite diagram showing species distribution along transects"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.patch.set_facecolor('#1a1a1a')

    # Horizontal transect (middle row)
    middle_row = V_DIVISIONS // 2
    horizontal_data = []
    for j in range(H_DIVISIONS):
        quadrant_idx = middle_row * H_DIVISIONS + j
        count = quadrant_counts.get(quadrant_idx, 0)
        horizontal_data.append(count)

    # Vertical transect (middle column)
    middle_col = H_DIVISIONS // 2
    vertical_data = []
    for i in range(V_DIVISIONS):
        quadrant_idx = i * H_DIVISIONS + middle_col
        count = quadrant_counts.get(quadrant_idx, 0)
        vertical_data.append(count)

    # Plot horizontal kite diagram
    x_positions = range(H_DIVISIONS)
    ax1.fill_between(x_positions, 0, horizontal_data, alpha=0.7, color='#fab633', label='Species Count')
    ax1.fill_between(x_positions, 0, [-x for x in horizontal_data], alpha=0.7, color='#fab633')
    ax1.plot(x_positions, horizontal_data, 'o-', color='#e74c3c', linewidth=2, markersize=6)
    ax1.plot(x_positions, [-x for x in horizontal_data], 'o-', color='#e74c3c', linewidth=2, markersize=6)

    ax1.set_title('Horizontal Transect (Middle Row)', fontsize=14, color='#e0e0e0')
    ax1.set_xlabel('Quadrat Position (West to East)', fontsize=12, color='#e0e0e0')
    ax1.set_ylabel('Species Count', fontsize=12, color='#e0e0e0')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(colors='#e0e0e0')
    ax1.axhline(y=0, color='white', linewidth=1)

    # Plot vertical kite diagram
    y_positions = range(V_DIVISIONS)
    ax2.fill_betweenx(y_positions, 0, vertical_data, alpha=0.7, color='#3498db', label='Species Count')
    ax2.fill_betweenx(y_positions, 0, [-x for x in vertical_data], alpha=0.7, color='#3498db')
    ax2.plot(vertical_data, y_positions, 'o-', color='#e74c3c', linewidth=2, markersize=6)
    ax2.plot([-x for x in vertical_data], y_positions, 'o-', color='#e74c3c', linewidth=2, markersize=6)

    ax2.set_title('Vertical Transect (Middle Column)', fontsize=14, color='#e0e0e0')
    ax2.set_ylabel('Quadrat Position (South to North)', fontsize=12, color='#e0e0e0')
    ax2.set_xlabel('Species Count', fontsize=12, color='#e0e0e0')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(colors='#e0e0e0')
    ax2.axvline(x=0, color='white', linewidth=1)

    plt.tight_layout()
    return fig_to_base64(fig)


def create_diversity_pie_chart(images):
    """Species diversity donut chart"""
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#1a1a1a')

    species_types = [img['species_type'] or 'Unknown' for img in images]
    type_counts = Counter(species_types)

    colors = ['#fab633', '#f39c12', '#e74c3c', '#9b59b6', '#3498db', '#2ecc71', '#f1c40f', '#e67e22']

    # Create donut chart
    wedges, texts, autotexts = ax.pie(type_counts.values(), labels=type_counts.keys(),
                                      autopct='%1.1f%%', colors=colors[:len(type_counts)],
                                      pctdistance=0.85, wedgeprops=dict(width=0.5))

    # Beautify text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')

    for text in texts:
        text.set_color('#e0e0e0')
        text.set_fontsize(11)

    ax.set_title('Species Diversity Distribution', fontsize=16, color='#e0e0e0', pad=20)

    # Add center text
    ax.text(0, 0, f'Total\nSpecies\n{len(images)}', ha='center', va='center',
            fontsize=14, fontweight='bold', color='#fab633')

    plt.tight_layout()
    return fig_to_base64(fig)


def create_statistical_summary(quadrant_counts, images):
    """Statistical summary of quadrant data"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.patch.set_facecolor('#1a1a1a')

    # Get all quadrant counts (including zeros)
    all_counts = []
    for i in range(TOTAL_QUADRATS):
        all_counts.append(quadrant_counts.get(i, 0))

    # Calculate statistics
    mean_count = np.mean(all_counts)
    std_count = np.std(all_counts)
    max_count = np.max(all_counts)
    min_count = np.min(all_counts)

    # 1. Histogram of quadrat counts
    ax1.hist(all_counts, bins=max(max_count, 5), color='#fab633', alpha=0.7, edgecolor='white')
    ax1.axvline(mean_count, color='#e74c3c', linestyle='--', linewidth=2, label=f'Mean: {mean_count:.2f}')
    ax1.set_title('Distribution of Species Counts per Quadrat', fontsize=12, color='#e0e0e0')
    ax1.set_xlabel('Species Count', fontsize=10, color='#e0e0e0')
    ax1.set_ylabel('Frequency', fontsize=10, color='#e0e0e0')
    ax1.legend()
    ax1.tick_params(colors='#e0e0e0')

    # 2. Box plot
    box = ax2.boxplot(all_counts, patch_artist=True)
    box['boxes'][0].set_facecolor('#fab633')
    box['boxes'][0].set_alpha(0.7)
    ax2.set_title('Statistical Distribution', fontsize=12, color='#e0e0e0')
    ax2.set_ylabel('Species Count', fontsize=10, color='#e0e0e0')
    ax2.tick_params(colors='#e0e0e0')

    # 3. Statistics text
    stats_text = f"""
    Statistical Summary:

    Mean: {mean_count:.2f}
    Standard Deviation: {std_count:.2f}
    Maximum: {max_count}
    Minimum: {min_count}
    Total Quadrats: {TOTAL_QUADRATS}
    Occupied Quadrats: {len(quadrant_counts)}
    Empty Quadrats: {TOTAL_QUADRATS - len(quadrant_counts)}

    Occupancy Rate: {len(quadrant_counts) / TOTAL_QUADRATS * 100:.1f}%
    """

    ax3.text(0.1, 0.9, stats_text, fontsize=11, color='#e0e0e0',
             transform=ax3.transAxes, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='#2d2d2d', alpha=0.8))
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')

    # 4. Variance analysis
    variance = np.var(all_counts)
    cv = (std_count / mean_count) * 100 if mean_count > 0 else 0

    # Create categories for analysis
    categories = ['Low (0-1)', 'Medium (2-3)', 'High (4+)']
    cat_counts = [
        sum(1 for x in all_counts if 0 <= x <= 1),
        sum(1 for x in all_counts if 2 <= x <= 3),
        sum(1 for x in all_counts if x >= 4)
    ]

    bars = ax4.bar(categories, cat_counts, color=['#3498db', '#fab633', '#e74c3c'])
    ax4.set_title('Density Categories', fontsize=12, color='#e0e0e0')
    ax4.set_ylabel('Number of Quadrats', fontsize=10, color='#e0e0e0')
    ax4.tick_params(colors='#e0e0e0')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                 f'{int(height)}', ha='center', va='bottom', color='#e0e0e0')

    plt.tight_layout()
    return fig_to_base64(fig)


def create_spatial_scatter(images):
    """Spatial distribution scatter plot"""
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('#1a1a1a')

    # Group by species type
    species_data = {}
    for img in images:
        species_type = img['species_type'] or 'Unknown'
        if species_type not in species_data:
            species_data[species_type] = {'lat': [], 'lng': []}
        species_data[species_type]['lat'].append(img['latitude'])
        species_data[species_type]['lng'].append(img['longitude'])

    colors = ['#fab633', '#f39c12', '#e74c3c', '#9b59b6', '#3498db', '#2ecc71']

    # Plot each species type
    for i, (species, data) in enumerate(species_data.items()):
        ax.scatter(data['lng'], data['lat'], c=colors[i % len(colors)],
                   label=species, s=60, alpha=0.7, edgecolors='white', linewidth=1)

    # Add forest boundary
    boundary_coords = [
        FOREST_BOUNDS['top_left'],
        FOREST_BOUNDS['top_right'],
        FOREST_BOUNDS['bottom_right'],
        FOREST_BOUNDS['bottom_left'],
        FOREST_BOUNDS['top_left']  # Close the polygon
    ]

    boundary_lng = [coord[1] for coord in boundary_coords]
    boundary_lat = [coord[0] for coord in boundary_coords]

    ax.plot(boundary_lng, boundary_lat, color='#fab633', linewidth=3, alpha=0.8, label='Forest Boundary')

    ax.set_title('Spatial Distribution of Species', fontsize=16, color='#e0e0e0', pad=20)
    ax.set_xlabel('Longitude', fontsize=12, color='#e0e0e0')
    ax.set_ylabel('Latitude', fontsize=12, color='#e0e0e0')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.tick_params(colors='#e0e0e0')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig_to_base64(fig)


def create_upload_timeline(images):
    """Timeline of species uploads"""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#1a1a1a')

    # Parse upload dates
    upload_dates = []
    for img in images:
        try:
            # Assuming upload_date is in format 'YYYY-MM-DD HH:MM:SS'
            date_str = img['upload_date'].split(' ')[0]  # Get just the date part
            upload_dates.append(date_str)
        except:
            continue

    if not upload_dates:
        # Create dummy data if no dates available
        ax.text(0.5, 0.5, 'No upload date data available', ha='center', va='center',
                fontsize=16, color='#e0e0e0', transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig_to_base64(fig)

    # Count uploads per date
    date_counts = Counter(upload_dates)
    dates = sorted(date_counts.keys())
    counts = [date_counts[date] for date in dates]

    # Plot timeline
    ax.plot(dates, counts, 'o-', color='#fab633', linewidth=3, markersize=8, markerfacecolor='#f39c12')
    ax.fill_between(dates, counts, alpha=0.3, color='#fab633')

    ax.set_title('Species Upload Timeline', fontsize=16, color='#e0e0e0', pad=20)
    ax.set_xlabel('Date', fontsize=12, color='#e0e0e0')
    ax.set_ylabel('Number of Uploads', fontsize=12, color='#e0e0e0')
    ax.tick_params(colors='#e0e0e0')
    ax.grid(True, alpha=0.3)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    plt.tight_layout()
    return fig_to_base64(fig)


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', facecolor='#1a1a1a',
                bbox_inches='tight', dpi=300, edgecolor='none')
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{img_str}"


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
        return redirect(url_for('index'))
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


@app.route('/output/')
def output():
    """Generate and display ecological analysis graphs"""
    db = DatabaseManager('users.db')
    query = '''
        SELECT file_path, latitude, longitude, username, upload_date, species_name, species_type, icon_class
        FROM images 
        WHERE latitude IS NOT NULL AND longitude IS NOT NULL
    '''
    images = db.search(query, ())
    db.close()

    if not images:
        # Return template with no data message
        return render_template('output.html', graphs={}, logged_in='logged_in' in session,
                               username=session.get('username'), no_data=True)

    # Generate all graphs
    graphs = generate_matplotlib_graphs(images)

    return render_template('output.html', graphs=graphs, logged_in='logged_in' in session,
                           username=session.get('username'), no_data=False)


@app.route('/download_graph/<graph_type>')
def download_graph(graph_type):
    """Download individual graph as PNG"""
    db = DatabaseManager('users.db')
    query = '''
        SELECT file_path, latitude, longitude, username, upload_date, species_name, species_type, icon_class
        FROM images 
        WHERE latitude IS NOT NULL AND longitude IS NOT NULL
    '''
    images = db.search(query, ())
    db.close()

    if not images:
        return jsonify({'error': 'No data available'}), 404

    graphs = generate_matplotlib_graphs(images)

    if graph_type not in graphs:
        return jsonify({'error': 'Graph type not found'}), 404

    # Convert base64 to bytes and return as download
    import base64
    from flask import Response

    graph_data = graphs[graph_type]
    # Remove the data URL prefix
    base64_data = graph_data.split(',')[1]
    image_bytes = base64.b64decode(base64_data)

    return Response(
        image_bytes,
        mimetype='image/png',
        headers={'Content-Disposition': f'attachment; filename={graph_type}_analysis.png'}
    )


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
    app.run(port=8080, debug=True)