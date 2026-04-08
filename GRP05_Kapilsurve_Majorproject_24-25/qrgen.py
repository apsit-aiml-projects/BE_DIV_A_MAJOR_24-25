import streamlit as st
import ephem
from datetime import datetime
from geopy.geocoders import Nominatim
import qrcode
from PIL import Image, ImageDraw, ImageFont
import io
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import streamlit as st

# Download punkt
nltk.download('punkt')
nltk.download('punkt_tab')
import nltk
import os

# Check if punkt is installed
try:
    nltk.data.find('tokenizers/punkt')
    print("Punkt tokenizer is installed.")
except LookupError:
    print("Punkt tokenizer is not found. Please check your NLTK data path.")

# Set your nltk_data directory

# Load pre-trained BERT model (fine-tuned for quality scoring)
@st.cache_resource
def load_model1():
    model = BertForSequenceClassification.from_pretrained("fine_tuned_bert_model")
    tokenizer = BertTokenizer.from_pretrained("fine_tuned_bert_tokenizer")
    return model, tokenizer
# Predefine qualities
qualities = ['Analytical', 'Practical', 'Creative', 'Leadership', 'Hard',
             'Smart', 'Technical', 'Caring', 'Communication', 'Persuasive',
             'Integrity', 'Imagination', 'Risk', 'Spontaneous', 'Determination',
             'Patience', 'Knowledge', 'Wisdom']

# Load PDF document
def load_pdf_document(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

# Preprocess the document text
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalnum()]
    processed_text = ' '.join(tokens)
    return processed_text

# Predict quality scores using the fine-tuned BERT model
def predict_quality_scores(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits.detach().numpy()
    scores = np.where(logits > 0, 1, 0)
    return scores.flatten()

# Extract and predict qualities for the entire document
def extract_and_predict_from_resume(pdf_file, model, tokenizer):
    document_text = load_pdf_document(pdf_file)

    # Use standard sent_tokenize with explicit language
    sentences = sent_tokenize(document_text, language='english')

    total_quality_scores = np.zeros(len(qualities))

    for sentence in sentences:
        processed_sentence = preprocess_text(sentence)
        predicted_scores = predict_quality_scores(processed_sentence, model, tokenizer)
        total_quality_scores += predicted_scores

    aggregated_quality_scores = dict(zip(qualities, total_quality_scores))
    return aggregated_quality_scores

# Plot the aggregated quality scores
def plot_quality_scores(aggregated_quality_scores):
    """Prepare a dataset from qualities and scores, display it in a single-row format, and plot the scores."""
    # Convert the dictionary into a DataFrame with a single row
    df = pd.DataFrame([aggregated_quality_scores])

    # Display the dataset in Streamlit
    st.write("### Aggregated Quality Scores Dataset")
    st.dataframe(df)  # Show as an interactive table

    # Plot the scores
    qualities = list(aggregated_quality_scores.keys())
    scores = list(aggregated_quality_scores.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(qualities, scores, color='skyblue')
    ax.set_xlabel('Score')
    ax.set_title('Aggregated Quality Scores for the Document')
    plt.tight_layout()
    
    # Display the plot in Streamlit
    st.pyplot(fig)

# Streamlit UI





# Function to get planetary positions (same as your original code)
def get_planetary_positions(birthdate, latitude, longitude):
    observer = ephem.Observer()
    observer.date = birthdate
    observer.lat, observer.lon = str(latitude), str(longitude)

    planets = {
        'Sun': ephem.Sun(),
        'Moon': ephem.Moon(),
        'Mercury': ephem.Mercury(),
        'Venus': ephem.Venus(),
        'Mars': ephem.Mars(),
        'Jupiter': ephem.Jupiter(),
        'Saturn': ephem.Saturn()
    }

    planetary_positions = {}
    for planet, body in planets.items():
        body.compute(observer)
        degrees = int((body.ra * 180 / ephem.pi) % 360)
        adjusted_degrees = (degrees - 28) % 360
        rashi, degrees_in_rashi = divmod(adjusted_degrees, 30)
        rashi = rashi + 1
        planetary_positions[planet] = {'Rashi': int(rashi), 'Degrees': int(degrees_in_rashi)}

    # Calculate Rahu and Ketu positions (same logic as above)
    moon = ephem.Moon(observer)
    sun = ephem.Sun(observer)
    moon.compute(observer)
    sun.compute(observer)
    moon_ecl = ephem.Ecliptic(moon)
    sun_ecl = ephem.Ecliptic(sun)
    lunar_longitude = moon_ecl.lon * 180 / ephem.pi
    solar_longitude = sun_ecl.lon * 180 / ephem.pi
    rahu_position = (lunar_longitude - solar_longitude) % 360
    ketu_position = (rahu_position + 180) % 360
    rahu_rashi, rahu_degrees = divmod(rahu_position, 30)
    ketu_rashi, ketu_degrees = divmod(ketu_position, 30)
    planetary_positions['Rahu'] = {'Rashi': int(rahu_rashi) + 1, 'Degrees': int(rahu_degrees)}
    planetary_positions['Ketu'] = {'Rashi': int(ketu_rashi) + 1, 'Degrees': int(ketu_degrees)}

    # Calculate Ascendant (same as above)
    observer.pressure = 0  # Disable atmospheric refraction
    lst = observer.sidereal_time()  # Local Sidereal Time
    ascendant_degrees = int((lst * 180 / ephem.pi) % 360)
    ascendant_adjusted_degrees = ((ascendant_degrees) % 360) - 18
    ascendant_rashi, ascendant_degrees_in_rashi = divmod(ascendant_adjusted_degrees, 30)
    ascendant_rashi = ascendant_rashi + 1
    if ascendant_rashi == 0:
        ascendant_rashi = 12

    planetary_positions['Ascendant'] = {'Rashi': int(ascendant_rashi), 'Degrees': int(ascendant_degrees_in_rashi)}

     # Calculate house placements
    for planet in planetary_positions:
        if planet != 'Ascendant':
            planet_rashi = planetary_positions[planet]['Rashi']
            house_number = ((planet_rashi - ascendant_rashi + 12) % 12)+ 1
            if house_number == 0:
                house_number = 12
            planetary_positions[planet]['House'] = house_number

    planetary_positions['Ascendant']['House'] = 1  # Ascendant is always in the 1st house
    




    def Shadbala(planet, planet_pos, planet_rashi, planetarypos):
        def Digbala(planet, planet_pos):
            def checkbala(planet, planet_pos):
                if planet_pos == 1:
                    return 7
                elif planet_pos in [2, 12]:
                    return 5
                elif planet_pos in [3, 11]:
                    return 4
                elif planet_pos in [4, 10]:
                    return 3
                elif planet_pos in [5, 9]:
                    return 2
                elif planet_pos in [6, 8]:
                    return 1
                elif planet_pos == 7:
                    return 0

            if planet in ["Jupiter", "Mercury"]:
                return checkbala(planet, planet_pos)
            elif planet == "Saturn":
                return checkbala(planet, (planet_pos + 6) % 12)
            elif planet in ["Mars", "Sun"]:
                return checkbala(planet, (planet_pos + 3) % 12)
            elif planet in ["Venus", "Moon"]:
                return checkbala(planet, (planet_pos + 9) % 12)
            else:
                return 4

        # Calculate Rashibala
        def Rashibala(planet, planet_rashi):
            rashibala_scores = {
                'Sun': {'exalted': 1, 'own': 5, 'debilitated': 7},
                'Moon': {'exalted': 2, 'own': 4, 'debilitated': 8},
                'Mars': {'exalted': 10, 'own': [1, 8], 'debilitated': 4},
                'Mercury': {'exalted': 6, 'own': [3, 6], 'debilitated': 12},
                'Jupiter': {'exalted': 4, 'own': [9, 12], 'debilitated': 10},
                'Venus': {'exalted': 12, 'own': [2, 7], 'debilitated': 6},
                'Saturn': {'exalted': 7, 'own': [10, 11], 'debilitated': 1}
            }

            if planet in rashibala_scores:
                rashi_info = rashibala_scores[planet]
                if planet_rashi == rashi_info['exalted']:
                    return 7
                elif planet_rashi in (rashi_info['own'] if isinstance(rashi_info['own'], list) else [rashi_info['own']]):
                    return 6
                elif planet_rashi == rashi_info['debilitated']:
                    return 1
                else:
                    return 5
            return 4

        # Calculate Bhavbala
        def Bhavbala(planet,planet_pos):
            if planet_pos in [1, 4, 7, 10]:
                return 7
            elif planet_pos in [2, 5, 9, 11]:
                return 5
            elif planet_pos in [3, 6, 8, 12]:
                return 2
            else:
                return 0

        # Calculate Sthanbala
        def Sthanbala(planet, planet_pos):
            if planet in ["Sun", "Moon", "Mercury", "Venus", "Jupiter"]:
                if planet_pos in [1, 4, 7, 10, 11]:
                    return 7
                elif planet_pos in [2, 6, 8, 12]:
                    return 4
                else:
                    return 5
            elif planet in ["Saturn", "Mars"]:
                if planet_pos in [1, 4, 7, 10]:
                    return 7
                elif planet_pos in [2, 6, 8, 12, 11]:
                    return 6
                else:
                    return 3
            return 4

        # Calculate Drigbala
        def get_aspects(planet, planet_pos):
            aspects = []
            aspscore = []
            if planet in ['Sun', 'Moon', 'Mercury', 'Venus']:
                aspects.append((planet_pos + 5) % 12 + 1)
                aspscore.append(1)
            elif planet == 'Mars':
                aspects.extend([(planet_pos + 2) % 12 + 1, (planet_pos + 5) % 12 + 1, (planet_pos + 6) % 12 + 1])
                aspscore.extend([0, 0, 0])
            elif planet == 'Jupiter':
                aspects.extend([(planet_pos + 3) % 12 + 1, (planet_pos + 5) % 12 + 1, (planet_pos + 7) % 12 + 1])
                aspscore.extend([1, 1, 1])
            elif planet == 'Saturn':
                aspects.extend([(planet_pos + 1) % 12 + 1, (planet_pos + 5) % 12 + 1, (planet_pos + 8) % 12 + 1])
                aspscore.extend([0, 0, 0])
            elif planet in ['Rahu', 'Ketu']:
                aspects.extend([(planet_pos + 3) % 12 + 1, (planet_pos + 5) % 12 + 1, (planet_pos + 7) % 12 + 1])
                aspscore.extend([0, 0, 0])
            return aspects, aspscore

        def calculate_drigbala(dataset, target_planet):
            target_position = int(dataset[target_planet])
            drigbala_score = 0
            aspects = []
            aspscore = []
            for planet, position in dataset.items():
                aspect_positions, aspect_scores = get_aspects(planet, position)
                aspects.extend(aspect_positions)
                aspscore.extend(aspect_scores)
            for aspect, score in zip(aspects, aspscore):
                if aspect == target_position and score == 1:
                    drigbala_score += 1
            return drigbala_score

        # Calculate Naisargikbala
        def Naisargikbala(planet):
            planet_scores = {
                "Sun": 7, "Moon": 6, "Venus": 5, "Jupiter": 4, 
                "Mercury": 3, "Mars": 2, "Saturn": 1
            }
            return planet_scores.get(planet, 0)

        # Calculate Shadbala
        Digbal = Digbala(planet,planet_pos)
        Rashibal = Rashibala(planet,planet_rashi)
        Bhavbal = Bhavbala(planet,planet_pos)
        Sthanbal = Sthanbala(planet,planet_pos)
        Drigbal = calculate_drigbala(planetarypos,planet)
        Naisargikbal = Naisargikbala(planet)
        if Digbal is None: Digbal = 0
        if Rashibal is None: Rashibal = 0
        if Bhavbal is None: Bhavbal = 0
        if Sthanbal is None: Sthanbal = 0
        if Drigbal is None: Drigbal = 0
        if Naisargikbal is None: Naisargikbal = 0
        planetbal = Digbal + Rashibal + Bhavbal + Sthanbal + Drigbal + Naisargikbal

        return planetbal
    
    # Get the 10th house lord
    def get_tenth_house_lord(ascendant_rashi):
        rashi_to_planet = {1: 'Mars', 2: 'Venus', 3: 'Mercury', 4: 'Moon', 
                        5: 'Sun', 6: 'Mercury', 7: 'Venus', 8: 'Mars',
                        9: 'Jupiter', 10: 'Saturn', 11: 'Saturn', 12: 'Jupiter'}
        tenth_house_rashi = (ascendant_rashi + 9) % 12
        return rashi_to_planet[tenth_house_rashi] if tenth_house_rashi != 0 else rashi_to_planet[12]

    # Get Amatyakaraka
    def get_amatyakaraka(planetarydegrees):
    # Exclude Rahu and Ketu from the calculation
        filtered_degrees = {planet: degrees for planet, degrees in planetarydegrees.items() if planet not in ['Rahu', 'Ketu']}
    
    # Get unique degrees in descending order
        unique_degrees = sorted(set(filtered_degrees.values()), reverse=True)
        
        # Check if there are at least two unique degrees
        if len(unique_degrees) < 2:
            return None
        
        # Find the second-highest degree
        second_highest_degree = unique_degrees[1]
        
        # Find the planet with the second-highest degree
        for planet, degrees in filtered_degrees.items():
            if degrees == second_highest_degree:
                return planet

    # Calculate planet balance
    def planetbal(planet, planetarypos, planetaryrashi):
        planet_pos = planetarypos[planet]
        planet_rashi = planetaryrashi[planet]
        return Shadbala(planet, planet_pos, planet_rashi, planetarypos)

    ascendant_rashi = planetary_positions['Ascendant']['Rashi']
    sun_house = planetary_positions['Sun']['House']
    moon_house = planetary_positions['Moon']['House']
    mercury_house = planetary_positions['Mercury']['House']
    venus_house = planetary_positions['Venus']['House']
    mars_house = planetary_positions['Mars']['House']
    saturn_house = planetary_positions['Saturn']['House']
    jupiter_house = planetary_positions['Jupiter']['House']
    rahu_house = planetary_positions['Rahu']['House']
    ketu_house = planetary_positions['Ketu']['House']

    sun_degrees = planetary_positions['Sun']['Degrees']
    moon_degrees = planetary_positions['Moon']['Degrees']
    mercury_degrees = planetary_positions['Mercury']['Degrees']
    venus_degrees = planetary_positions['Venus']['Degrees']
    mars_degrees = planetary_positions['Mars']['Degrees']
    jupiter_degrees = planetary_positions['Jupiter']['Degrees']
    saturn_degrees = planetary_positions['Saturn']['Degrees']
    rahu_degrees = planetary_positions['Rahu']['Degrees']
    ketu_degrees = planetary_positions['Ketu']['Degrees']

    sun_rashi = planetary_positions['Sun']['Rashi']
    moon_rashi = planetary_positions['Moon']['Rashi']
    mercury_rashi = planetary_positions['Mercury']['Rashi']
    venus_rashi = planetary_positions['Venus']['Rashi']
    mars_rashi = planetary_positions['Mars']['Rashi']
    saturn_rashi = planetary_positions['Saturn']['Rashi']
    jupiter_rashi = planetary_positions['Jupiter']['Rashi']
    rahu_rashi = planetary_positions['Rahu']['Rashi']
    ketu_rashi = planetary_positions['Ketu']['Rashi']

    planetarypos = {
        'Sun': sun_house,
        'Moon': moon_house,
        'Mars': mars_house,
        'Mercury': mercury_house,
        'Jupiter': jupiter_house,
        'Venus': venus_house,
        'Saturn': saturn_house,
        'Rahu': rahu_house,
        'Ketu': ketu_house
    }
    planetaryrashi = {
        'Sun': sun_rashi,
        'Moon': moon_rashi,
        'Mars': mars_rashi,
        'Mercury': mercury_rashi,
        'Jupiter': jupiter_rashi,
        'Venus': venus_rashi,
        'Saturn': saturn_rashi,
        'Rahu': rahu_rashi,
        'Ketu': ketu_rashi
    }

    planetarydegrees = {
        'Sun': sun_degrees,
        'Moon': moon_degrees,
        'Mars': mars_degrees,
        'Mercury': mercury_degrees,
        'Jupiter': jupiter_degrees,
        'Venus': venus_degrees,
        'Saturn': saturn_degrees,
        'Rahu': rahu_degrees,
        'Ketu': ketu_degrees
    }
   
    Ascendant = planetary_positions['Ascendant']['Rashi'] 
    
    Tenthlord = get_tenth_house_lord(Ascendant)
    Amatyakarak = get_amatyakaraka(planetarydegrees)
    Tenthlordbal = planetbal(Tenthlord, planetarypos ,planetaryrashi)
    Amatyakarakbal = planetbal(Amatyakarak, planetarypos,planetaryrashi)

    Sunbal = planetbal("Sun",planetarypos,planetaryrashi)
    Moonbal = planetbal("Moon",planetarypos,planetaryrashi)
    Mercurybal = planetbal("Mercury",planetarypos,planetaryrashi)
    Venusbal = planetbal("Venus",planetarypos,planetaryrashi)
    Marsbal = planetbal("Mars",planetarypos,planetaryrashi)
    Jupiterbal = planetbal("Jupiter",planetarypos,planetaryrashi)
    Saturnbal = planetbal("Saturn",planetarypos,planetaryrashi)
    Rahubal = planetbal("Rahu",planetarypos,planetaryrashi)
    Ketubal = planetbal("Ketu",planetarypos,planetaryrashi)
    planetary_positions['Tenthlord'] = Tenthlord
    planetary_positions['Amatyakarak'] = Amatyakarak
    planetary_positions['Tenthlordbal'] = Tenthlordbal
    planetary_positions['Amatyakarakbal'] = Amatyakarakbal
    planetary_positions['Sun_Bal'] = Sunbal
    planetary_positions['Moon_Bal']= Moonbal
    planetary_positions['Mercury_Bal']= Mercurybal
    planetary_positions['Venus_Bal']= Venusbal
    planetary_positions['Mars_Bal'] = Marsbal
    planetary_positions['Jupiter_Bal']= Jupiterbal
    planetary_positions['Saturn_Bal']= Saturnbal
    planetary_positions['Rahu_Bal']= Rahubal
    planetary_positions['Ketu_Bal']= Ketubal

    return planetary_positions


    




# Function to get latitude and longitude using Geopy
def get_lat_lon(city_name):
    geolocator = Nominatim(user_agent="city_locator")
    location = geolocator.geocode(city_name)
    if location:
        return location.latitude, location.longitude
    else:
        return 28.70405920, 77.10249020  # Default to Delhi coordinates if city not found















def load_data():
    # Load training data and model
    train_df = pd.read_csv('Training_data.csv')
    test_df = pd.read_csv('Test_data.csv')
    sector_jobs_df = pd.read_csv('sector_jobs_dataset.csv')
    model = load_model('DAIVA_Class_ANN.h5')
    return train_df, test_df, sector_jobs_df, model

# Data Preprocessing
def preprocess_data(df, is_train=True):
    # Drop unnecessary columns
    X = df.drop(columns=['id', 'name', 'occupation']) if is_train else df.drop(columns=['id', 'name'])

    # Define preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['Sun_degrees', 'Moon_degrees', 'Mercury_degrees', 'Venus_degrees', 'Mars_degrees', 'Jupiter_degrees', 
                                    'Saturn_degrees', 'Rahu_degrees', 'Ketu_degrees', 'Ascendant_degrees', 'Tenthlordbal', 'Amatyakarakbal']),
            ('cat', OneHotEncoder(), ['gender', 'Sun_rashi', 'Moon_rashi', 'Mercury_rashi', 'Venus_rashi', 'Mars_rashi', 
                                    'Jupiter_rashi', 'Saturn_rashi', 'Rahu_rashi', 'Ketu_rashi', 'Ascendant_rashi', 'Tenthlord', 'Amatyakarak'])
        ])

    # Fit and transform training data
    return preprocessor.fit_transform(X) if is_train else preprocessor.transform(X)

# Predict Occupation
def predict_occupation(model, X_new):
    predictions = model.predict(X_new)
    predicted_labels = np.argmax(predictions, axis=1)
    occupation_mapping = {
        0: 'Science', 1: 'Technology', 2: 'Service', 3: 'Arts', 
        4: 'Medical', 5: 'Government', 6: 'Teaching', 7: 'Commerce', 8: 'Agriculture'
    }
    return [occupation_mapping[label] for label in predicted_labels]

# Calculate Qualities Based on Astrology
def calculate_qualities(row, sector_qualities):
    # Define qualities and scoring rules
    qualities = {
        'Analytical': 1, 'Practical': 1, 'Creative': 1, 'Leadership': 1, 'Hard': 1, 'Smart': 1,
        'Technical': 1, 'Caring': 1, 'Communication': 1, 'Persuasive': 1, 'Integrity': 1, 'Imagination': 1,
        'Risk': 1, 'Spontaneous': 1, 'Determination': 1, 'Patience': 1, 'Knowledge': 1, 'Wisdom': 1
    }
    # Add specific planet-based scores, friendly/exalted rashis, and 10th house lord effects here...
    
    # Update qualities with sector-based quality scores
    predicted_sector = row['Predicted_Occupation']
    if predicted_sector in sector_qualities:
        for quality, score in sector_qualities[predicted_sector].items():
            qualities[quality] += score
    return qualities

# Final Job Recommendation with KNN
# Define the qualities and their scoring rules



def load_and_preprocess_data():
    # Load training data
    train_df = pd.read_csv('Training_data.csv')
    X_train = train_df.drop(columns=['id', 'name', 'occupation'])
    y_train = train_df['occupation']

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['Sun_degrees', 'Moon_degrees', 'Mercury_degrees', 'Venus_degrees', 'Mars_degrees', 
                                    'Jupiter_degrees', 'Saturn_degrees', 'Rahu_degrees', 'Ketu_degrees', 'Ascendant_degrees', 
                                    'Tenthlordbal', 'Amatyakarakbal']),
            ('cat', OneHotEncoder(), ['gender', 'Sun_rashi', 'Moon_rashi', 'Mercury_rashi', 'Venus_rashi', 'Mars_rashi', 
                                    'Jupiter_rashi', 'Saturn_rashi', 'Rahu_rashi', 'Ketu_rashi', 'Ascendant_rashi', 
                                    'Tenthlord', 'Amatyakarak'])
        ])

    # Fit preprocessor on training data
    X_train = preprocessor.fit_transform(X_train)

    # Convert y_train to one-hot encoding
    y_train = pd.get_dummies(y_train).values

    return preprocessor, X_train, y_train












def get_zodiac_sign(number):
    zodiac_signs = {
        1: "Aries",
        2: "Taurus",
        3: "Gemini",
        4: "Cancer",
        5: "Leo",
        6: "Virgo",
        7: "Libra",
        8: "Scorpio",
        9: "Sagittarius",
        10: "Capricorn",
        11: "Aquarius",
        12: "Pisces"
    }
    return zodiac_signs.get(number, "Invalid number. Please enter a number between 1 and 12.")












































# Streamlit interface
st.title("DAIVA")

# Input fields
name = st.text_input("Name")
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
dob = st.date_input("Date of Birth")
tob = st.time_input("Time of Birth")
place_of_birth = st.text_input("Place of Birth")
mail = st.text_input("E-Mail")
id_number = 0
st.header("Step 1: Astrological data")
    
        # Load pre-existing data and model
preprocessor, X_train, y_train = load_and_preprocess_data()

# Load model
model = load_model('DAIVA_Class_ANN.h5')
birth_datetime = datetime.combine(dob, tob)
 # Get latitude and longitude for the place of birth
lat, lon = get_lat_lon(place_of_birth)

positions = get_planetary_positions(birth_datetime, lat, lon)
result = st.text(positions)
record = {}
if 'id' and 'gender' not in record:
    record['id'] = id_number
    record['gender'] = gender  # or specify a default value
# Flattening the nested structure for each planet
for planet, attributes in positions.items():
    if isinstance(attributes, dict):
        for attr, value in attributes.items():
            record[f"{planet}_{attr.lower()}"] = value
    else:
        # Direct assignments for non-dict elements like Tenthlord and Amatykarak
        record[planet] = attributes

result = st.text(record)
record = {key: [value] if not isinstance(value, list) else value for key, value in record.items()}
# Load new data for prediction
new_data_df = pd.DataFrame(record)

# Drop columns that are not needed for the prediction
X_new = new_data_df

# Transform the new data using the already fitted preprocessor
X_new = preprocessor.transform(X_new)

# Make predictions
predictions = model.predict(X_new)
predicted_labels = np.argmax(predictions, axis=1)

# Map predicted labels to occupation sectors
occupation_mapping = {0: 'Science', 1: 'Technology', 2: 'Service', 3: 'Arts', 
                        4: 'Medical', 5: 'Government', 6: 'Teaching', 7: 'Commerce', 8: 'Agriculture'}
predicted_occupations = [occupation_mapping[label] for label in predicted_labels]

# Add predictions to new data
new_data_df['Predicted_Occupation'] = predicted_occupations





#
#
#
#
#
#
#
kundalidf = pd.DataFrame(new_data_df)
Sunplacement = pd.read_csv('sunplacement.csv')
Moonplacement = pd.read_csv('moonplacement.csv')
Mercuryplacement = pd.read_csv('mercuryplacement.csv')
Venusplacement = pd.read_csv('venusplacement.csv')
Jupiterplacement = pd.read_csv('jupiterplacement.csv')
Marsplacement = pd.read_csv('marsplacement.csv')


n=0
Sun_house = kundalidf.iloc[n]['Sun_house']
Sun_rashi = kundalidf.iloc[n]['Sun_rashi']
Moon_rashi = kundalidf.iloc[n]['Moon_rashi']
Moon_house = kundalidf.iloc[n]['Moon_house']
Mercury_house = kundalidf.iloc[n]['Mercury_house']
Mercury_rashi = kundalidf.iloc[n]['Mercury_rashi']
Venus_house = kundalidf.iloc[n]['Venus_house']
Venus_rashi = kundalidf.iloc[n]['Venus_rashi']
Jupiter_house = kundalidf.iloc[n]['Jupiter_house']
Jupiter_rashi = kundalidf.iloc[n]['Jupiter_rashi']
Mars_house = kundalidf.iloc[n]['Mars_house']
Mars_rashi = kundalidf.iloc[n]['Mars_rashi']

Sun_house = str(Sun_house)+"th"+" House"
Moon_house = str(Moon_house)+"th"+" House"
Mercury_house = str(Mercury_house)+"th"+" House"
Venus_house = str(Venus_house)+"th"+" House"
Jupiter_house = str(Jupiter_house)+"th"+" House"
Mars_house = str(Mars_house)+"th"+" House"

Sun_sign = get_zodiac_sign(Sun_rashi)
Moon_sign = get_zodiac_sign(Moon_rashi)
Mercury_sign = get_zodiac_sign(Mercury_rashi)
Venus_sign = get_zodiac_sign(Venus_rashi)
Jupiter_sign = get_zodiac_sign(Jupiter_rashi)
Mars_sign = get_zodiac_sign(Mars_rashi)

sundesc = Sunplacement.loc[Sunplacement['House'] == Sun_house, Sun_sign].values[0]
moondesc = Moonplacement.loc[Moonplacement['House'] == Moon_house, Moon_sign].values[0]
mercurydesc = Mercuryplacement.loc[Mercuryplacement['House'] == Mercury_house, Mercury_sign].values[0]
venusdesc = Venusplacement.loc[Venusplacement['House'] == Venus_house, Venus_sign].values[0]
jupiterdesc = Jupiterplacement.loc[Jupiterplacement['House'] == Jupiter_house, Jupiter_sign].values[0]
marsdesc = Marsplacement.loc[Marsplacement['House'] == Mars_house, Mars_sign].values[0]



#Output Section
Generalinformation = sundesc+moondesc+mercurydesc+venusdesc+jupiterdesc+marsdesc



















new_data_df['General Information'] = Generalinformation


# Display results
st.write("Predicted Occupations:")
st.dataframe(new_data_df)

import pandas as pd

# Define the qualities and their scoring rules
qualities = [
    'Analytical', 'Practical', 'Creative', 'Leadership', 'Hard',
    'Smart', 'Technical', 'Caring', 'Communication', 'Persuasive',
    'Integrity', 'Imagination', 'Risk', 'Spontaneous', 'Determination',
    'Patience', 'Knowledge', 'Wisdom'
]

# Define which qualities are associated with which planets
planet_qualities = {
    'Moon': ['Caring', 'Imagination', 'Patience'],
    'Sun': ['Leadership', 'Hard', 'Determination'],
    'Mars': ['Risk', 'Spontaneous', 'Hard'],
    'Mercury': ['Analytical', 'Smart', 'Communication'],
    'Jupiter': ['Knowledge', 'Wisdom', 'Caring'],
    'Venus': ['Creative', 'Practical', 'Communication'],
    'Saturn': ['Hard', 'Determination', 'Patience'],
    'Rahu': ['Risk', 'Creative', 'Leadership'],
    'Ketu': ['Imagination', 'Wisdom', 'Spontaneous']
}

# Define friendly and exalted signs with their numerical representations
rashi_mapping = {
    1: 'Aries', 2: 'Taurus', 3: 'Gemini', 4: 'Cancer', 5: 'Leo',
    6: 'Virgo', 7: 'Libra', 8: 'Scorpio', 9: 'Sagittarius', 10: 'Capricorn',
    11: 'Aquarius', 12: 'Pisces'
}

friendly_signs = {
    'Sun': [4, 5, 1],  # Cancer, Leo, Aries
    'Moon': [2, 4, 6],  # Taurus, Cancer, Virgo
    'Mars': [1, 8, 10],  # Aries, Scorpio, Capricorn
    'Mercury': [3, 6],  # Gemini, Virgo
    'Jupiter': [9, 12, 4],  # Sagittarius, Pisces, Cancer
    'Venus': [2, 7, 12],  # Taurus, Libra, Pisces
    'Saturn': [10, 11, 7]  # Capricorn, Aquarius, Libra
}

exalted_signs = {
    'Sun': 1,  # Aries
    'Moon': 2,  # Taurus
    'Mars': 10,  # Capricorn
    'Mercury': 6,  # Virgo
    'Jupiter': 4,  # Cancer
    'Venus': 12,  # Pisces
    'Saturn': 7  # Libra
}

# Define sector-based quality scores
sector_qualities = {
    'Science': {'Smart': 4, 'Technical': 3, 'Analytical': 2},
    'Technology': {'Technical': 4, 'Analytical': 3, 'Smart': 2},
    'Service': {'Practical': 4, 'Communication': 3, 'Patience': 2,'Hard':2,'Determination':1},
    'Arts': {'Creative': 4, 'Imagination': 3, 'Communication': 2 ,'Spontaneous':1},
    'Medical': {'Caring': 4, 'Knowledge': 3, 'Practical': 2},
    'Government': {'Leadership': 4, 'Determination': 3, 'Integrity': 2},
    'Teaching': {'Communication': 4, 'Patience': 2,'Caring': 2,'Knowledge':2},
    'Commerce': {'Analytical': 4, 'Smart': 3, 'Practical': 2,'Persuasive':2,'Risk':1},
    'Agriculture': {'Hard': 4, 'Wisdom': 3, 'Knowledge': 2,'Risk':1}
}

def get_scores(row):
    scores = {quality: 1 for quality in qualities}  # Default score for all qualities

    # Check all _house columns for the 10th house
    for column in df.columns:
        if column.endswith('_house'):
            planet_name = column.split('_')[0]  # Extract planet name from column name
            if planet_name in ['Sun', 'Moon', 'Mars', 'Mercury', 'Jupiter', 'Venus', 'Saturn', 'Rahu', 'Ketu']:
                if row[column] == 10:  # Check if the planet is in the 10th house
                    if planet_name in planet_qualities:
                        for quality in planet_qualities[planet_name]:
                            scores[quality] += 2  # 3 points total, 1 is default
                    break

    # Check planet rashis for friendly and exalted signs
    for planet in ['Sun', 'Moon', 'Mars', 'Mercury', 'Jupiter', 'Venus', 'Saturn']:
        planet_rashi_num = row[f'{planet}_rashi']
        if planet_rashi_num in friendly_signs.get(planet, []):
            # Award 2 extra points for friendly sign
            for quality in planet_qualities.get(planet, []):
                scores[quality] += 2
        if planet_rashi_num == exalted_signs.get(planet):
            # Award 2 extra points for exalted sign
            for quality in planet_qualities.get(planet, []):
                scores[quality] += 2

    # Assign scores based on the 10th house lord
    tenthlord = row['Tenthlord']
    if tenthlord in planet_qualities:
        for quality in planet_qualities[tenthlord]:
            scores[quality] += 2  # 2 points for qualities associated with the Tenthlord

    # Add sector-based quality scores
    predicted_sector = row['Predicted_Occupation']
    if predicted_sector in sector_qualities:
        for quality, score in sector_qualities[predicted_sector].items():
            scores[quality] += score

    return scores


df = new_data_df

# Create a new DataFrame for the results
results = []

for _, row in df.iterrows():
    scores = get_scores(row)
    result = {'id': row['id'], 'Predicted_Occupation': row['Predicted_Occupation']}
    result.update(scores)
    results.append(result)

results_df = pd.DataFrame(results)









import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Load your datasets
train_df = pd.read_csv('sector_jobs_dataset.csv')
test_df = results_df

# List of qualities/features
qualities = [
    'Analytical', 'Practical', 'Creative', 'Leadership', 'Hard',
    'Smart', 'Technical', 'Caring', 'Communication', 'Persuasive',
    'Integrity', 'Imagination', 'Risk', 'Spontaneous', 'Determination',
    'Patience', 'Knowledge', 'Wisdom'
]

# Ensure all qualities are in the test dataset
for quality in qualities:
    if quality not in test_df.columns:
        test_df[quality] = 0  # or some other default value

# Prepare training data
X_train = train_df[qualities]
y_train = train_df['Job']  # Assuming 'Job' is your target variable

# Prepare testing data
X_test = test_df[qualities]

# Train the model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Get the top 5 nearest neighbors
distances, indices = knn.kneighbors(X_test, n_neighbors=5)

# Retrieve top 5 predicted jobs for each test sample
top_5_jobs = [[y_train.iloc[i] for i in index_list] for index_list in indices]

# Convert to DataFrame
top_5_jobs_df = pd.DataFrame(top_5_jobs, columns=['Job_1', 'Job_2', 'Job_3', 'Job_4', 'Job_5'])

# Add predictions to test_df
jobs_df = pd.concat([test_df, top_5_jobs_df], axis=1)

# Save the results
st.write(jobs_df)

















vimshottara=(7,20,6,10,7,18,16,19,17)
Vimshot=("Ketu","Venus","Sun","Moon","Mars","Rahu","Jupiter","Saturn","Mercury")
Rashilords = ("Jupiter","Mars","Venus","Mercury","Moon","Sun","Mercury","Venus","Mars","Jupiter","Saturn","Saturn")
Rashislist = ("Aries","Taurus","Gemini","Cancer","Leo","Virgo","Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces")
nakshatras = (
    "Ashwini",
    "Bharani",
    "Krittika",
    "Rohini",
    "Mrigashira",
    "Ardra",
    "Punarvasu",
    "Pushya",
    "Ashlesha",
    "Magha",
    "Purva Phalguni",
    "Uttara Phalguni",
    "Hasta",
    "Chitra",
    "Swati",
    "Vishakha",
    "Anuradha",
    "Jyeshtha",
    "Mula",
    "Purva Ashadha",
    "Uttara Ashadha",
    "Shravana",
    "Dhanishta",
    "Shatabhisha",
    "Purva Bhadrapada",
    "Uttara Bhadrapada",
    "Revati"
)
nakshatra_symbols = (
    "Horse's Head",
    "(Yoni) Female Reproductive Organ",
    "(Knife) or Spear",
    "(Cart) or Chariot",
    "(Deer's Head)",
    "(Teardrop) or Diamond",
    "(Bow and Quiver) of Arrows",
    "(Flower, Circle,) or Cow's Udder",
    "(Coiled Serpent)",
    "(Throne) or Palanquin",
    "(Front of a Couch)",
    "(Back Legs of a Couch)",
    "(Hand) or Fist",
    "(Pearl) or Bright Jewel",
    "(Young Plant Shoot,) Coral",
    "(Arch, Potter's Wheel)",
    "(Lotus Flower) or Triumphal Arch",
    "(Earring, Circular) Talisman",
    "(Roots, Umbrella)",
    "(Fan) or Winnowing Basket",
    "(Elephant's Tusk) or Planks of a Bed",
    "(Ear) or Three Footprints in an Ascending Path",
    "(Drum) or Flute",
    "(Empty Circle) or 1000 Flowers",
    "(Two-Faced Man) or Sword",
    "(Back Legs of a Funeral Cot) or Snake in the Water",
    "(Fish) or Drum"
)

nakshatra_to_varna = {
    'Ashwini': 'Brahmin',
    'Bharani': 'Brahmin',
    'Krittika': 'Brahmin',
    'Rohini': 'Brahmin',
    'Mrigashira': 'Brahmin',
    'Ardra': 'Brahmin',
    'Punarvasu': 'Brahmin',
    'Pushya': 'Brahmin',
    'Ashlesha': 'Kshatriya',
    'Magha': 'Kshatriya',
    'Purva Phalguni': 'Kshatriya',
    'Uttara Phalguni': 'Kshatriya',
    'Hasta': 'Vaishya',
    'Chitra': 'Vaishya',
    'Swati': 'Vaishya',
    'Vishakha': 'Vaishya',
    'Anuradha': 'Vaishya',
    'Jyeshta': 'Vaishya',
    'Mula': 'Vaishya',
    'Purva Ashadha': 'Shudra',
    'Uttara Ashadha': 'Shudra',
    'Shravana': 'Shudra',
    'Dhanishta': 'Shudra',
    'Shatabhisha': 'Shudra',
    'Purva Bhadrapada': 'Shudra',
    'Uttara Bhadrapada': 'Shudra',
    'Revati': 'Shudra'
}
nakshatra_to_vashya = {
    'Ashwini': 'Manav',
    'Bharani': 'Vanchar',
    'Krittika': 'Chatushpad',
    'Rohini': 'Chatushpad',
    'Mrigashira': 'Manav',
    'Ardra': 'Jalchar',
    'Punarvasu': 'Manav',
    'Pushya': 'Manav',
    'Ashlesha': 'Jalchar',
    'Magha': 'Keet',
    'Purva Phalguni': 'Vanchar',
    'Uttara Phalguni': 'Manav',
    'Hasta': 'Manav',
    'Chitra': 'Manav',
    'Swati': 'Manav',
    'Vishakha': 'Vanchar',
    'Anuradha': 'Manav',
    'Jyeshta': 'Keet',
    'Mula': 'Jalchar',
    'Purva Ashadha': 'Chatushpad',
    'Uttara Ashadha': 'Chatushpad',
    'Shravana': 'Manav',
    'Dhanishta': 'Vanchar',
    'Shatabhisha': 'Manav',
    'Purva Bhadrapada': 'Chatushpad',
    'Uttara Bhadrapada': 'Chatushpad',
    'Revati': 'Manav'
}

nakshatra_to_nadi = {
    'Ashwini': 'Adi',
    'Bharani': 'Adi',
    'Krittika': 'Adi',
    'Rohini': 'Adi',
    'Mrigashira': 'Adi',
    'Ardra': 'Adi',
    'Punarvasu': 'Adi',
    'Pushya': 'Adi',
    'Ashlesha': 'Adi',
    'Magha': 'Madhya',
    'Purva Phalguni': 'Madhya',
    'Uttara Phalguni': 'Madhya',
    'Hasta': 'Madhya',
    'Chitra': 'Madhya',
    'Swati': 'Madhya',
    'Vishakha': 'Madhya',
    'Anuradha': 'Madhya',
    'Jyeshta': 'Madhya',
    'Mula': 'Madhya',
    'Purva Ashadha': 'Antya',
    'Uttara Ashadha': 'Antya',
    'Shravana': 'Antya',
    'Dhanishta': 'Antya',
    'Shatabhisha': 'Antya',
    'Purva Bhadrapada': 'Antya',
    'Uttara Bhadrapada': 'Antya',
    'Revati': 'Antya'
}

#This is Rolling function
def roll(num):
  house = num % 12
  if (house==0):
    num =  num - 12
  else:
    num = house
  return num

def roll2(num):
  planet = num % 9
  if(planet == 0):
    num = num - 9
  else:
    num = planet
  return num


def Nakshatracalculator(nakshatranum):
  nakshatraname = nakshatras[nakshatranum - 1]
  print(nakshatraname)
  if nakshatraname == "Ashwini" or nakshatraname == "Magha" or nakshatraname == "Mula":
    Nlord = "Ketu"
  elif nakshatraname == "Bharani" or nakshatraname == "Purva Phalguni" or nakshatraname == "Purva Ashadha":
    Nlord = "Venus"
  elif nakshatraname == "Krittika" or nakshatraname == "Uttara Phalguni" or nakshatraname == "Uttara Ashadha":
    Nlord = "Sun"
  elif nakshatraname == "Rohini" or nakshatraname == "Hasta" or nakshatraname == "Shravana":
    Nlord = "Moon"
  elif nakshatraname == "Mrigashira" or nakshatraname == "Chitra" or nakshatraname == "Dhanishta":
    Nlord = "Mars"
  elif nakshatraname == "Ardra" or nakshatraname == "Swati" or nakshatraname == "Shatabhisha":
    Nlord = "Rahu"
  elif nakshatraname == "Punarvasu" or nakshatraname == "Vishakha" or nakshatraname == "Purva Bhadrapada":
    Nlord = "Jupiter"
  elif nakshatraname == "Pushya" or nakshatraname == "Anuradha" or nakshatraname == "Uttara Bhadrapada":
    Nlord = "Saturn"
  elif nakshatraname == "Ashlesha" or nakshatraname == "Jyeshtha" or nakshatraname == "Revati":
    Nlord = "Mercury"
  print("The Nakshatra lord is",Nlord)
  return Nlord,nakshatraname



def Vimshotcalculator(Wanted_Dasha):
    Counter = 0
    for i in Vimshot:
        Counter = Counter + 1
        print(f"Checking: {i} (Counter: {Counter})")  # Debug output
        if i == Wanted_Dasha:
            print(f"Match found for {Wanted_Dasha} at position {Counter}")
            break
        else:
            print(f"Not found for {Wanted_Dasha}")
    return Counter




def Vimshottaracalculator(Counter):
    # Return the years based on the Counter position from vimshottara
    Years = 0
    Counter1 = 0
    for i in vimshottara:
        Counter1 = Counter1 + 1
        if Counter1 == Counter:
            Years = int(i)
            break  # Exit the loop once the correct period is found
    return Years


import pandas as pd
from datetime import datetime, timedelta

def Jyotantar(Rashinamedb, moondeg, dob):
    # Parse the date of birth input
    dob = datetime.strptime(dob, '%Y-%m-%d')  # Assuming DOB format 'YYYY-MM-DD'

    moondeg, moonmin = map(int, moondeg.split("."))
    Rashcounter = 0
    # Assume Rashislist is defined elsewhere
    for Rashiname in Rashislist:
        Rashcounter = Rashcounter + 1
        if Rashiname == Rashinamedb:
            break
    Degrees = moondeg
    Minute = moonmin
    Rashinumber = Rashcounter - 1
    cras = Rashinumber * 30
    cdega = cras + Degrees
    cdeg = cdega * 60
    cmin = cdeg + Minute
    print(cmin)
    rashi = float(cmin / 800)
    nakshatranum = int(cmin / 800 + 1)
    print("The Nakshatra number is", nakshatranum)
    rashika = str(rashi)
    rashi, deg = map(int, rashika.split("."))
    diff = cmin - (800 * rashi)
    print(diff)
    if diff <= 200:
        pada = 1
    elif 200 < diff <= 400:
        pada = 2
    elif 400 < diff <= 600:
        pada = 3
    elif 600 < diff <= 800:
        pada = 4
    print("The Nakshatra pada is", pada)
    bhogya = 800 - diff
    print("The Bhogya remaining is", bhogya)
    Nlord, nakshatraname = Nakshatracalculator(nakshatranum)
    Counter = Vimshotcalculator(Nlord)
    Kalah = Vimshottaracalculator(Counter)
    print(Kalah)
    Bhogyakalah = Kalah / 800 * bhogya
    print(Bhogyakalah)
    Salmah = str(Bhogyakalah)
    Sal, mah = map(int, Salmah.split("."))
    if Sal == 0:
        Mah = 0
    else:
        Mah = int(Bhogyakalah % Sal * 12)
    print(
        "The Person born in",
        nakshatraname,
        "Nakshatra of planet",
        Nlord,
        "Remaining bhogyakalh is",
        Sal,
        "years and",
        Mah,
        "months that sum up the person will have ",
        Nlord,
        "Mahadasha First From",
    )

    # Arrays to store values
    dashalist = []
    kallist = []

    # Initialize counters to the correct position
    Counter2 = Counter - 1
    Counter3 = Counter - 1

    # For Vimshot calculation and populating dashalist and kallist
    for i in range(len(Vimshot)):  # Use range and len to loop through Vimshot list
        Counter4 = roll2(Counter3 + 1)  # Roll the counter for next planet
        if Counter4 == 0:  # Ensure the counter doesn't go out of bounds
            Counter4 = 9

        # Get the Vimshottara period for the current planet
        nextkal = Vimshottaracalculator(Counter4)

        # Get the next dasha using the roll2 function on Counter2
        nextdasha = Vimshot[roll2(Counter2)]

        # Store values in arrays
        dashalist.append(nextdasha)
        kallist.append(nextkal)

        print(f"{nextdasha} has a period of {nextkal} years")

        # Increment counters to move to the next values
        Counter2 += 1
        Counter3 += 1

    print(dashalist)
    print(kallist)

    # Create a DataFrame to store the Dasha information
    dasha_data = []
    firstdashatime = kallist[0]
    print(firstdashatime)
    firstdashatimeinmonth = firstdashatime * 12
    print(firstdashatimeinmonth)
    salmahtime = Sal * 12 + Mah
    timeperiodpast = firstdashatimeinmonth - salmahtime
    print(timeperiodpast)
    timeperiodpastinyears = timeperiodpast // 12
    timeperiodpastinmonths = timeperiodpast % 12
    print(timeperiodpastinyears)
    print(timeperiodpastinmonths)

    current_date = dob   # Use the provided Date of Birth as the start date
    current_day, current_month, current_year = current_date.day, current_date.month, current_date.year
    current_month = current_month -  timeperiodpastinmonths
    current_year = current_year - timeperiodpastinyears
    if current_month < 1:
        current_month += 12
        current_year -= 1
    current_date = datetime(current_year, current_month, current_day)
    print(current_date)
    # Loop through the dashalist and kallist to create start and end dates
    for dasha, period in zip(dashalist, kallist):
        start_date = current_date
        # Adding the period (in years) to the start date
        end_date = start_date + timedelta(days=period * 365)  # assuming the period is in years

        # Add data to the dasha_data list
        dasha_data.append([dasha, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), period])

        # Update the current date for the next Dasha to be the end date of the previous one
        current_date = end_date

    # Create a Pandas DataFrame
    df = pd.DataFrame(dasha_data, columns=["Dasha", "Start Date", "End Date", "Duration (Years)"])

    # Print the table

    return df

# Example Usage:
st.write(birth_datetime)
birthdate = str(birth_datetime).split(" ")[0]
st.write(Moon_rashi)

Moon_rashiname = Rashislist[int(Moon_rashi)-1]
st.write(Moon_rashiname)
df12 = Jyotantar(Moon_rashiname, str(str(new_data_df['Moon_degrees'].values[0]) + ".00"), birthdate)
print(df12)

st.dataframe(df12)

import pandas as pd
from datetime import datetime, timedelta

# Vimshottari Mahadasha sequence and their respective durations
vimshottari_sequence = {
    'Ketu': 7, 'Venus': 20, 'Sun': 6, 'Moon': 10, 'Mars': 7,
    'Rahu': 18, 'Jupiter': 16, 'Saturn': 19, 'Mercury': 17
}

def calculate_antardasha(mahadasha_planet, start_date, mahadasha_end_date):
    # Ensure the Mahadasha planet is the first in the sequence
    sequence = list(vimshottari_sequence.keys())
    idx = sequence.index(mahadasha_planet)
    antardasha_sequence = sequence[idx:] + sequence[:idx]  # Reordering

    # Calculate total duration of Mahadasha in days
    total_days = (mahadasha_end_date - start_date).days
    antardasha_list = []

    # Proportionally calculate Antardasha durations
    for antardasha_planet in antardasha_sequence:
        antar_duration_days = round((vimshottari_sequence[antardasha_planet] / sum(vimshottari_sequence.values())) * total_days)
        end_date = start_date + timedelta(days=antar_duration_days)
        antardasha_list.append({
            'Mahadasha': mahadasha_planet,
            'Antardasha': antardasha_planet,
            'Start_Date': start_date,
            'End_Date': end_date
        })
        start_date = end_date  # Move to next Antardasha

    return antardasha_list

# Load Mahadasha data
df12['Start Date'] = pd.to_datetime(df12['Start Date'])
df12['End Date'] = pd.to_datetime(df12['End Date'])

# Get the current date
current_date = datetime.today()

# Find the active Mahadasha
active_mahadasha = df12[(df12["Start Date"] <= current_date) & (df12["End Date"] >= current_date)].iloc[0]
mahadasha_planet = active_mahadasha["Dasha"]
mahadasha_start = active_mahadasha["Start Date"]
mahadasha_end = active_mahadasha["End Date"]

# Compute Antardasha periods based on Mahadasha duration
antardasha_data = calculate_antardasha(mahadasha_planet, mahadasha_start, mahadasha_end)

# Convert to DataFrame and print
antardasha_df = pd.DataFrame(antardasha_data)
print(antardasha_df)
st.dataframe(antardasha_df)




import pandas as pd

# Mahadasha-Antardasha dataset
mahadasha_data = antardasha_df
# Planetary strengths (Shadbal)
shadbal_data = {
    "Sun_Bal": new_data_df["Sun_Bal"].values[0],
    "Moon_Bal": new_data_df["Moon_Bal"].values[0],
    "Mercury_Bal": new_data_df["Mercury_Bal"].values[0],
    "Venus_Bal": new_data_df["Venus_Bal"].values[0],
    "Mars_Bal": new_data_df["Mars_Bal"].values[0],
    "Jupiter_Bal": new_data_df["Jupiter_Bal"].values[0],
    "Saturn_Bal": new_data_df["Saturn_Bal"].values[0],
    "Rahu_Bal":new_data_df["Rahu_Bal"].values[0],
    "Ketu_Bal":new_data_df["Ketu_Bal"].values[0],
}

# Convert Mahadasha data to DataFrame
mahadasha_df = pd.DataFrame(mahadasha_data)

# Convert Shadbal data to DataFrame
shadbal_df = pd.DataFrame(list(shadbal_data.items()), columns=["Planet", "Shadbal"])
shadbal_df["Planet"] = shadbal_df["Planet"].str.replace("_Bal", "")  # Clean planet names

# Function to analyze the period
def analyze_period(row, shadbal_df):
    antardasha_planet = row["Antardasha"]
    shadbal_row = shadbal_df[shadbal_df["Planet"] == antardasha_planet]

    if not shadbal_row.empty:
        shadbal = shadbal_row["Shadbal"].values[0]

        # Analyze based on Shadbal strength
        if shadbal >= 30:
            strength = "Very Strong"
            recommendation = f"Excellent period for career growth. Focus on leadership roles and new opportunities."
        elif shadbal >= 25:
            strength = "Strong"
            recommendation = f"Good period for career advancement. Take calculated risks and network effectively."
        elif shadbal >= 20:
            strength = "Moderate"
            recommendation = f"Stable period. Focus on skill development and maintaining current responsibilities."
        else:
            strength = "Weak"
            recommendation = f"Challenging period. Avoid major decisions and focus on resolving existing issues."
    else:
        strength = "Unknown"
        recommendation = f"No Shadbal data available for {antardasha_planet}."

    return pd.Series([strength, recommendation])

# Apply the analysis to each row
mahadasha_df[["Strength", "Recommendation"]] = mahadasha_df.apply(analyze_period, axis=1, shadbal_df=shadbal_df)

# Print the results
print(mahadasha_df[["Mahadasha", "Antardasha", "Start_Date", "End_Date", "Strength", "Recommendation"]])

st.dataframe(mahadasha_df)


mahadasha_df = pd.DataFrame(mahadasha_df)

# Convert Start_Date and End_Date to datetime
mahadasha_df['Start_Date'] = pd.to_datetime(mahadasha_df['Start_Date'])
mahadasha_df['End_Date'] = pd.to_datetime(mahadasha_df['End_Date'])

# Define a color map for Strength
strength_colors = {
    'Weak': 'red',
    'Moderate': 'orange',
    'Strong': 'yellow',
    'Very Strong': 'green'
}

# Create the Gantt chart
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each period as a horizontal bar
for i, row in mahadasha_df.iterrows():
    ax.barh(
        y=row['Antardasha'],  # Use Antardasha as the y-axis label
        width=(row['End_Date'] - row['Start_Date']).days,  # Duration in days
        left=row['Start_Date'],  # Start date
        color=strength_colors[row['Strength']],  # Color based on Strength
        edgecolor='black'
    )

# Format the x-axis to show dates
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_major_locator(mdates.YearLocator())
plt.xticks(rotation=45)

# Add labels and title
ax.set_xlabel('Timeline')
ax.set_ylabel('Antardasha')
ax.set_title('Mahadasha and Antardasha Timeline')

# Add a legend for Strength
handles = [plt.Rectangle((0, 0), 1, 1, color=strength_colors[strength]) for strength in strength_colors]
labels = list(strength_colors.keys())
ax.legend(handles, labels, title='Strength')

# Show the plot in Streamlit
st.pyplot(fig)



























# Download predictions
st.download_button(
    label="Download Predictions as CSV",
    data=new_data_df.to_csv(index=False).encode('utf-8'),
    file_name='classified_data.csv',
    mime='text/csv'
)




qualities = [
    'Analytical', 'Practical', 'Creative', 'Leadership', 'Hard',
    'Smart', 'Technical', 'Caring', 'Communication', 'Persuasive',
    'Integrity', 'Imagination', 'Risk', 'Spontaneous', 'Determination',
    'Patience', 'Knowledge', 'Wisdom'
]

# Planet qualities mapping
planet_qualities = {
    'Moon': ['Caring', 'Imagination', 'Patience'],
    'Sun': ['Leadership', 'Hard', 'Determination'],
    'Mars': ['Risk', 'Spontaneous', 'Hard'],
    'Mercury': ['Analytical', 'Smart', 'Communication'],
    'Jupiter': ['Knowledge', 'Wisdom', 'Caring'],
    'Venus': ['Creative', 'Practical', 'Communication'],
    'Saturn': ['Hard', 'Determination', 'Patience'],
    'Rahu': ['Risk', 'Creative', 'Leadership'],
    'Ketu': ['Imagination', 'Wisdom', 'Spontaneous']
}

# Define friendly and exalted signs with their numerical representations
friendly_signs = {
    'Sun': [4, 5, 1],
    'Moon': [2, 4, 6],
    'Mars': [1, 8, 10],
    'Mercury': [3, 6],
    'Jupiter': [9, 12, 4],
    'Venus': [2, 7, 12],
    'Saturn': [10, 11, 7]
}

exalted_signs = {
    'Sun': 1,
    'Moon': 2,
    'Mars': 10,
    'Mercury': 6,
    'Jupiter': 4,
    'Venus': 12,
    'Saturn': 7
}

# Define sector-based quality scores
sector_qualities = {
    'Science': {'Smart': 4, 'Technical': 3, 'Analytical': 2},
    'Technology': {'Technical': 4, 'Analytical': 3, 'Smart': 2},
    'Service': {'Practical': 4, 'Communication': 3, 'Patience': 2, 'Hard': 2, 'Determination': 1},
    'Arts': {'Creative': 4, 'Imagination': 3, 'Communication': 2, 'Spontaneous': 1},
    'Medical': {'Caring': 4, 'Knowledge': 3, 'Practical': 2},
    'Government': {'Leadership': 4, 'Determination': 3, 'Integrity': 2},
    'Teaching': {'Communication': 4, 'Patience': 2, 'Caring': 2, 'Knowledge': 2},
    'Commerce': {'Analytical': 4, 'Smart': 3, 'Practical': 2, 'Persuasive': 2, 'Risk': 1},
    'Agriculture': {'Hard': 4, 'Wisdom': 3, 'Knowledge': 2, 'Risk': 1}
}

def get_scores(row):
    scores = {quality: 1 for quality in qualities}  # Default score for all qualities

    # Check all _house columns for the 10th house
    for column in df.columns:
        if column.endswith('_house'):
            planet_name = column.split('_')[0]  # Extract planet name from column name
            if planet_name in ['Sun', 'Moon', 'Mars', 'Mercury', 'Jupiter', 'Venus', 'Saturn', 'Rahu', 'Ketu']:
                if row[column] == 10:  # Check if the planet is in the 10th house
                    if planet_name in planet_qualities:
                        for quality in planet_qualities[planet_name]:
                            scores[quality] += 2  # 3 points total, 1 is default
                    break

    # Check planet rashis for friendly and exalted signs
    for planet in ['Sun', 'Moon', 'Mars', 'Mercury', 'Jupiter', 'Venus', 'Saturn']:
        planet_rashi_num = row[f'{planet}_rashi']
        if planet_rashi_num in friendly_signs.get(planet, []):
            # Award 2 extra points for friendly sign
            for quality in planet_qualities.get(planet, []):
                scores[quality] += 2
        if planet_rashi_num == exalted_signs.get(planet):
            # Award 2 extra points for exalted sign
            for quality in planet_qualities.get(planet, []):
                scores[quality] += 2

    # Assign scores based on the 10th house lord
    tenthlord = row['Tenthlord']
    if tenthlord in planet_qualities:
        for quality in planet_qualities[tenthlord]:
            scores[quality] += 2  # 2 points for qualities associated with the Tenthlord

    # Add sector-based quality scores
    predicted_sector = row['Predicted_Occupation']
    if predicted_sector in sector_qualities:
        for quality, score in sector_qualities[predicted_sector].items():
            scores[quality] += score

    return scores

# Load the CSV file

df = new_data_df

# Create a new DataFrame for the results
results = []

for _, row in df.iterrows():
    scores = get_scores(row)
    result = {'id': row['id'], 'Predicted_Occupation': row['Predicted_Occupation']}
    result.update(scores)
    results.append(result)

results_df = pd.DataFrame(results)

# Save the results to a new CSV file
results_df.to_csv('scored_qualities.csv', index=False)
st.dataframe(results_df)
train_df, test_df, sector_jobs_df, model = load_data()

#This is CV Insights
st.title("CV Insights")
    
model, tokenizer = load_model1()

st.write("Upload a PDF document and view the quality scores graph.")

pdf_file = st.file_uploader("Choose a PDF file", type="pdf")

if pdf_file is not None:
    st.write("Processing your document...")
    aggregated_quality_scores = extract_and_predict_from_resume(pdf_file, model, tokenizer)
    
 
    plot_quality_scores(aggregated_quality_scores)
     

    st.write("Aggregated Quality Scores:")
    cvscoredf = pd.DataFrame([aggregated_quality_scores])
    cvscoredf1 = cvscoredf
    cvscoredf = cvscoredf.squeeze()

    # Ensure both DataFrames are 2D and have the same columns
    cvscoredf = pd.DataFrame(cvscoredf).T if cvscoredf.ndim == 1 else cvscoredf

    # List of 18 qualities
    qualities = ['Analytical', 'Practical', 'Creative', 'Leadership', 'Hard', 'Smart', 
                'Technical', 'Caring', 'Communication', 'Persuasive', 'Integrity', 
                'Imagination', 'Risk', 'Spontaneous', 'Determination', 'Patience', 
                'Knowledge', 'Wisdom']

    # Filter both DataFrames for only the qualities
    cvscoredf = cvscoredf[qualities]
    results_df = results_df[qualities]

    # Convert everything to numeric
    results_df = results_df.apply(pd.to_numeric, errors='coerce')
    cvscoredf = cvscoredf.apply(pd.to_numeric, errors='coerce')

    # Align indices if needed
    cvscoredf.reset_index(drop=True, inplace=True)
    results_df.reset_index(drop=True, inplace=True)

    # Sum them up
    results_df = results_df.add(cvscoredf, fill_value=0)
    results_df['id'] = 0

    # Display results
    st.dataframe(results_df)


df1 = new_data_df  # First dataset
df2 = jobs_df  # Second dataset

# Merge the datasets on the common column (e.g., 'id') and keep all columns
merged_df = pd.merge(df1, df2, on="id", how="outer", suffixes=("", "_dup"))

# Remove duplicate columns (keeping only one copy)
merged_df = merged_df.loc[:, ~merged_df.columns.str.endswith("_dup")]

# Assign new variables as columns
merged_df = merged_df.assign(Name=name, dob=str(dob), tob=str(tob), pob=str(place_of_birth))

# Reorder columns to place "id", "Name", "dob", "tob", "pob" at the front
new_order = ["id", "Name", "dob", "tob", "pob"] + [col for col in merged_df.columns if col not in ["id", "Name", "dob", "tob", "pob"]]
merged_df = merged_df[new_order]

# Display the merged dataset
st.dataframe(merged_df)


import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Define the scope for Google Sheets and Drive API
SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/spreadsheets",
         "https://www.googleapis.com/auth/drive"]

# Path to your service account credentials file
SERVICE_ACCOUNT_FILE = ".streamlit/db-gsheetskapil-d740f97d71c0.json"

# Authenticate using the service account
credentials = ServiceAccountCredentials.from_json_keyfile_name(SERVICE_ACCOUNT_FILE, SCOPE)
client = gspread.authorize(credentials)

# Open the Google Sheet
sheet_url = "https://docs.google.com/spreadsheets/d/1CFinJcd72seWKhWFMWV6IuZB00ipkaA7q0VASaUAZaA/edit?usp=sharing"
sheet = client.open_by_url(sheet_url).sheet1

# Load dataset
  # Replace with actual file path

# Convert DataFrame to list format (including headers)
data_with_headers = [merged_df.columns.tolist()] + merged_df.values.tolist()

# Check if the sheet is empty and initialize headers
if not sheet.get_all_values():
    st.warning("Google Sheet is empty! Initializing with headers...")
    sheet.append_row(merged_df.columns.tolist())
    st.success("Headers added to the Google Sheet!")

# Get the current data from Google Sheets to find the next ID
existing_data = pd.DataFrame(sheet.get_all_records())
existing_ids = existing_data['id'].tolist() if not existing_data.empty else []

# Get the next ID (incrementing)
next_id = max(existing_ids, default=0) + 1

# Display the current data
st.write("### Current Data in Google Sheet:")
st.dataframe(existing_data)

# Button to add records










# Encrypt function (same as your original QR code encryption logic)
def encrypt(data):
    laxmi_chautisa = [
        [16, 9, 4, 5],
        [3, 6, 14, 10],
        [13, 12, 1, 8],
        [2, 7, 14, 11]
    ]
    encrypted_text = ""
    n = len(laxmi_chautisa)

    for i, char in enumerate(data):
        unicode_val = ord(char)
        row = i % n
        col = i % n
        new_unicode = unicode_val + laxmi_chautisa[row][col]
        encrypted_text += chr(new_unicode)

    return encrypted_text

# Generate QR code and save as an image
def generate_qr_code(planetary_positions,id_number):
    
    # Extract necessary data from planetary positions
    ascendant_rashi = planetary_positions['Ascendant']['Rashi']
    sun_house = planetary_positions['Sun']['House']
    moon_house = planetary_positions['Moon']['House']
    mercury_house = planetary_positions['Mercury']['House']
    venus_house = planetary_positions['Venus']['House']
    mars_house = planetary_positions['Mars']['House']
    saturn_house = planetary_positions['Saturn']['House']
    jupiter_house = planetary_positions['Jupiter']['House']
    rahu_house = planetary_positions['Rahu']['House']
    ketu_house = planetary_positions['Ketu']['House']

    sun_degrees = planetary_positions['Sun']['Degrees']
    moon_degrees = planetary_positions['Moon']['Degrees']
    mercury_degrees = planetary_positions['Mercury']['Degrees']
    venus_degrees = planetary_positions['Venus']['Degrees']
    mars_degrees = planetary_positions['Mars']['Degrees']
    jupiter_degrees = planetary_positions['Jupiter']['Degrees']
    saturn_degrees = planetary_positions['Saturn']['Degrees']
    rahu_degrees = planetary_positions['Rahu']['Degrees']
    ketu_degrees = planetary_positions['Ketu']['Degrees']

    # Format data for QR code as per your format
    qr_data = f"{ascendant_rashi}/{moon_degrees}#\
{sun_house}/{moon_house}/{mercury_house}/{venus_house}/{mars_house}/{saturn_house}/{jupiter_house}/{rahu_house}/{ketu_house}#\
{sun_degrees}/{mercury_degrees}/{venus_degrees}/{mars_degrees}/{jupiter_degrees}/{saturn_degrees}/{rahu_degrees}/{ketu_degrees}#\
{next_id}"

   

    return qr_data





# Button to generate QR code
if st.button("Generate QR Code"):
    if name and place_of_birth:
        # Get birthdate and time
        birth_datetime = datetime.combine(dob, tob)

        # Get latitude and longitude for the place of birth
        lat, lon = get_lat_lon(place_of_birth)

        # Get planetary positions
        positions = get_planetary_positions(birth_datetime, lat, lon)
        result = st.text(positions)
        # Prepare the data for QR code
        
        # Generate QR code
        qr_data = generate_qr_code(positions,id_number)
        # Encrypt data
        encrypted_data = encrypt(qr_data)

        qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
        )
        qr.add_data(encrypted_data)  # Use your encrypted QR data
        qr.make(fit=True)

        # Create QR image
        img = qr.make_image(fill='black', back_color=(193, 236, 249))

        # Convert the QR image to RGBA to allow for transparency
        img = img.convert("RGBA")

        # Create a transparent overlay for the text
        txt_overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))  # Transparent background
        draw = ImageDraw.Draw(txt_overlay)

        # Load custom font (make sure the path is correct)
        custom_font_path = "Hind-Bold.ttf"  # Replace with your custom font path
        font_size = 30  # Adjust the font size
        font = ImageFont.truetype(custom_font_path, font_size)

        # Add custom text to the image
        text = "DAIVA"
        text_position = (50, 0)  # Adjust the position for the text
        text_color = (193, 236, 249, 255)  # Text color

        text_size = [90, 25]

        # Define background rectangle position (a bit bigger than the text)
        background_position = [
            text_position[0] - 10, text_position[1] - 10,  # Top left
            text_position[0] + text_size[0] + 10, text_position[1] + text_size[1] + 10  # Bottom right
        ]

        # Draw the background rectangle
        bg_color = (6, 8, 33)  # Background color
        draw.rectangle(background_position, fill=bg_color)

        # Draw the text on the overlay
        draw.text(text_position, text, font=font, fill=text_color)

        # Add additional text (e.g., instructions)
        font = ImageFont.truetype(custom_font_path, 10)
        instruction_text = "Scan the QR on DAIVA web APP id : "+ str(next_id)
        instruction_position = (50, 340)  # Adjust position for this text
        instruction_color = (6, 8, 33, 210)  # Text color

        # Draw the instruction text on the overlay
        draw.text(instruction_position, instruction_text, font=font, fill=instruction_color)

        # Composite the QR image with the text overlay
        final_img = Image.alpha_composite(img, txt_overlay)

        # Convert the final image to bytes for display in Streamlit
        img_byte_arr = io.BytesIO()
        final_img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        qr_img = img_byte_arr


        # Convert bytes to image
        image = Image.open(io.BytesIO(qr_img))

        import yagmail
        import io
        from PIL import Image

        # Email credentials
        username = "daivathepredictor@gmail.com"
        app_password = "elgx htbf tofv yfnk"  # Use the App Password you generated

        # Email content
        to_email = mail
        subject = f"Namaste {name}, Your Destiny has Unfolded. We have your QR Chart Ready !"
        body = f"""
        Namaste {name},

        This is your QR Chart. Save it and use it afterward.  

        ** For Predictions **
        Go to **DAIVA Web App**, use **Scan My Chart** or upload this in the **Choose File** option.  
        Once done, click **Predict** to see the results.

        ** For Matchmaking **
        Go to **DAIVA Web App**, use **Scan My Chart** or upload this in the **Choose File** option.  
        Once done, click **Match Chart** and Scan the QR of your match and get the compatibility results.

        Best Regards,  
        DAIVA 
        """

        # Convert PIL Image to an in-memory file
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="PNG")  # Save image as PNG in memory
        img_buffer.seek(0)  # Reset file pointer to the beginning
        img_buffer.name = "QRChart.png"  # Assign a name for the email attachment

        try:
            # Initialize yagmail
            yag = yagmail.SMTP(username, app_password)

            # Send the email with the in-memory image attachment
            yag.send(to=to_email, subject=subject, contents=body, attachments=[img_buffer])

            print("Email sent successfully with images!")
        except Exception as e:
            print(f"Failed to send email: {e}")

        # Save the image if needed
        image.save("qr_code.png")
        # Display the QR code
        st.image(qr_img)

        for index, row in merged_df.iterrows():
                row['id'] = next_id
                new_record =  row.tolist()  # Add the incremented ID
                sheet.append_row(new_record, value_input_option="RAW")
                next_id += 1  # Increment the ID for next record
            
        st.success(f"✅ New records with incremented IDs added successfully! 🚀")

        with open("qr_code.png", "rb") as img_file:
            st.download_button("Download QR Code", img_file, file_name="qr_code.png")
    else:
        st.error("Please provide all required details!")
