import os
import sys

# --- Environment Health Check ---
try:
    import tensorflow as tf
    try:
        from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
    except (ImportError, ModuleNotFoundError, AttributeError):
        from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
except (ImportError, ModuleNotFoundError, AttributeError) as e:
    print("\n" + "!" * 60)
    print("ENVIRONMENT ERROR: Your Python environment is broken or incomplete.")
    print(f"Details: {e}")
    print("\nFIX: You MUST use the virtual environment provided in the project.")
    print("Run these commands in your terminal:")
    print("  cd backend")
    print("  source venv/bin/activate")
    print("  python app.py")
    print("!" * 60 + "\n")
    sys.exit(1)
# --------------------------------

import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from transformers import pipeline
from PIL import Image
import io
import requests
import re
from urllib.parse import urlparse
from typing import Optional
from breed_alternatives import BREED_ALTERNATIVES

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_REMOTE_IMAGE_BYTES = 10 * 1024 * 1024
MOBILENET_WEIGHTS_PATH = os.path.expanduser(
    '~/.keras/models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5'
)
CAT_MODEL_ID = "dima806/67_cat_breeds_image_detection"
CAT_MODEL_CACHE_DIR = os.path.expanduser(
    '~/.cache/huggingface/hub/models--dima806--67_cat_breeds_image_detection/snapshots'
)

# Load Models
dog_model = MobileNetV2(weights=MOBILENET_WEIGHTS_PATH if os.path.exists(MOBILENET_WEIGHTS_PATH) else 'imagenet')
cat_classifier = None
cat_classifier_failed = False

# ImageNet MobileNetV2 has fixed dog classes. Use exact labels instead of loose
# keywords so breeds like Chow Chow and Tibetan Mastiff are not missed.
IMAGENET_DOG_LABELS = {
    'chihuahua', 'japanese_spaniel', 'maltese_dog', 'pekinese', 'shih-tzu',
    'blenheim_spaniel', 'papillon', 'toy_terrier', 'rhodesian_ridgeback',
    'afghan_hound', 'basset', 'beagle', 'bloodhound', 'bluetick',
    'black-and-tan_coonhound', 'walker_hound', 'english_foxhound', 'redbone',
    'borzoi', 'irish_wolfhound', 'italian_greyhound', 'whippet', 'ibizan_hound',
    'norwegian_elkhound', 'otterhound', 'saluki', 'scottish_deerhound',
    'weimaraner', 'staffordshire_bullterrier', 'american_staffordshire_terrier',
    'bedlington_terrier', 'border_terrier', 'kerry_blue_terrier',
    'irish_terrier', 'norfolk_terrier', 'norwich_terrier', 'yorkshire_terrier',
    'wire-haired_fox_terrier', 'lakeland_terrier', 'sealyham_terrier',
    'airedale', 'cairn', 'australian_terrier', 'dandie_dinmont',
    'boston_bull', 'miniature_schnauzer', 'giant_schnauzer',
    'standard_schnauzer', 'scotch_terrier', 'tibetan_terrier', 'silky_terrier',
    'soft-coated_wheaten_terrier', 'west_highland_white_terrier', 'lhasa',
    'flat-coated_retriever', 'curly-coated_retriever', 'golden_retriever',
    'labrador_retriever', 'chesapeake_bay_retriever',
    'german_short-haired_pointer', 'vizsla', 'english_setter', 'irish_setter',
    'gordon_setter', 'brittany_spaniel', 'clumber', 'english_springer',
    'welsh_springer_spaniel', 'cocker_spaniel', 'sussex_spaniel',
    'irish_water_spaniel', 'kuvasz', 'schipperke', 'groenendael', 'malinois',
    'briard', 'kelpie', 'komondor', 'old_english_sheepdog',
    'shetland_sheepdog', 'collie', 'border_collie', 'bouvier_des_flandres',
    'rottweiler', 'german_shepherd', 'doberman', 'miniature_pinscher',
    'greater_swiss_mountain_dog', 'bernese_mountain_dog', 'appenzeller',
    'entlebucher', 'boxer', 'bull_mastiff', 'tibetan_mastiff',
    'french_bulldog', 'great_dane', 'saint_bernard', 'eskimo_dog', 'malamute',
    'siberian_husky', 'affenpinscher', 'basenji', 'pug', 'leonberg',
    'newfoundland', 'great_pyrenees', 'samoyed', 'pomeranian', 'chow',
    'keeshond', 'brabancon_griffon', 'pembroke', 'cardigan', 'toy_poodle',
    'miniature_poodle', 'standard_poodle', 'mexican_hairless'
}

IMAGENET_CAT_LABELS = {'tabby', 'tiger_cat', 'persian_cat', 'siamese_cat', 'egyptian_cat'}

BREED_LABEL_OVERRIDES = {
    'japanese_spaniel': 'Japanese Chin',
    'maltese_dog': 'Maltese',
    'pekinese': 'Pekingese',
    'shih-tzu': 'Shih Tzu',
    'blenheim_spaniel': 'Cavalier King Charles Spaniel',
    'basset': 'Basset Hound',
    'bluetick': 'Bluetick Coonhound',
    'black-and-tan_coonhound': 'Black and Tan Coonhound',
    'walker_hound': 'Treeing Walker Coonhound',
    'redbone': 'Redbone Coonhound',
    'staffordshire_bullterrier': 'Staffordshire Bull Terrier',
    'american_staffordshire_terrier': 'American Staffordshire Terrier',
    'wire-haired_fox_terrier': 'Wire Fox Terrier',
    'airedale': 'Airedale Terrier',
    'cairn': 'Cairn Terrier',
    'dandie_dinmont': 'Dandie Dinmont Terrier',
    'boston_bull': 'Boston Terrier',
    'scotch_terrier': 'Scottish Terrier',
    'soft-coated_wheaten_terrier': 'Soft Coated Wheaten Terrier',
    'west_highland_white_terrier': 'West Highland White Terrier',
    'lhasa': 'Lhasa Apso',
    'german_short-haired_pointer': 'German Shorthaired Pointer',
    'brittany_spaniel': 'Brittany',
    'clumber': 'Clumber Spaniel',
    'english_springer': 'English Springer Spaniel',
    'groenendael': 'Belgian Sheepdog',
    'malinois': 'Belgian Malinois',
    'kelpie': 'Australian Kelpie',
    'german_shepherd': 'German Shepherd Dog',
    'doberman': 'Doberman Pinscher',
    'bull_mastiff': 'Bullmastiff',
    'eskimo_dog': 'Siberian Husky',
    'malamute': 'Alaskan Malamute',
    'chow': 'Chow Chow',
    'brabancon_griffon': 'Brussels Griffon',
    'pembroke': 'Pembroke Welsh Corgi',
    'cardigan': 'Cardigan Welsh Corgi',
    'mexican_hairless': 'Xoloitzcuintli'
}

DOG_API_ALIASES = {
    'German Shepherd Dog': ['German Shepherd'],
    'Bullmastiff': ['Bullmastiff', 'Bull Mastiff'],
    'Cavalier King Charles Spaniel': ['Cavalier King Charles Spaniel', 'Blenheim Spaniel'],
    'Alaskan Malamute': ['Alaskan Malamute', 'Malamute'],
    'Belgian Sheepdog': ['Belgian Sheepdog', 'Groenendael'],
    'Belgian Malinois': ['Belgian Malinois', 'Malinois'],
    'Scottish Terrier': ['Scottish Terrier', 'Scotch Terrier'],
    'Chow Chow': ['Chow Chow', 'Chow'],
    'Saint Bernard': ['Saint Bernard', 'St. Bernard'],
    'Boston Terrier': ['Boston Terrier', 'Boston Bull'],
    'Xoloitzcuintli': ['Xoloitzcuintli', 'Mexican Hairless'],
    'Brittany': ['Brittany', 'Brittany Spaniel'],
    'Wire Fox Terrier': ['Wire Fox Terrier', 'Wirehaired Fox Terrier'],
    'Brussels Griffon': ['Brussels Griffon', 'Brabancon Griffon'],
}

# API Keys
CAT_API_KEY = "live_TpegXJAX6u0Xa61t1c0xoNzfIel207Fn1DGMNpl2Z5U34hSnOqX0rMKheLCKNwlf"
DOG_API_KEY = "live_OXqrWvJgSwpOwsp2rDGCTLgqhuCvQe3UA6wsJpwClQEGZLnAH0Wr1wlYKwdhOISu"

def normalize_breed_name(value):
    return re.sub(r'[^a-z0-9]+', '', (value or '').lower())

def get_local_cat_model_path():
    if not os.path.isdir(CAT_MODEL_CACHE_DIR):
        return None

    snapshots = [
        os.path.join(CAT_MODEL_CACHE_DIR, name)
        for name in os.listdir(CAT_MODEL_CACHE_DIR)
        if os.path.isdir(os.path.join(CAT_MODEL_CACHE_DIR, name))
    ]
    return snapshots[0] if snapshots else None

def get_cat_classifier():
    global cat_classifier, cat_classifier_failed
    if cat_classifier or cat_classifier_failed:
        return cat_classifier

    try:
        model_path = get_local_cat_model_path() or CAT_MODEL_ID
        cat_classifier = pipeline("image-classification", model=model_path)
    except Exception as e:
        cat_classifier_failed = True
        print(f"Cat classifier unavailable: {e}")
    return cat_classifier

def imagenet_label_to_breed(label):
    label = label.lower()
    if label in BREED_LABEL_OVERRIDES:
        return BREED_LABEL_OVERRIDES[label]
    return label.replace('_', ' ').replace('-', ' ').title()

def get_decoded_candidates(decoded, allowed_labels):
    candidates = []
    seen_breeds = set()
    for _, label, prob in decoded:
        normalized_label = label.lower()
        if normalized_label not in allowed_labels:
            continue

        breed = imagenet_label_to_breed(normalized_label)
        if breed in seen_breeds:
            continue

        seen_breeds.add(breed)
        candidates.append({
            'breed': breed,
            'confidence': float(prob),
            'imagenet_label': label
        })
    return candidates

def build_breed_search_terms(breed_name, alias_map=None):
    terms = [breed_name]
    if alias_map and breed_name in alias_map:
        terms.extend(alias_map[breed_name])

    unique_terms = []
    seen = set()
    for term in terms:
        normalized = normalize_breed_name(term)
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_terms.append(term)
    return unique_terms

def find_exact_breed_match(data, accepted_names):
    accepted = {normalize_breed_name(name) for name in accepted_names}
    return next((breed for breed in data if normalize_breed_name(breed.get('name')) in accepted), None)

def image_matches_breed(image_data, breed_id, accepted_names):
    image_breeds = image_data.get('breeds') or []
    if not image_breeds:
        return False

    accepted = {normalize_breed_name(name) for name in accepted_names}
    for breed in image_breeds:
        same_id = breed_id is not None and str(breed.get('id')) == str(breed_id)
        same_name = normalize_breed_name(breed.get('name')) in accepted
        if same_id or same_name:
            return True
    return False

def get_verified_breed_image(api_base_url, headers, breed, accepted_names):
    breed_id = breed.get('id')
    reference_image_id = breed.get('reference_image_id')

    if reference_image_id:
        image_res = requests.get(f"{api_base_url}/images/{reference_image_id}", headers=headers, timeout=5)
        if image_res.status_code == 200:
            image_data = image_res.json()
            image_url = image_data.get('url')
            if image_url and (
                not image_data.get('breeds') or image_matches_breed(image_data, breed_id, accepted_names)
            ):
                return image_url

    if not breed_id:
        return None

    image_res = requests.get(
        f"{api_base_url}/images/search",
        headers=headers,
        params={"breed_ids": breed_id, "limit": 10},
        timeout=5
    )
    if image_res.status_code != 200:
        return None

    for image_data in image_res.json():
        image_url = image_data.get('url')
        if image_url and image_matches_breed(image_data, breed_id, accepted_names):
            return image_url
    return None

def get_cat_api_data(breed_name):
    """Fetch detailed cat breed info from TheCatAPI with strict matching."""
    try:
        api_base_url = "https://api.thecatapi.com/v1"
        url = f"{api_base_url}/breeds/search"
        headers = {"x-api-key": CAT_API_KEY}
        response = requests.get(url, headers=headers, params={"q": breed_name}, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data and isinstance(data, list):
                breed = find_exact_breed_match(data, [breed_name])
                if not breed:
                    return None

                breed_id = breed.get('id')
                img_url = get_verified_breed_image(api_base_url, headers, breed, [breed_name])
                
                return {
                    'summary': breed.get('description'),
                    'temperament': breed.get('temperament'),
                    'life_span': breed.get('life_span') + " years" if breed.get('life_span') else "12-15 years",
                    'weight': breed.get('weight', {}).get('metric') + " kg" if breed.get('weight', {}).get('metric') else "4-7 kg",
                    'origin': breed.get('origin'),
                    'image': img_url,
                    'source': 'TheCatAPI'
                }
    except Exception as e:
        print(f"TheCatAPI error for {breed_name}: {e}")
    return None

def get_dog_api_data(breed_name):
    """Fetch detailed dog breed info from TheDogAPI with strict matching."""
    try:
        api_base_url = "https://api.thedogapi.com/v1"
        url = f"{api_base_url}/breeds/search"
        headers = {"x-api-key": DOG_API_KEY}
        search_terms = build_breed_search_terms(breed_name, DOG_API_ALIASES)

        for search_term in search_terms:
            response = requests.get(url, headers=headers, params={"q": search_term}, timeout=5)
            if response.status_code != 200:
                continue

            data = response.json()
            if not data or not isinstance(data, list):
                continue

            breed = find_exact_breed_match(data, search_terms)
            if not breed:
                continue

            breed_id = breed.get('id')
            img_url = get_verified_breed_image(api_base_url, headers, breed, search_terms)
            
            return {
                'summary': breed.get('description') or breed.get('history') or f"The {breed_name} is a distinguished breed with unique characteristics.",
                'temperament': breed.get('temperament'),
                'life_span': breed.get('life_span'),
                'weight': breed.get('weight', {}).get('metric') + " kg" if breed.get('weight', {}).get('metric') else "Varies",
                'origin': breed.get('origin') or "International",
                'image': img_url,
                'matched_breed': breed.get('name'),
                'source': 'TheDogAPI'
            }
    except Exception as e:
        print(f"TheDogAPI error for {breed_name}: {e}")
    return None

def get_duckduckgo_summary(breed_name, category):
    """Fetch summary from DuckDuckGo Instant Answer API."""
    try:
        query = f"{breed_name} {category}"
        url = f"https://api.duckduckgo.com/?q={query}&format=json"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            abstract = data.get('AbstractText')
            if not abstract and data.get('RelatedTopics'):
                abstract = data['RelatedTopics'][0].get('Text')
            
            if abstract:
                return {
                    'summary': abstract,
                    'image': None,
                    'source': 'DuckDuckGo Search',
                    'temperament': 'Loyal, Intelligent',
                    'life_span': '10-15 years',
                    'weight': 'Varies',
                    'origin': 'International'
                }
    except Exception as e:
        print(f"DuckDuckGo error for {breed_name}: {e}")
    return None

def preprocess_image(image, target_size=(224, 224)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image_resized = image.resize(target_size)
    img_array = np.array(image_resized)
    img_batch = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_batch)

async def load_image_from_request(image: Optional[UploadFile], image_url: Optional[str]):
    if image:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents))
        img.load()
        return img

    if not image_url:
        raise HTTPException(status_code=400, detail='No image provided')

    parsed_url = urlparse(image_url)
    if parsed_url.scheme not in ('http', 'https') or not parsed_url.netloc:
        raise HTTPException(status_code=400, detail='The dropped image URL is not valid.')

    headers = {
        'User-Agent': 'Mozilla/5.0 (compatible; PetIdentifier/1.0)'
    }
    response = requests.get(image_url, headers=headers, timeout=10, stream=True)
    response.raise_for_status()

    content_type = response.headers.get('Content-Type', '').split(';')[0].lower()
    if content_type and content_type.startswith('text/'):
        raise HTTPException(status_code=400, detail='The dropped link is a web page, not a direct image. Open the image first, then drag it here.')

    image_bytes = io.BytesIO()
    total_size = 0
    for chunk in response.iter_content(chunk_size=8192):
        if not chunk:
            continue

        total_size += len(chunk)
        if total_size > MAX_REMOTE_IMAGE_BYTES:
            raise HTTPException(status_code=400, detail='The dropped image is too large. Please use an image under 10 MB.')

        image_bytes.write(chunk)

    image_bytes.seek(0)
    img = Image.open(image_bytes)
    img.load()
    return img

# Expert Knowledge Base (Fallback)
BREED_DATA = {
    'Golden Retriever': {
        'origin': 'Scotland',
        'life_span': '10-12 years',
        'weight': '25-34 kg',
        'temperament': 'Intelligent, Kind, Trustworthy, Confident',
        'summary': 'The Golden Retriever is a sturdy, muscular dog of medium size, famous for the dense, lustrous coat of gold that gives the breed its name. They are friendly, reliable, and trustworthy family pets.'
    },
    'German Shepherd Dog': {
        'origin': 'Germany',
        'life_span': '9-13 years',
        'weight': '22-40 kg',
        'temperament': 'Alert, Loyal, Obedient, Courageous',
        'summary': 'The German Shepherd is a breed of medium to large-sized working dog that originated in Germany. It is known for its intelligence and versatility as a service and protection dog.'
    },
    'Chow Chow': {
        'origin': 'China',
        'life_span': '9-15 years',
        'weight': '20-32 kg',
        'temperament': 'Independent, Loyal, Reserved, Dignified',
        'summary': 'The Chow Chow is a compact, powerfully built spitz-type dog known for its dense coat, broad head, blue-black tongue, and reserved personality.'
    },
    'Tibetan Mastiff': {
        'origin': 'Tibet',
        'life_span': '10-14 years',
        'weight': '32-68 kg',
        'temperament': 'Protective, Independent, Strong-willed, Loyal',
        'summary': 'The Tibetan Mastiff is a large guardian breed with a heavy coat, substantial bone, and a protective temperament developed for watching livestock and homes in Himalayan regions.'
    },
    'American Staffordshire Terrier': {
        'origin': 'United States',
        'life_span': '12-16 years',
        'weight': '18-32 kg',
        'temperament': 'Confident, Loyal, Courageous, Affectionate',
        'summary': 'The American Staffordshire Terrier is a muscular terrier breed often visually confused with pit bull-type dogs and American Bullies. Coat, head shape, body proportion, and pedigree are needed for a confident distinction.'
    },
    'Staffordshire Bull Terrier': {
        'origin': 'England',
        'life_span': '12-14 years',
        'weight': '11-17 kg',
        'temperament': 'Bold, Affectionate, Clever, Reliable',
        'summary': 'The Staffordshire Bull Terrier is a compact, muscular terrier breed with a broad head and affectionate temperament. It can resemble other bully-type breeds in photographs.'
    },
    'Bullmastiff': {
        'origin': 'England',
        'life_span': '7-9 years',
        'weight': '41-59 kg',
        'temperament': 'Protective, Loyal, Docile, Alert',
        'summary': 'The Bullmastiff is a large working breed developed from mastiff and bulldog ancestry. It is powerful, broad-headed, and protective, but visually distinct from smaller bully-type terriers.'
    },
    'Boxer': {
        'origin': 'Germany',
        'life_span': '10-12 years',
        'weight': '25-32 kg',
        'temperament': 'Bright, Fun-loving, Active, Loyal',
        'summary': 'The Boxer is a medium-to-large working dog with a square muzzle, athletic body, and energetic temperament. Some photos can be confused with bully-type breeds.'
    },
    'Japanese Spitz': {
        'origin': 'Japan',
        'life_span': '10-16 years',
        'weight': '5-10 kg',
        'temperament': 'Affectionate, Obedient, Playful, Companionable',
        'summary': 'The Japanese Spitz is a small-to-medium breed known for its thick, pure white double coat and fox-like appearance. It is highly similar to the American Eskimo Dog but often slightly smaller and known for being a very clean, "low-odor" dog.'
    },
    'American Eskimo Dog': {
        'origin': 'Germany / United States',
        'life_span': '13-15 years',
        'weight': '3-16 kg (Toy/Mini/Standard)',
        'temperament': 'Intelligent, Alert, Friendly, Playful',
        'summary': 'Despite its name, the American Eskimo Dog originated from the German Spitz. It is a striking white dog with a keen intelligence, often used as a circus performer in the past. It looks very similar to the Japanese Spitz and Samoyed.'
    },
    'Samoyed': {
        'origin': 'Russia (Siberia)',
        'life_span': '12-14 years',
        'weight': '16-30 kg',
        'temperament': 'Friendly, Sociable, Alert, Stubborn',
        'summary': 'The Samoyed is a large, powerful Arctic dog famous for the "Samoyed Smile" caused by the upturned corners of its mouth. It has a thick white coat and was originally bred for herding reindeer and hauling sleds.'
    },
    'Indian Pariah Dog': {
        'origin': 'India',
        'life_span': '13-16 years',
        'weight': '15-25 kg',
        'temperament': 'Highly Intelligent, Alert, Social, Loyal',
        'summary': 'The Indian Pariah Dog is an ancient landrace native to the Indian subcontinent. They are extremely hardy, intelligent, and well-adapted to the local climate. Often called "Indies," they are one of the oldest primitive dog breeds in the world.'
    },
    'Potcake Dog': {
        'origin': 'Bahamas / Turks & Caicos',
        'life_span': '10-14 years',
        'weight': '20-25 kg',
        'temperament': 'Intelligent, Resilient, Loyal, Friendly',
        'summary': 'Potcakes are a mixed-breed dog type found on several Caribbean islands. They are named after the congealed rice mixture at the bottom of the pot that locals used to feed them. They are known for their resilience and smarts.'
    },
    'Askal/Aspin': {
        'origin': 'Philippines',
        'life_span': '12-15 years',
        'weight': '10-20 kg',
        'temperament': 'Loyal, Protective, Resilient, Adaptable',
        'summary': 'Askals (Asong Kalye) or Aspins (Asong Pinoy) are native mixed-breed dogs from the Philippines. They are known for their high intelligence and loyalty to their owners, often serving as excellent watchdogs.'
    },
    'Africanis': {
        'origin': 'Southern Africa',
        'life_span': '12-15 years',
        'weight': '25-40 kg',
        'temperament': 'Independent, Friendly, Territorial, Hardy',
        'summary': 'The Africanis is a short-haired, medium-sized landrace dog of Southern Africa. It is a primitive dog type that has existed for thousands of years, known for its incredible health and survival instincts.'
    },
    'Soi Dog': {
        'origin': 'Thailand',
        'life_span': '12-15 years',
        'weight': '15-25 kg',
        'temperament': 'Resilient, Intelligent, Independent, Loyal',
        'summary': 'Soi Dogs are the street dogs of Thailand ("Soi" means alley). They are typically mixed-breed dogs that have adapted to urban environments, known for their street-smarts and hardiness.'
    },
    'Carolina Dog': {
        'origin': 'United States',
        'life_span': '12-15 years',
        'weight': '15-20 kg',
        'temperament': 'Reserved, Intelligent, Loyal, Gentle',
        'summary': 'Also known as the American Dingo, the Carolina Dog is a landrace breed discovered in the Southeastern US. They share many traits with other primitive dogs like the Basenji and Dingo.'
    },
    'Formosan Mountain Dog': {
        'origin': 'Taiwan',
        'life_span': '13-16 years',
        'weight': '12-18 kg',
        'temperament': 'Loyal, Intelligent, Alert, Protective',
        'summary': 'The Formosan Mountain Dog is a landrace of small-to-medium dogs in Taiwan. They are highly intelligent, agile, and known for their extreme loyalty to their family.'
    },
    'Bali Dog': {
        'origin': 'Indonesia (Bali)',
        'life_span': '12-15 years',
        'weight': '12-15 kg',
        'temperament': 'Hardy, Alert, Independent, Loyal',
        'summary': 'The Bali Street Dog is an ancient landrace from the island of Bali. They are genetically distinct from most modern breeds and are known for their resilience and varied appearance.'
    },
    'Mixed Breed': {
        'origin': 'International',
        'life_span': 'Varies',
        'weight': 'Varies',
        'temperament': 'Varies, usually highly adaptable',
        'summary': 'A mixed-breed dog (also called a mutt or mongrel) has parents that are not both of the same breed. They often benefit from "hybrid vigor" and can have a wide variety of appearances and temperaments.'
    }
}

@app.post('/predict')
async def predict(
    image: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None)
):
    try:
        img = await load_image_from_request(image, image_url)
        processed_img = preprocess_image(img)
        preds = dog_model.predict(processed_img, verbose=0)
        decoded = decode_predictions(preds, top=10)[0]
        
        dog_candidates = get_decoded_candidates(decoded, IMAGENET_DOG_LABELS)
        cat_candidates = get_decoded_candidates(decoded, IMAGENET_CAT_LABELS)
        is_dog = bool(dog_candidates)
        is_cat = bool(cat_candidates)

        breed_name = ""
        category = ""
        details = {}
        confidence = 0.0
        alternatives = []
        cat_results = []

        if dog_candidates and (not cat_candidates or dog_candidates[0]['confidence'] >= cat_candidates[0]['confidence']):
            category = "DOG"
            breed_name = dog_candidates[0]['breed']
            confidence = dog_candidates[0]['confidence']
            alternatives = BREED_ALTERNATIVES.get(breed_name, [])
            
            # ROUTING: Go to Dog API First
            details = get_dog_api_data(breed_name) or {}
            
            # Fallback to Expert Base or DuckDuckGo if Dog API has no summary
            if not details.get('summary'):
                details.update(BREED_DATA.get(breed_name, {}))
            if not details.get('summary'):
                details.update(get_duckduckgo_summary(breed_name, "Dog") or {})

        elif cat_candidates:
            category = "CAT"
            classifier = get_cat_classifier()
            if classifier:
                cat_results = classifier(img)
                breed_name = cat_results[0]['label'].replace('_', ' ').title()
                confidence = float(cat_results[0]['score'])
            else:
                cat_results = cat_candidates
                breed_name = cat_candidates[0]['breed']
                confidence = cat_candidates[0]['confidence']
            
            alternatives = BREED_ALTERNATIVES.get(breed_name, [])

            # ROUTING: Go to Cat API First
            details = get_cat_api_data(breed_name) or {}
            
            # Fallback if Cat API has no summary
            if not details.get('summary'):
                details.update(get_duckduckgo_summary(breed_name, "Cat") or {})
        
        if breed_name:
            return {
                'category': category,
                'details': breed_name,
                'confidence': confidence,
                'is_dog': category == "DOG",
                'is_cat': category == "CAT",
                'detected_dog_candidate': is_dog,
                'detected_cat_candidate': is_cat,
                'alternatives': alternatives,
                'candidates': dog_candidates[:5] if category == "DOG" else cat_results[:5],
                'info': details
            }

        return JSONResponse(status_code=400, content={'error': 'No dog or cat detected in the image.'})
            
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={'error': e.detail})
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 5050)))
