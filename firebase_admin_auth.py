import firebase_admin
from firebase_admin import credentials, firestore, auth
from config import config
from logging_config import logger
import secrets
import datetime
import requests

def initialize_firebase():
    """
    Initialize the Firebase Admin SDK.
    """
    try:
        cred = credentials.Certificate(config.FIREBASE_SERVICE_ACCOUNT_KEY_PATH)
        firebase_admin.initialize_app(cred, {
            'projectId': config.FIREBASE_PROJECT_ID,
        })
        logger.info("Firebase Admin SDK initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Firebase Admin SDK: {str(e)}", exc_info=True)
        raise

initialize_firebase()

def verify_firebase_token(token: str):
    """
    Verify Firebase ID token.
    Returns the user's UID if the token is valid, otherwise None.
    """
    try:
        decoded_token = auth.verify_id_token(token)
        return decoded_token['uid']
    except Exception as e:
        logger.error(f"Invalid Firebase token: {str(e)}", exc_info=True)
        return None

from datetime import timezone

def generate_api_key(user_id: str) -> str:
    """
    Generate a new API key for a user and store it in Firestore.
    """
    db = firestore.client()
    api_key = secrets.token_hex(32)
    key_data = {
        'user_id': user_id,
        'api_key': api_key,
        'created_at': datetime.datetime.now(timezone.utc),
        'active': True,
        'usage_count': 0,
        'expires_at': datetime.datetime.now(timezone.utc) + datetime.timedelta(days=365) # Key expires in 1 year
    }
    db.collection('api_keys').document(api_key).set(key_data)
    logger.info(f"Generated API key for user {user_id}")
    return api_key

def validate_api_key(api_key: str):
    """
    Validate an API key and increment its usage count.
    """
    db = firestore.client()
    key_ref = db.collection('api_keys').document(api_key)
    key_doc = key_ref.get()

    if not key_doc.exists:
        return None

    key_data = key_doc.to_dict()

    if not key_data.get('active') or key_data.get('expires_at') < datetime.datetime.now(timezone.utc):
        return None

    # Increment usage count
    key_ref.update({'usage_count': firestore.Increment(1)})
    
    return key_data

def create_firebase_user(email, password):
    """
    Create a new Firebase user.
    """
    try:
        user = auth.create_user(email=email, password=password)
        logger.info(f"Successfully created new user: {user.uid}")
        return user.uid
    except Exception as e:
        logger.error(f"Failed to create user: {str(e)}", exc_info=True)
        return None

def login_with_email_and_password(email, password):
    """
    Sign in a user with email and password using Firebase Auth REST API.
    Returns the user's UID and ID token if successful, otherwise None.
    """
    rest_api_url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={config.FIREBASE_WEB_API_KEY}"
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    try:
        response = requests.post(rest_api_url, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        user_id = data["localId"]
        id_token = data["idToken"]
        logger.info(f"User {user_id} logged in successfully.")
        return user_id, id_token
    except requests.exceptions.HTTPError as e:
        logger.error(f"Failed to login user: {e.response.text}", exc_info=True)
        return None, None
    except Exception as e:
        logger.error(f"An unexpected error occurred during login: {str(e)}", exc_info=True)
        return None, None