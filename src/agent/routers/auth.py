import os
from datetime import datetime, timedelta, timezone

import httpx
from fastapi import APIRouter
from fastapi.responses import RedirectResponse
from fastapi.security import HTTPBearer
from jose import jwt

router = APIRouter(prefix="/auth", tags=["auth"])
bearer = HTTPBearer()

# Configurations
CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI")
JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGO = "HS256"
JWT_TTL_HOURS = 8

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO = "https://www.googleapis.com/oauth2/v3/userinfo"

# Redirect to Google
@router.get("/login")
def login():
    params = {
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "online"
    }
    query = "&".join(f"{k}={v}" for k, v in params.items())
    return RedirectResponse(f"{GOOGLE_AUTH_URL}?{query}")

@router.get("/callback")
async def callback(code: str):
    # Exchange code for Google access token
    async with httpx.AsyncClient() as client:
        token_resp = await client.post(GOOGLE_TOKEN_URL, data={
            "code": code,
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "redirect_uri": REDIRECT_URI,
            "grant_type": "authorization_code",
        })
        token_resp.raise_for_status()
        google_token = token_resp.json()["access_token"]

        # Fetch user info from Google
        user_resp = await client.get(
            GOOGLE_USERINFO,
            headers={"Authorization": f"Bearer {google_token}"}
        )
        user_resp.raise_for_status()
        user = user_resp.json()

    # Mint your own JWT — no DB write needed
    payload = {
        "sub": user["sub"],          # Google's stable user ID
        "email": user["email"],
        "name": user.get("name"),
        "pic": user.get("picture"),
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_TTL_HOURS),
    }
    your_jwt = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)

    # Return the token however your frontend expects it
    # e.g. redirect to Streamlit with token in query param:
    # return RedirectResponse(f"https://localhost:8501?token={your_jwt}")
    return {"access_token": your_jwt, "token_type": "bearer"}
