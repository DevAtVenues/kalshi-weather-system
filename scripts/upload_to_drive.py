"""
Upload image files to Google Drive and return shareable links.

Usage:
    .venv/bin/python scripts/upload_to_drive.py file1.png file2.png ...

Uses the same OAuth token gspread already authorized.
"""
import json
import sys
from pathlib import Path

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

CREDS_PATH = Path.home() / ".config" / "gspread" / "authorized_user.json"
DRIVE_FOLDER = "Kalshi Trade Screenshots"


def _get_credentials() -> Credentials:
    raw = json.loads(CREDS_PATH.read_text())
    creds = Credentials(
        token=raw.get("token"),
        refresh_token=raw["refresh_token"],
        token_uri=raw["token_uri"],
        client_id=raw["client_id"],
        client_secret=raw["client_secret"],
        scopes=["https://www.googleapis.com/auth/drive"],
    )
    if not creds.valid:
        creds.refresh(Request())
    return creds


def _get_or_create_folder(service, name: str) -> str:
    """Return the folder ID, creating it if it doesn't exist."""
    results = service.files().list(
        q=f"name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false",
        fields="files(id)",
    ).execute()
    files = results.get("files", [])
    if files:
        return files[0]["id"]
    folder = service.files().create(
        body={"name": name, "mimeType": "application/vnd.google-apps.folder"},
        fields="id",
    ).execute()
    return folder["id"]


def upload_images(paths: list[Path]) -> None:
    creds = _get_credentials()
    service = build("drive", "v3", credentials=creds)
    folder_id = _get_or_create_folder(service, DRIVE_FOLDER)

    for path in paths:
        media = MediaFileUpload(str(path), mimetype="image/png", resumable=False)
        file_meta = {"name": path.name, "parents": [folder_id]}
        f = service.files().create(body=file_meta, media_body=media, fields="id").execute()
        file_id = f["id"]

        # make it viewable by anyone with the link
        service.permissions().create(
            fileId=file_id,
            body={"type": "anyone", "role": "reader"},
        ).execute()

        url = f"https://drive.google.com/file/d/{file_id}/view"
        print(f"{path.name}: {url}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: upload_to_drive.py file1.png file2.png ...")
        sys.exit(1)
    upload_images([Path(p) for p in sys.argv[1:]])
