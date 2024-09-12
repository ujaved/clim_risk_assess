import requests
from datetime import datetime, timedelta
import os

class ZoomClient:

    def __init__(self) -> None:
        self.account_id = os.environ["ZOOM_ACCOUNT_ID"]
        self.client_id = os.environ["ZOOM_CLIENT_ID"]
        self.client_secret = os.environ["ZOOM_CLIENT_SECRET"]
        self.access_token = self.get_access_token()

    def get_access_token(self):
        data = {
            "grant_type": "account_credentials",
            "account_id": self.account_id,
            "client_id": self.client_id,
            "client_secret": self.client_secret
        }
        response = requests.post("https://zoom.us/oauth/token", data=data)
        return response.json()["access_token"]

    def get_recordings(self, num_lookback_months: int)-> dict:
        headers = {"Authorization": f"Bearer {self.access_token}"}
        url = f"https://api.zoom.us/v2/users/me/recordings"
        rv = []
        for m in range(num_lookback_months,0,-1):
            from_ts = datetime.now() - timedelta(days=30*m)
            to_ts = from_ts + timedelta(days=30)
            response_json = requests.get(url, headers=headers, params={"from": from_ts.strftime('%Y-%m-%d'), "to": to_ts.strftime('%Y-%m-%d'), "page_size": 300}).json()
            rv.extend(response_json["meetings"])
        return rv
    
    """
    usage
    for meeting in recs:
        transcript_download_url = [
            f["download_url"]
            for f in meeting["recording_files"]
            if f["file_type"] == "TRANSCRIPT"
        ][0]
        transcript_download_url = (
            f"{transcript_download_url}?access_token={client.access_token}"
        )
        name = f"{meeting['topic']}_{meeting['start_time']}"
    """
    
    def get_audio_download_url(self, meeting_id):
        headers = {"Authorization": f"Bearer {self.access_token}"}
        url = f"https://api.zoom.us/v2/meetings/{meeting_id}/recordings"
        r = requests.get(url, headers=headers).json()
        
        url = [i['download_url'] for i in r['recording_files'] if i['recording_type'] == 'audio_only'][0]
        download_link = f'{url}?access_token={self.access_token}&playback_access_token={r["password"]}'
        return download_link
    
    def get_transcript_download_url(self, meeting_id):
        headers = {"Authorization": f"Bearer {self.access_token}"}
        url = f"https://api.zoom.us/v2/meetings/{meeting_id}/recordings"
        r = requests.get(url, headers=headers).json()
        
        url = [i['download_url'] for i in r['recording_files'] if i['file_type'] == 'TRANSCRIPT'][0]
        return f'{url}?access_token={self.access_token}'