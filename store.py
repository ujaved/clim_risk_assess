from supabase import create_client, Client
from datetime import date


class DBClient:
    def __init__(self, supabase_url: str, supabase_key: str) -> Client:
        self.client = create_client(supabase_url, supabase_key)

    def sign_in(self, email: str, password: str):
        return self.client.auth.sign_in_with_password(
            {"email": email, "password": password}
        ).user

    def update_user_password(self, user_id: str, password: str):
        return self.client.auth.update_user_by_id(user_id, {"password": password}).user

    def invite_user_by_email(
        self, email: str, first_name: str, last_name: str, org: str
    ) -> None:
        self.client.auth.auth.admin.invite_user_by_email(
            email,
            options={
                "data": {
                    "first_name": first_name,
                    "last_name": last_name,
                    "organization": org,
                }
            },
        )

    def insert_num_questions(self, recording_id: str, json: dict) -> None:
        self.client.table("recording_stats").insert(
            {"recording_id": recording_id, "num_questions": json}
        ).execute()

    def get_num_questions(self, recording_id: str) -> dict | None:
        recording_stats = (
            self.client.table("recording_stats")
            .select("num_questions")
            .eq("recording_id", recording_id)
            .maybe_single()
            .execute()
        )
        if recording_stats:
            return recording_stats.data["num_questions"]
        return None

    def get_recording_stats(self, recordind_ids: list[str]) -> list[dict]:
        return (
            self.client.table("recording_stats")
            .select("*")
            .in_("recording_id", recordind_ids)
            .execute()
            .data
        )

    def get_recordings(self, teacher_id: str, class_id: str) -> list[dict]:
        return (
            self.client.table("recordings")
            .select("*")
            .eq("user_id", teacher_id)
            .eq("class_id", class_id)
            .execute()
            .data
        )

    def get_classes(self, teacher_id: str) -> list[dict]:
        return (
            self.client.table("classes")
            .select("*")
            .eq("teacher_id", teacher_id)
            .execute()
            .data
        )

    def get_speakers(self, class_id: str) -> dict:
        return (
            self.client.table("speakers")
            .select("*")
            .eq("class_id", class_id)
            .execute()
            .data
        )

    def insert_speaker(self, class_id: str, name: str, image_s3_key: str) -> None:
        self.client.table("speakers").insert(
            {
                "class_id": class_id,
                "name": name,
                "s3_key": image_s3_key,
            }
        ).execute()

    def get_orgs(self) -> dict:
        return self.client.table("organizations").select("*").execute().data

    def insert_org(self, name: str) -> None:
        self.client.table("organizations").insert({"name": name}).execute()

    def insert_recording(
        self, user_id: str, link: str, date: date, class_id: str
    ) -> None:
        self.client.table("recordings").insert(
            {
                "user_id": user_id,
                "link": link,
                "date": date.isoformat(),
                "class_id": class_id,
            }
        ).execute()

    def insert_class(self, name: str, teacher_id: str, org_id: str) -> None:
        self.client.table("classes").insert(
            {
                "name": name,
                "teacher_id": teacher_id,
                "org_id": org_id,
            }
        ).execute()
