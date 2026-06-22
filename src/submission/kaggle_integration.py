from typing import Literal

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import ApiSubmission
from kagglesdk.competitions.types.submission_status import SubmissionStatus


class KaggleSubmitError(Exception):
    pass


class KaggleSubmitter:
    def __init__(self, competition):
        self.competition = competition
        self._api = None
        self.submissions: list[ApiSubmission] | None = None

    def auth(self) -> None:
        if self._api is None:
            try:
                self._api: KaggleApi = KaggleApi()
                self._api.authenticate()
            except Exception as e:
                raise KaggleSubmitError(f"Failed to Authenticate: {e}")

    def submit_entry(self, submission_hash: str, submission_path: str) -> None:
        self.auth()
        try:
            self._api.competition_submit(message=submission_hash,
                                         competition=self.competition,
                                         file_name=submission_path)
        except Exception as e:
            raise KaggleSubmitError(f"Failed to submit: {e}")

    def get_submissions(self, refresh: bool = True) -> list[ApiSubmission]:
        self.auth()
        try:
            if not refresh and self.submissions is not None:
                return self.submissions

            submissions: list[ApiSubmission] = self._api.competition_submissions(self.competition)
            self.submissions = [s for s in submissions if s.status == SubmissionStatus.COMPLETE]
            return self.submissions
        except Exception as e:
            raise KaggleSubmitError(f"Failed to collect submissions: {e}")

    def get_submission_by_hash(
            self,
            submission_hash: str,
            which: Literal["latest", "all"] = "all",
            refresh: bool = True) -> list[ApiSubmission] | ApiSubmission | None:

        submissions: list[ApiSubmission] = self.get_submissions(refresh=refresh)
        matches: list[ApiSubmission] = [s for s in submissions if submission_hash in s.description]

        if not matches:
            return None

        match which:
            case "all":
                return matches if len(matches) > 1 else matches[0]
            case "latest":
                return max(matches, key=lambda x: x.date)

    def get_kaggle_public_score_by_hash(
            self,
            submission_hash: str,
            which: Literal["latest", "min", "max"] = "latest",
            refresh: bool = True) -> float | None:

        submissions: list[ApiSubmission] = self.get_submissions(refresh=refresh)
        matches: list[ApiSubmission] = [s for s in submissions if submission_hash in s.description]

        if not matches:
            return None

        match which:
            case "latest":
                return float(max(matches, key=lambda x: x.date).public_score)
            case "min":
                return float(min(matches, key=lambda x: x.public_score).public_score)
            case "max":
                return float(max(matches, key=lambda x: x.public_score).public_score)
