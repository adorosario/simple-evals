"""
CustomGPT Explainability API Client

Fetches claims breakdown and trust scores for post-mortem analysis of failed queries.
Uses the new Verify Responses (Explainability) feature available via API.

API Endpoints:
- GET /api/v1/projects/{projectId}/conversations/{sessionId}/messages/{promptId}/claims
- GET /api/v1/projects/{projectId}/conversations/{sessionId}/messages/{promptId}/trust-score

Note: Explainability analysis costs 4 queries per message and runs asynchronously.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import requests


@dataclass
class Claim:
    """A single claim extracted from a response"""
    claim_id: int
    text: str
    source: Optional[Dict[str, Any]] = None  # citation_id, snippet if sourced
    flagged: bool = False
    confidence: float = 0.0
    stakeholder_assessments: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class TrustScore:
    """Trust score metrics for a response"""
    overall: float = 0.0
    sourced_claims_ratio: float = 0.0
    flagged_claims_count: int = 0
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExplainabilityResult:
    """Complete explainability analysis result"""
    claims: List[Claim] = field(default_factory=list)
    trust_score: Optional[TrustScore] = None
    stakeholder_analysis: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    overall_status: str = "UNKNOWN"
    analysis_cost_queries: int = 4
    raw_claims_response: Optional[Dict[str, Any]] = None
    raw_trust_response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ExplainabilityClient:
    """
    Client for CustomGPT Explainability API (Verify Responses feature)

    Usage:
        client = ExplainabilityClient()
        result = client.get_explainability(project_id, session_id, prompt_id)
    """

    BASE_URL = "https://app.customgpt.ai/api/v1"

    # Stakeholder perspectives analyzed by the explainability feature
    STAKEHOLDERS = [
        "end_user",
        "security_it",
        "risk_compliance",
        "legal_compliance",
        "public_relations",
        "executive_leadership"
    ]

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the explainability client.

        Args:
            api_key: CustomGPT API key. If not provided, reads from CUSTOMGPT_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("CUSTOMGPT_API_KEY")
        if not self.api_key:
            raise ValueError("CUSTOMGPT_API_KEY environment variable not set")

        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json"
        })

    def _build_url(self, project_id: str, session_id: str, prompt_id: str, endpoint: str) -> str:
        """Build the API URL for the given endpoint"""
        return f"{self.BASE_URL}/projects/{project_id}/conversations/{session_id}/messages/{prompt_id}/{endpoint}"

    def _make_request(
        self,
        url: str,
        max_retries: int = 3,
        initial_delay: float = 2.0,
        timeout: float = 30.0
    ) -> Optional[Dict[str, Any]]:
        """
        Make a GET request with retry logic.

        The explainability analysis runs asynchronously, so we may need to retry
        if the analysis hasn't completed yet.
        """
        for attempt in range(max_retries):
            try:
                response = self._session.get(url, timeout=timeout)

                if response.status_code == 200:
                    return response.json()

                elif response.status_code == 202:
                    # Analysis still in progress - wait and retry
                    delay = initial_delay * (2 ** attempt)
                    print(f"Explainability analysis in progress, waiting {delay:.1f}s...")
                    time.sleep(delay)
                    continue

                elif response.status_code == 404:
                    # Message not found or explainability not available
                    print(f"Explainability data not found (404): {url}")
                    return None

                elif response.status_code == 429:
                    # Rate limited
                    retry_after = int(response.headers.get("Retry-After", 30))
                    print(f"Rate limited, waiting {retry_after}s...")
                    time.sleep(retry_after)
                    continue

                else:
                    print(f"API error {response.status_code}: {response.text[:200]}")
                    if attempt < max_retries - 1:
                        time.sleep(initial_delay * (attempt + 1))
                        continue
                    return None

            except requests.exceptions.Timeout:
                print(f"Request timeout (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(initial_delay * (attempt + 1))
                    continue
                return None

            except requests.exceptions.RequestException as e:
                print(f"Request error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(initial_delay)
                    continue
                return None

        return None

    def get_claims(
        self,
        project_id: str,
        session_id: str,
        prompt_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch claims breakdown for a message.

        Returns the raw API response containing:
        - List of claims made in the response
        - Source attribution per claim
        - Flagged status (true if no source found)
        - Stakeholder assessments (6 perspectives)
        """
        url = self._build_url(project_id, session_id, str(prompt_id), "claims")
        print(f"Fetching claims: {url}")
        return self._make_request(url)

    def get_trust_score(
        self,
        project_id: str,
        session_id: str,
        prompt_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch trust score for a message.

        Returns the raw API response containing:
        - Overall trust score
        - Confidence metrics
        """
        url = self._build_url(project_id, session_id, str(prompt_id), "trust-score")
        print(f"Fetching trust score: {url}")
        return self._make_request(url)

    def get_explainability(
        self,
        project_id: str,
        session_id: str,
        prompt_id: str
    ) -> ExplainabilityResult:
        """
        Fetch complete explainability analysis for a message.

        Combines claims and trust score into a unified result with parsed data structures.

        Args:
            project_id: CustomGPT project ID
            session_id: Conversation session ID
            prompt_id: Message/prompt ID

        Returns:
            ExplainabilityResult with claims, trust score, and stakeholder analysis
        """
        result = ExplainabilityResult()

        # Fetch claims
        claims_response = self.get_claims(project_id, session_id, prompt_id)
        result.raw_claims_response = claims_response

        if claims_response:
            result.claims = self._parse_claims(claims_response)
            result.stakeholder_analysis = self._parse_stakeholder_analysis(claims_response)
            result.overall_status = self._determine_overall_status(result.claims)
        else:
            result.error = "Failed to fetch claims data"

        # Fetch trust score
        trust_response = self.get_trust_score(project_id, session_id, prompt_id)
        result.raw_trust_response = trust_response

        if trust_response:
            result.trust_score = self._parse_trust_score(trust_response, result.claims)

        return result

    def _parse_claims(self, response: Any) -> List[Claim]:
        """Parse claims from API response"""
        claims = []

        # Handle different response structures
        # API may return: {"data": {"claims": [...]}} or {"claims": [...]} or [...]
        if isinstance(response, list):
            # Direct list of claims
            claims_list = response
        elif isinstance(response, dict):
            data = response.get("data", response)
            if isinstance(data, list):
                claims_list = data
            else:
                claims_list = data.get("claims", [])
        else:
            claims_list = []

        if isinstance(claims_list, list):
            for i, claim_data in enumerate(claims_list):
                if isinstance(claim_data, dict):
                    claim = Claim(
                        claim_id=claim_data.get("id", i + 1),
                        text=claim_data.get("text", claim_data.get("claim", "")),
                        flagged=claim_data.get("flagged", claim_data.get("is_flagged", False)),
                        confidence=claim_data.get("confidence", 0.0)
                    )

                    # Parse source if present
                    source = claim_data.get("source", claim_data.get("citation"))
                    if source:
                        claim.source = source if isinstance(source, dict) else {"citation_id": source}

                    # Parse stakeholder assessments for this claim
                    assessments = claim_data.get("stakeholder_assessments", {})
                    if assessments:
                        claim.stakeholder_assessments = assessments

                    claims.append(claim)
                elif isinstance(claim_data, str):
                    # Simple string claim
                    claims.append(Claim(
                        claim_id=i + 1,
                        text=claim_data,
                        flagged=False
                    ))

        return claims

    def _parse_stakeholder_analysis(self, response: Any) -> Dict[str, Dict[str, Any]]:
        """Parse stakeholder analysis from API response"""
        if isinstance(response, list):
            # Response is a list of claims, stakeholder data might be nested
            return {}
        elif isinstance(response, dict):
            data = response.get("data", response)
            if isinstance(data, list):
                return {}
            stakeholder_data = data.get("stakeholder_analysis", data.get("stakeholders", {}))
        else:
            return {}

        if not isinstance(stakeholder_data, dict):
            return {}

        analysis = {}
        for stakeholder in self.STAKEHOLDERS:
            if stakeholder in stakeholder_data:
                analysis[stakeholder] = stakeholder_data[stakeholder]
            else:
                # Try variations of the key
                alt_key = stakeholder.replace("_", " ").title().replace(" ", "_")
                if alt_key in stakeholder_data:
                    analysis[stakeholder] = stakeholder_data[alt_key]

        return analysis

    def _parse_trust_score(
        self,
        response: Any,
        claims: List[Claim]
    ) -> TrustScore:
        """Parse trust score from API response"""
        # Handle None or non-dict responses
        if not isinstance(response, dict):
            data = {}
        else:
            data = response.get("data")
            if data is None:
                data = response  # Fall back to response itself
            if not isinstance(data, dict):
                data = {}

        # Calculate derived metrics from claims
        total_claims = len(claims)
        sourced_claims = sum(1 for c in claims if c.source is not None)
        flagged_claims = sum(1 for c in claims if c.flagged)

        return TrustScore(
            overall=data.get("trust_score", data.get("score", 0.0)) if data else 0.0,
            sourced_claims_ratio=sourced_claims / total_claims if total_claims > 0 else 0.0,
            flagged_claims_count=flagged_claims,
            raw_data=data if data else {}
        )

    def _determine_overall_status(self, claims: List[Claim]) -> str:
        """Determine overall status based on claims"""
        if not claims:
            return "NO_CLAIMS"

        flagged_count = sum(1 for c in claims if c.flagged)

        if flagged_count == 0:
            return "VERIFIED"
        elif flagged_count == len(claims):
            return "ALL_FLAGGED"
        else:
            return "FLAGGED"


def test_client():
    """Test the explainability client with a sample query"""
    client = ExplainabilityClient()

    # Example IDs from a recent benchmark run
    # These would be replaced with actual values from provider_requests.jsonl
    project_id = os.environ.get("CUSTOMGPT_PROJECT", "81643")
    session_id = "test-session-id"
    prompt_id = "test-prompt-id"

    print(f"Testing explainability client...")
    print(f"Project: {project_id}")
    print(f"Session: {session_id}")
    print(f"Prompt: {prompt_id}")

    result = client.get_explainability(project_id, session_id, prompt_id)

    print(f"\nResult:")
    print(f"  Claims: {len(result.claims)}")
    print(f"  Trust Score: {result.trust_score}")
    print(f"  Overall Status: {result.overall_status}")
    print(f"  Error: {result.error}")

    return result


if __name__ == "__main__":
    test_client()
