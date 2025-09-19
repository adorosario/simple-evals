"""
ScrapingBee URL Fetcher
Enhanced fetching with JavaScript rendering, premium proxies, and anti-bot detection
Used as fallback for URLs that fail with regular HTTP requests
"""

import os
import requests
import time
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from urllib.parse import quote

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ScrapingBeeResult:
    """Result of a ScrapingBee fetch operation"""
    url: str
    success: bool
    content: Optional[str] = None
    content_type: Optional[str] = None
    status_code: Optional[int] = None
    error_message: Optional[str] = None
    response_time: Optional[float] = None
    api_cost: Optional[int] = None  # API credits used


class ScrapingBeeFetcher:
    """
    ScrapingBee API fetcher with JavaScript rendering and premium proxy support.
    Used as fallback for difficult URLs that fail with regular HTTP requests.
    """

    def __init__(self,
                 api_key: str = None,
                 enable_js: bool = True,
                 premium_proxy: bool = True,
                 stealth_proxy: bool = False,
                 timeout: int = 30):
        """
        Initialize ScrapingBee fetcher.

        Args:
            api_key: ScrapingBee API key (defaults to env var)
            enable_js: Enable JavaScript rendering (costs more credits)
            premium_proxy: Use premium proxy pool (costs more but higher success)
            stealth_proxy: Use stealth proxy (most expensive but highest success)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.environ.get('SCRAPINGBEE_API_KEY')
        if not self.api_key:
            raise ValueError("ScrapingBee API key required. Set SCRAPINGBEE_API_KEY environment variable.")

        self.enable_js = enable_js
        self.premium_proxy = premium_proxy
        self.stealth_proxy = stealth_proxy
        self.timeout = timeout
        self.base_url = "https://app.scrapingbee.com/api/v1/"

    def fetch(self, url: str, **kwargs) -> ScrapingBeeResult:
        """
        Fetch URL content using ScrapingBee API.

        Args:
            url: URL to fetch
            **kwargs: Additional ScrapingBee parameters

        Returns:
            ScrapingBeeResult with content and metadata
        """
        start_time = time.time()

        try:
            # Build ScrapingBee parameters
            params = {
                'api_key': self.api_key,
                'url': url,
                'timeout': self.timeout * 1000,  # ScrapingBee expects milliseconds
            }

            # Enhanced features for difficult sites
            if self.enable_js:
                params['render_js'] = 'true'
                params['wait'] = 3000  # Wait 3s for JS to load

            if self.stealth_proxy:
                params['stealth_proxy'] = 'true'
            elif self.premium_proxy:
                params['premium_proxy'] = 'true'

            # Additional anti-bot measures
            params.update({
                'block_ads': 'true',          # Block ads for faster loading
                'block_resources': 'false',   # Don't block CSS/images (might be needed)
                'window_width': 1920,         # Desktop viewport
                'window_height': 1080,
                'wait_for': '',               # Don't wait for specific elements
            })

            # Override with any custom parameters
            params.update(kwargs)

            logger.debug(f"ScrapingBee request: {url} (JS: {self.enable_js}, Premium: {self.premium_proxy})")

            # Make API request
            response = requests.get(
                self.base_url,
                params=params,
                timeout=self.timeout + 10  # Add buffer for API processing
            )

            response_time = time.time() - start_time

            # Check for API errors
            if response.status_code != 200:
                error_msg = f"ScrapingBee API error: {response.status_code}"
                if response.text:
                    try:
                        error_data = response.json()
                        error_msg += f" - {error_data.get('message', response.text[:200])}"
                    except:
                        error_msg += f" - {response.text[:200]}"

                return ScrapingBeeResult(
                    url=url,
                    success=False,
                    error_message=error_msg,
                    response_time=response_time,
                    status_code=response.status_code
                )

            # Extract metadata from headers
            api_cost = response.headers.get('spb-cost')
            original_status = response.headers.get('spb-original-status')

            # Check if the original website returned an error
            if original_status and int(original_status) >= 400:
                return ScrapingBeeResult(
                    url=url,
                    success=False,
                    error_message=f"Website returned HTTP {original_status}",
                    response_time=response_time,
                    status_code=int(original_status),
                    api_cost=int(api_cost) if api_cost else None
                )

            # Success!
            content = response.text
            content_type = response.headers.get('content-type', 'text/html')

            # Basic content validation
            if not content or len(content.strip()) < 100:
                return ScrapingBeeResult(
                    url=url,
                    success=False,
                    error_message="Content too short or empty",
                    response_time=response_time,
                    api_cost=int(api_cost) if api_cost else None
                )

            logger.info(f"ScrapingBee success: {url} ({len(content)} chars, {api_cost} credits)")

            return ScrapingBeeResult(
                url=url,
                success=True,
                content=content,
                content_type=content_type,
                status_code=int(original_status) if original_status else 200,
                response_time=response_time,
                api_cost=int(api_cost) if api_cost else None
            )

        except requests.exceptions.Timeout:
            return ScrapingBeeResult(
                url=url,
                success=False,
                error_message="ScrapingBee API timeout",
                response_time=time.time() - start_time
            )

        except requests.exceptions.RequestException as e:
            return ScrapingBeeResult(
                url=url,
                success=False,
                error_message=f"ScrapingBee API request failed: {str(e)}",
                response_time=time.time() - start_time
            )

        except Exception as e:
            return ScrapingBeeResult(
                url=url,
                success=False,
                error_message=f"ScrapingBee unexpected error: {str(e)}",
                response_time=time.time() - start_time
            )

    def get_remaining_credits(self) -> Optional[int]:
        """
        Check remaining API credits.

        Returns:
            Number of remaining credits or None if unable to check
        """
        try:
            response = requests.get(
                "https://app.scrapingbee.com/api/v1/usage",
                params={'api_key': self.api_key},
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                return data.get('max_api_credit', 0) - data.get('used_api_credit', 0)

        except Exception as e:
            logger.warning(f"Could not check ScrapingBee credits: {e}")

        return None