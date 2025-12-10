"""
URL Cleaner for SimpleQA-Verified Knowledge Base Builder

Handles malformed URLs from the SimpleQA-Verified dataset including:
- Chrome extension artifacts
- Malformed protocol prefixes
- Missing closing parentheses
- Garbage prefixes
"""

import re
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from urllib.parse import urlparse, urlunparse
import json
from pathlib import Path


@dataclass
class CleaningResult:
    """Result of cleaning a single URL."""
    original_url: str
    cleaned_url: Optional[str]
    fix_applied: Optional[str]
    success: bool
    error: Optional[str] = None


@dataclass
class CleaningReport:
    """Aggregate report of URL cleaning operation."""
    total_urls: int = 0
    successful: int = 0
    failed: int = 0
    already_valid: int = 0
    fixes_by_type: dict = field(default_factory=dict)
    results: List[CleaningResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'summary': {
                'total_urls': self.total_urls,
                'successful': self.successful,
                'failed': self.failed,
                'already_valid': self.already_valid,
                'success_rate': self.successful / max(self.total_urls, 1),
            },
            'fixes_by_type': self.fixes_by_type,
            'failed_urls': [
                {'url': r.original_url, 'error': r.error}
                for r in self.results if not r.success
            ],
            'fixed_urls': [
                {'original': r.original_url, 'cleaned': r.cleaned_url, 'fix': r.fix_applied}
                for r in self.results if r.success and r.fix_applied
            ]
        }


class URLCleaner:
    """
    Sanitizes and fixes malformed URLs from SimpleQA-Verified dataset.

    Known issues handled:
    1. chrome-extension://... artifacts
    2. ://'https://...' malformed protocol
    3. hillhttps://... garbage prefix
    4. Missing closing parentheses in Wikipedia URLs
    5. Trailing garbage after valid URLs
    """

    # Patterns for common issues (ordered by specificity)
    PATTERNS = {
        # chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://...
        'chrome_extension': re.compile(
            r'^chrome-extension://[^/]+/(https?://.+)$',
            re.IGNORECASE
        ),

        # ://'https://en.wikipedia.org/...'
        'protocol_prefix_quoted': re.compile(
            r"^:/?/?'?(https?://[^']+)'?$",
            re.IGNORECASE
        ),

        # hillhttps://... or any garbage before https://
        'garbage_prefix': re.compile(
            r'^[^:]+?(https?://)',
            re.IGNORECASE
        ),

        # Detect Wikipedia URLs missing closing paren
        'wikipedia_incomplete_paren': re.compile(
            r'https?://[^/]*wikipedia\.org/wiki/[^)]+\([^)]*$',
            re.IGNORECASE
        ),

        # General incomplete parentheses
        'missing_close_paren': re.compile(r'\([^)]+$'),

        # Trailing parentheses that shouldn't be there
        'trailing_paren': re.compile(r'\)+$'),
    }

    def clean(self, url: str) -> CleaningResult:
        """
        Clean a potentially malformed URL.

        Args:
            url: Raw URL string from dataset

        Returns:
            CleaningResult with cleaned URL or error
        """
        url = url.strip()

        # Handle empty URLs
        if not url:
            return CleaningResult(url, None, None, False, "Empty URL")

        # Check if already valid
        if self._validate_url(url):
            return CleaningResult(url, url, None, True)

        # Try each fix in order of specificity
        fixes = [
            ('chrome_extension', self._fix_chrome_extension),
            ('protocol_prefix_quoted', self._fix_protocol_prefix),
            ('garbage_prefix', self._fix_garbage_prefix),
            ('missing_close_paren', self._fix_missing_paren),
            ('trailing_garbage', self._fix_trailing_garbage),
        ]

        for fix_name, fix_func in fixes:
            result = fix_func(url)
            if result:
                cleaned_url = result
                if self._validate_url(cleaned_url):
                    return CleaningResult(url, cleaned_url, fix_name, True)

        # No fix worked
        return CleaningResult(url, None, None, False, "Unable to fix URL")

    def clean_batch(self, urls: List[str]) -> CleaningReport:
        """
        Clean a batch of URLs and generate a report.

        Args:
            urls: List of raw URL strings

        Returns:
            CleaningReport with all results and statistics
        """
        report = CleaningReport()
        report.total_urls = len(urls)

        for url in urls:
            result = self.clean(url)
            report.results.append(result)

            if result.success:
                report.successful += 1
                if result.fix_applied:
                    report.fixes_by_type[result.fix_applied] = \
                        report.fixes_by_type.get(result.fix_applied, 0) + 1
                else:
                    report.already_valid += 1
            else:
                report.failed += 1

        return report

    def _fix_chrome_extension(self, url: str) -> Optional[str]:
        """Extract embedded URL from Chrome extension artifact."""
        match = self.PATTERNS['chrome_extension'].match(url)
        if match:
            return match.group(1)
        return None

    def _fix_protocol_prefix(self, url: str) -> Optional[str]:
        """Fix URLs with malformed protocol prefix like ://'https://...'"""
        match = self.PATTERNS['protocol_prefix_quoted'].match(url)
        if match:
            cleaned = match.group(1).rstrip("'\"")
            return cleaned
        return None

    def _fix_garbage_prefix(self, url: str) -> Optional[str]:
        """Remove non-URL prefixes like 'hill' in 'hillhttps://'."""
        # Only apply if URL doesn't start with valid protocol
        if url.startswith(('http://', 'https://')):
            return None

        match = self.PATTERNS['garbage_prefix'].search(url)
        if match:
            # Extract from where https:// starts
            start_pos = match.start(1)
            return url[start_pos:]
        return None

    def _fix_missing_paren(self, url: str) -> Optional[str]:
        """Add missing closing parenthesis to Wikipedia-style URLs."""
        # Count open vs close parens
        open_count = url.count('(')
        close_count = url.count(')')

        if open_count > close_count:
            # Add missing closing parens
            return url + ')' * (open_count - close_count)
        return None

    def _fix_trailing_garbage(self, url: str) -> Optional[str]:
        """Remove trailing garbage like extra parentheses."""
        # Try removing trailing parens if they make URL invalid
        cleaned = self.PATTERNS['trailing_paren'].sub('', url)
        if cleaned != url and self._validate_url(cleaned):
            return cleaned
        return None

    def _validate_url(self, url: str) -> bool:
        """
        Validate that URL is well-formed.

        Args:
            url: URL string to validate

        Returns:
            True if URL has valid scheme and netloc
        """
        try:
            parsed = urlparse(url)
            # Must have http/https scheme and a network location (domain)
            return all([
                parsed.scheme in ('http', 'https'),
                parsed.netloc,
                len(parsed.netloc) > 2,  # At least "x.y"
            ])
        except Exception:
            return False

    def save_report(self, report: CleaningReport, output_path: Path) -> None:
        """Save cleaning report to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)


def extract_urls_from_csv_row(urls_field: str) -> List[str]:
    """
    Extract individual URLs from comma-separated URL field.

    SimpleQA-Verified stores URLs as: "url1,url2,url3"

    Args:
        urls_field: Raw URLs string from CSV

    Returns:
        List of individual URL strings
    """
    if not urls_field or urls_field == 'nan':
        return []

    # Split by comma, but be careful of URLs containing commas (rare)
    urls = []
    current_url = ""

    for part in str(urls_field).split(','):
        part = part.strip()

        # If this part starts with http, it's a new URL
        if part.startswith(('http://', 'https://')) or not current_url:
            if current_url:
                urls.append(current_url)
            current_url = part
        else:
            # This is continuation of previous URL (has comma in it)
            current_url += ',' + part

    if current_url:
        urls.append(current_url)

    return [u.strip() for u in urls if u.strip()]


if __name__ == "__main__":
    # Test with known problematic URLs
    test_urls = [
        "://'https://en.wikipedia.org/wiki/ben_m._bogard'",
        "hillhttps://en.wikipedia.org/wiki/margaret_gardiner_(art_collector",
        "chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://www.ams.org/notices/199808/comm-fulkerson.pdf",
        "https://en.wikipedia.org/wiki/douglas_bennett_(cricketer",
        "https://en.wikipedia.org/wiki/Stella_Obasanjo",  # Valid URL
        "",  # Empty
    ]

    cleaner = URLCleaner()

    print("URL Cleaner Test Results")
    print("=" * 60)

    for url in test_urls:
        result = cleaner.clean(url)
        print(f"\nOriginal: {url[:60]}...")
        print(f"Success:  {result.success}")
        if result.success:
            print(f"Cleaned:  {result.cleaned_url[:60]}...")
            if result.fix_applied:
                print(f"Fix:      {result.fix_applied}")
        else:
            print(f"Error:    {result.error}")
