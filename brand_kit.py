"""
Unified Brand Kit for SimpleEvals RAG Benchmark Dashboard
=========================================================

Apple-inspired design system for consistent, beautiful reporting.
Used across all HTML report generation (quality benchmarks, forensics, statistical analysis).

Design Philosophy (Apple-inspired):
- Minimalism: Remove unnecessary elements, focus on content
- Clarity: Typography and layout optimized for readability
- Depth: Subtle shadows and layering create visual hierarchy
- Delight: Smooth animations and polished interactions
- Consistency: Every element follows the same design language
- Accessibility: WCAG AA compliant, works on all devices
"""

from datetime import datetime
from typing import Dict, List, Optional


# ============================================================================
# BRAND COLORS & DESIGN TOKENS
# ============================================================================

BRAND_COLORS = {
    # Primary academic palette
    "primary": "#1e40af",        # Deep blue - professional, trustworthy
    "primary_light": "#3b82f6",  # Lighter blue for accents
    "primary_dark": "#1e3a8a",   # Darker blue for headers

    # Semantic colors
    "success": "#10b981",        # Green for correct answers
    "warning": "#f59e0b",        # Amber for warnings
    "danger": "#ef4444",         # Red for errors/penalties
    "info": "#06b6d4",           # Cyan for informational

    # Strategy-specific colors
    "quality": "#8b5cf6",        # Purple for quality metrics
    "volume": "#06b6d4",         # Cyan for volume metrics

    # Neutral palette
    "text_primary": "#1e293b",   # Dark slate for body text
    "text_secondary": "#64748b", # Medium slate for secondary text
    "bg_light": "#f8fafc",       # Almost white background
    "bg_white": "#ffffff",       # Pure white
    "border": "#e2e8f0",         # Light border color

    # Provider-specific badges
    "customgpt": "#10b981",      # Green
    "openai_rag": "#3b82f6",     # Blue
    "openai_vanilla": "#f59e0b", # Amber
}


# ============================================================================
# HTML HEAD & DEPENDENCIES
# ============================================================================

def get_html_head(title: str, description: str = "") -> str:
    """
    Generate consistent HTML head with all dependencies.

    Args:
        title: Page title for <title> tag
        description: Optional meta description for SEO/sharing

    Returns:
        Complete HTML head section with CDN dependencies
    """
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>

    <!-- External Dependencies -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.datatables.net/1.13.6/css/dataTables.bootstrap5.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">

    <!-- JavaScript Dependencies -->
    <script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.6/js/dataTables.bootstrap5.min.js"></script>

    <!-- Open Graph / Social Media Meta Tags -->
    <meta property="og:title" content="{title}">
    <meta property="og:description" content="{description or 'Academic RAG Benchmark Analysis - Why RAGs Hallucinate'}">
    <meta property="og:type" content="website">

{get_unified_css()}
</head>"""


# ============================================================================
# UNIFIED CSS STYLING
# ============================================================================

def get_unified_css() -> str:
    """
    Generate unified CSS for all dashboard pages.
    Apple-inspired design with smooth animations and refined details.
    """
    return f"""    <style>
        /* ============================================
           BRAND KIT - Apple-Inspired Design System
           ============================================ */

        :root {{
            /* Brand colors */
            --primary: {BRAND_COLORS['primary']};
            --primary-light: {BRAND_COLORS['primary_light']};
            --primary-dark: {BRAND_COLORS['primary_dark']};
            --success: {BRAND_COLORS['success']};
            --warning: {BRAND_COLORS['warning']};
            --danger: {BRAND_COLORS['danger']};
            --info: {BRAND_COLORS['info']};
            --quality: {BRAND_COLORS['quality']};
            --volume: {BRAND_COLORS['volume']};

            /* Text colors */
            --text-primary: {BRAND_COLORS['text_primary']};
            --text-secondary: {BRAND_COLORS['text_secondary']};

            /* Backgrounds */
            --bg-light: {BRAND_COLORS['bg_light']};
            --bg-white: {BRAND_COLORS['bg_white']};
            --border: {BRAND_COLORS['border']};

            /* Apple-inspired shadows (softer, more refined) */
            --shadow-sm: 0 1px 3px 0 rgba(0, 0, 0, 0.04), 0 1px 2px 0 rgba(0, 0, 0, 0.02);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.08), 0 2px 4px -1px rgba(0, 0, 0, 0.04);
            --shadow-lg: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
            --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);

            /* Animation timing (Apple-like easing) */
            --transition-fast: 150ms cubic-bezier(0.4, 0, 0.2, 1);
            --transition-base: 250ms cubic-bezier(0.4, 0, 0.2, 1);
            --transition-slow: 350ms cubic-bezier(0.4, 0, 0.2, 1);

            /* Spacing rhythm (8px base grid) */
            --spacing-xs: 0.25rem;  /* 4px */
            --spacing-sm: 0.5rem;   /* 8px */
            --spacing-md: 1rem;     /* 16px */
            --spacing-lg: 1.5rem;   /* 24px */
            --spacing-xl: 2rem;     /* 32px */
            --spacing-2xl: 3rem;    /* 48px */

            /* Border radius (consistent with Apple's design) */
            --radius-sm: 6px;
            --radius-md: 10px;
            --radius-lg: 14px;
            --radius-xl: 18px;
        }}

        /* ============================================
           BASE STYLES - Apple Typography
           ============================================ */

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'SF Pro Text',
                         'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
            background-color: var(--bg-light);
            color: var(--text-primary);
            line-height: 1.5;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
            font-feature-settings: 'kern' 1, 'liga' 1, 'calt' 1;
        }}

        * {{
            box-sizing: border-box;
        }}

        /* Smooth scrolling */
        html {{
            scroll-behavior: smooth;
        }}

        /* ============================================
           NAVIGATION BAR
           ============================================ */

        .brand-navbar {{
            background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary) 100%);
            border-bottom: 3px solid var(--primary-light);
            box-shadow: var(--shadow-md);
            padding: 1rem 0;
        }}

        .brand-navbar .navbar-brand {{
            color: white !important;
            font-weight: 700;
            font-size: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }}

        .brand-navbar .nav-link {{
            color: rgba(255, 255, 255, 0.9) !important;
            font-weight: 500;
            padding: 0.5rem 1rem !important;
            transition: all var(--transition-base);
            border-radius: var(--radius-md);
        }}

        .brand-navbar .nav-link:hover {{
            background: rgba(255, 255, 255, 0.15);
            color: white !important;
            transform: translateY(-1px);
        }}

        .brand-navbar .nav-link.active {{
            background: rgba(255, 255, 255, 0.25);
            color: white !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
        }}

        /* ============================================
           MAIN CONTAINER
           ============================================ */

        .main-container {{
            background: var(--bg-white);
            margin: var(--spacing-xl) auto;
            max-width: 1400px;
            border-radius: var(--radius-xl);
            box-shadow: var(--shadow-xl);
            overflow: hidden;
            animation: fadeInUp var(--transition-slow) ease-out;
        }}

        @keyframes fadeInUp {{
            from {{
                opacity: 0;
                transform: translateY(20px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        .page-header {{
            background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary) 100%);
            color: white;
            padding: 3rem 2rem;
            text-align: center;
        }}

        .page-header h1 {{
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0 0 0.75rem 0;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }}

        .page-header .subtitle {{
            font-size: 1.25rem;
            opacity: 0.95;
            margin: 0.5rem 0;
        }}

        .page-header .meta {{
            margin-top: 1rem;
            opacity: 0.85;
            font-size: 0.9rem;
        }}

        /* ============================================
           CONTENT SECTIONS
           ============================================ */

        .content-section {{
            padding: 2rem;
        }}

        .section-header {{
            color: var(--text-primary);
            font-weight: 700;
            font-size: 1.75rem;
            margin: 2rem 0 1.5rem 0;
            padding-bottom: 0.75rem;
            border-bottom: 3px solid var(--primary);
        }}

        .info-box {{
            background: var(--bg-light);
            padding: 1.5rem;
            border-radius: 8px;
            margin: 1.5rem 0;
            border-left: 4px solid var(--primary);
        }}

        /* ============================================
           METRIC CARDS
           ============================================ */

        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin: 2rem 0;
        }}

        .metric-card {{
            background: var(--bg-white);
            padding: var(--spacing-lg);
            border-radius: var(--radius-lg);
            box-shadow: var(--shadow-md);
            border-left: 4px solid var(--primary);
            transition: all var(--transition-base);
            position: relative;
            overflow: hidden;
        }}

        .metric-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, var(--primary), var(--primary-light));
            opacity: 0;
            transition: opacity var(--transition-base);
        }}

        .metric-card:hover {{
            transform: translateY(-4px) scale(1.01);
            box-shadow: var(--shadow-xl);
        }}

        .metric-card:hover::before {{
            opacity: 1;
        }}

        .metric-card h3 {{
            color: var(--primary);
            font-size: 1rem;
            font-weight: 600;
            margin: 0 0 0.75rem 0;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .metric-card .value {{
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--text-primary);
            margin: 0.5rem 0;
        }}

        .metric-card .description {{
            color: var(--text-secondary);
            font-size: 0.9rem;
            margin: 0;
        }}

        /* ============================================
           TABLES
           ============================================ */

        .table-responsive {{
            margin: 2rem 0;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: var(--shadow-sm);
        }}

        .table {{
            margin: 0;
            font-size: 0.9rem;
        }}

        .table thead th {{
            background: var(--primary-dark);
            color: white;
            font-weight: 600;
            border: none;
            padding: 1rem;
            text-transform: uppercase;
            font-size: 0.85rem;
            letter-spacing: 0.5px;
        }}

        .table tbody tr {{
            transition: background-color 0.15s ease;
        }}

        .table tbody tr:hover {{
            background-color: var(--bg-light);
        }}

        .table tbody td {{
            padding: 1rem;
            vertical-align: middle;
            border-color: var(--border);
        }}

        /* ============================================
           BADGES & LABELS
           ============================================ */

        .provider-badge {{
            display: inline-block;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.9rem;
        }}

        .provider-customgpt {{
            background: #d1fae5;
            color: #065f46;
        }}

        .provider-openai-rag {{
            background: #dbeafe;
            color: #1e40af;
        }}

        .provider-openai-vanilla {{
            background: #fef3c7;
            color: #92400e;
        }}

        .grade-badge {{
            display: inline-block;
            padding: 0.35rem 0.75rem;
            border-radius: 4px;
            font-weight: 700;
            font-size: 0.85rem;
        }}

        .grade-A {{ background: #d1fae5; color: #065f46; }}
        .grade-B {{ background: #dbeafe; color: #1e40af; }}
        .grade-C {{ background: #fef3c7; color: #92400e; }}
        .grade-D {{ background: #fee2e2; color: #991b1b; }}
        .grade-F {{ background: #fecaca; color: #7f1d1d; }}

        /* ============================================
           SCORE STYLING
           ============================================ */

        .score-quality {{ color: var(--quality); font-weight: 600; }}
        .score-volume {{ color: var(--volume); font-weight: 600; }}
        .score-high {{ color: var(--success); font-weight: 600; }}
        .score-medium {{ color: var(--warning); font-weight: 600; }}
        .score-low {{ color: var(--danger); font-weight: 600; }}

        /* ============================================
           RESPONSIVE DESIGN
           ============================================ */

        @media (max-width: 768px) {{
            .main-container {{
                margin: 1rem;
                border-radius: 8px;
            }}

            .page-header {{
                padding: 2rem 1.5rem;
            }}

            .page-header h1 {{
                font-size: 2rem;
            }}

            .content-section {{
                padding: 1.5rem;
            }}

            .metric-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>"""


# ============================================================================
# NAVIGATION BAR
# ============================================================================

def get_navigation_bar(
    active_page: str = "",
    run_id: Optional[str] = None,
    base_path: str = "",
    quality_report: Optional[str] = None,
    statistical_report: Optional[str] = None,
    forensic_reports: Optional[dict] = None
) -> str:
    """
    Generate unified navigation bar for all dashboard pages.

    Args:
        active_page: One of 'home', 'quality', 'statistical', 'forensic'
        run_id: Optional run ID for linking to specific run results
        base_path: Path prefix for navigation (e.g., "../" if in subdirectory, "" if at root)
        quality_report: Actual filename of quality benchmark report (e.g., "quality_benchmark_report_20251025_115547.html")
        statistical_report: Actual filename of statistical analysis report
        forensic_reports: Dict mapping provider names to their forensic dashboard paths

    Returns:
        HTML navigation bar with links to all sub-dashboards
    """
    # Build links - use actual report paths if provided, otherwise use defaults
    home_link = f"{base_path}index.html"
    quality_link = f"{base_path}{quality_report}" if quality_report else "#"
    statistical_link = f"{base_path}{statistical_report}" if statistical_report else "#"

    # Build forensic dropdown items
    forensic_items = []
    if forensic_reports:
        # Create dropdown items for each available forensic report
        provider_display_names = {
            'customgpt': 'CustomGPT',
            'openai_rag': 'OpenAI RAG',
            'openai_vanilla': 'OpenAI Vanilla'
        }
        for provider_key, report_path in forensic_reports.items():
            display_name = provider_display_names.get(provider_key, provider_key.replace('_', ' ').title())
            forensic_items.append(
                f'<li><a class="dropdown-item" href="{base_path}{report_path}">{display_name} Forensics</a></li>'
            )

    # If no forensic reports available, show placeholder
    if not forensic_items:
        forensic_items = ['<li><a class="dropdown-item disabled" href="#">No forensic reports available</a></li>']

    active_class = lambda page: "active" if page == active_page else ""

    return f"""<nav class="navbar navbar-expand-lg brand-navbar">
    <div class="container-fluid">
        <a class="navbar-brand" href="{home_link}">
            <i class="fas fa-microscope"></i>
            Why RAGs Hallucinate
        </a>
        <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
            <span class="navbar-toggler-icon"></span>
        </button>
        <div class="collapse navbar-collapse" id="navbarNav">
            <ul class="navbar-nav ms-auto">
                <li class="nav-item">
                    <a class="nav-link {active_class('home')}" href="{home_link}">
                        <i class="fas fa-home me-1"></i>Dashboard
                    </a>
                </li>
                <li class="nav-item">
                    <a class="nav-link {active_class('quality')}" href="{quality_link}">
                        <i class="fas fa-trophy me-1"></i>Quality Benchmark
                    </a>
                </li>
                <li class="nav-item">
                    <a class="nav-link {active_class('statistical')}" href="{statistical_link}">
                        <i class="fas fa-chart-bar me-1"></i>Statistical Analysis
                    </a>
                </li>
                <li class="nav-item dropdown">
                    <a class="nav-link dropdown-toggle {active_class('forensic')}" href="#" id="forensicDropdown" role="button" data-bs-toggle="dropdown">
                        <i class="fas fa-bug me-1"></i>Forensics
                    </a>
                    <ul class="dropdown-menu dropdown-menu-dark">
                        {''.join(forensic_items)}
                    </ul>
                </li>
            </ul>
        </div>
    </div>
</nav>"""


# ============================================================================
# PAGE HEADER
# ============================================================================

def get_page_header(title: str, subtitle: str = "", meta_info: str = "") -> str:
    """
    Generate consistent page header.

    Args:
        title: Main page title
        subtitle: Optional subtitle
        meta_info: Optional metadata (timestamp, run info, etc.)

    Returns:
        HTML page header section
    """
    subtitle_html = f'<div class="subtitle">{subtitle}</div>' if subtitle else ""
    meta_html = f'<div class="meta">{meta_info}</div>' if meta_info else ""

    return f"""<div class="page-header">
    <h1><i class="fas fa-microscope me-3"></i>{title}</h1>
    {subtitle_html}
    {meta_html}
</div>"""


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_timestamp(dt: Optional[datetime] = None) -> str:
    """Format datetime for display"""
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%B %d, %Y at %H:%M:%S")


def get_provider_badge_class(provider: str) -> str:
    """Get CSS class for provider badge"""
    provider_lower = provider.lower().replace("_", "-")
    return f"provider-badge provider-{provider_lower}"


def get_grade_badge_class(grade: str) -> str:
    """Get CSS class for grade badge"""
    return f"grade-badge grade-{grade}"


def wrap_html_document(content: str) -> str:
    """Wrap content in complete HTML document structure"""
    return f"""{content}
</body>
</html>"""
