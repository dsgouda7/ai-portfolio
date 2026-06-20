"""
Domain-Specific Corpus Generators for Use Case Benchmarks

Generates realistic mock data for:
- Code repositories
- Support tickets
- Clinical notes
- Legal documents
- Research papers
- System logs
- Multilingual documentation
"""

import random
from typing import List, Dict, Tuple
from datetime import datetime, timedelta


# ============================================================================
# CODE REPOSITORY GENERATORS
# ============================================================================

CODE_FUNCTIONS = [
    "def authenticate_user(username: str, password: str) -> Optional[User]:",
    "async def handle_payment_request(request: PaymentRequest) -> PaymentResponse:",
    "class RateLimiter:",
    "def validate_token(token: str, secret: str) -> bool:",
    "async def fetch_user_profile(user_id: int) -> Dict[str, Any]:",
    "def hash_password(password: str, salt: bytes) -> bytes:",
    "class APIRouter:",
    "def log_error(message: str, context: Dict) -> None:",
    "async def send_notification(user_id: int, message: str) -> bool:",
    "def parse_request_headers(headers: Dict) -> AuthContext:"
]

CODE_PATTERNS = [
    "    # Check authentication\n    if not user.is_authenticated:\n        raise UnauthorizedException('User not authenticated')",
    "    try:\n        result = await db.execute(query)\n    except DatabaseError as e:\n        logger.error(f'Query failed: {e}')\n        raise",
    "    # Rate limiting check\n    if rate_limiter.is_exceeded(client_id):\n        return Response(status_code=429, content='Too many requests')",
    "    # Validate input\n    if not validate_email(email):\n        raise ValidationError('Invalid email format')",
    "    # Log request\n    logger.info(f'Processing request: {request_id}', extra={'user_id': user.id})"
]

def generate_code_repository(target_mb: int, name: str = "codebase") -> List[str]:
    """
    Generate mock Python codebase with functions, classes, and comments.

    Args:
        target_mb: Target size in MB
        name: Repository name

    Returns:
        List of code lines
    """
    lines = []
    lines.append(f"# {name.upper()} - Mock Code Repository")
    lines.append(f"# Generated: {datetime.now().isoformat()}")
    lines.append("")

    target_bytes = target_mb * 1024 * 1024
    current_bytes = sum(len(line.encode('utf-8')) for line in lines)

    file_count = 1

    while current_bytes < target_bytes:
        # File header
        lines.append(f"\n# ============ FILE: src/module_{file_count}.py ============")
        lines.append(f"\"\"\"Module {file_count}: Core business logic\"\"\"")
        lines.append("")
        lines.append("from typing import Optional, Dict, Any, List")
        lines.append("import logging")
        lines.append("")

        # Add 5-10 functions per file
        for _ in range(random.randint(5, 10)):
            # Function definition
            func = random.choice(CODE_FUNCTIONS)
            lines.append(func)
            lines.append('    """')
            lines.append(f'    {random.choice(["Handles user authentication", "Processes payment requests", "Validates input data", "Manages rate limiting", "Logs events"])}')
            lines.append('    """')

            # Function body
            for _ in range(random.randint(3, 8)):
                lines.append(random.choice(CODE_PATTERNS))

            lines.append("")

        file_count += 1
        current_bytes = sum(len(line.encode('utf-8')) for line in lines)

    return lines


# ============================================================================
# SUPPORT TICKET GENERATORS
# ============================================================================

TICKET_CATEGORIES = ["Login Issues", "Payment Failures", "API Errors", "Performance", "Feature Request", "Bug Report"]
TICKET_SEVERITIES = ["Critical", "High", "Medium", "Low"]
TICKET_PRODUCTS = ["Web App", "Mobile App", "API Gateway", "Payment Service", "Auth Service"]

TICKET_DESCRIPTIONS = [
    "Unable to log in after password reset. Getting 'Invalid credentials' error even with correct password.",
    "Payment processing fails with error code 21012. Transaction gets stuck in pending state.",
    "API endpoint /api/v2/users/{id} returns 504 Gateway Timeout after 30 seconds.",
    "Dashboard loads very slowly (>10s) when viewing last 30 days of data. Performance degraded significantly.",
    "Feature request: Add ability to export reports in CSV format. Current PDF export is not suitable for analysis.",
    "Mobile app crashes when uploading images larger than 5MB. Error: 'Out of memory'.",
    "Webhooks not firing for payment.completed events. Last successful webhook was 2 hours ago.",
    "Search functionality not working for special characters. Queries with '&' or '+' return empty results."
]

def generate_support_tickets(target_mb: int) -> List[str]:
    """
    Generate mock support ticket corpus.

    Args:
        target_mb: Target size in MB

    Returns:
        List of ticket lines
    """
    lines = []
    lines.append("# SUPPORT TICKETS CORPUS")
    lines.append(f"# Generated: {datetime.now().isoformat()}")
    lines.append("")

    target_bytes = target_mb * 1024 * 1024
    current_bytes = sum(len(line.encode('utf-8')) for line in lines)

    ticket_id = 1000
    base_date = datetime.now() - timedelta(days=365)

    while current_bytes < target_bytes:
        ticket_date = base_date + timedelta(days=random.randint(0, 365), hours=random.randint(0, 23))

        lines.append(f"\n--- TICKET #{ticket_id} ---")
        lines.append(f"Date: {ticket_date.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Category: {random.choice(TICKET_CATEGORIES)}")
        lines.append(f"Severity: {random.choice(TICKET_SEVERITIES)}")
        lines.append(f"Product: {random.choice(TICKET_PRODUCTS)}")
        lines.append(f"Customer ID: CUST-{random.randint(10000, 99999)}")
        lines.append(f"Status: {random.choice(['Open', 'In Progress', 'Resolved', 'Closed'])}")
        lines.append("")
        lines.append("Description:")
        lines.append(random.choice(TICKET_DESCRIPTIONS))
        lines.append("")

        # Add resolution if resolved
        if random.random() > 0.3:
            lines.append("Resolution:")
            lines.append(f"Issue resolved by {random.choice(['restarting service', 'clearing cache', 'updating configuration', 'deploying hotfix', 'escalating to engineering'])}. ")
            lines.append(f"Root cause: {random.choice(['database timeout', 'memory leak', 'invalid configuration', 'race condition', 'network partition'])}.")
            lines.append("")

        ticket_id += 1
        current_bytes = sum(len(line.encode('utf-8')) for line in lines)

    return lines


# ============================================================================
# CLINICAL NOTES GENERATORS
# ============================================================================

MEDICAL_CONDITIONS = ["Type 2 Diabetes", "Hypertension", "COPD", "Asthma", "CAD", "CHF", "Atrial Fibrillation"]
MEDICATIONS = ["Metformin", "Lisinopril", "Atorvastatin", "Aspirin", "Albuterol", "Warfarin", "Furosemide"]
ALLERGIES = ["Penicillin", "Sulfa drugs", "NSAIDs", "Latex", "Shellfish"]

def generate_clinical_notes(target_mb: int) -> List[str]:
    """
    Generate mock clinical notes corpus (MIMIC-III style).

    Args:
        target_mb: Target size in MB

    Returns:
        List of clinical note lines
    """
    lines = []
    lines.append("# CLINICAL NOTES CORPUS (Mock MIMIC-III Format)")
    lines.append(f"# Generated: {datetime.now().isoformat()}")
    lines.append("# DISCLAIMER: Synthetic data for research purposes only")
    lines.append("")

    target_bytes = target_mb * 1024 * 1024
    current_bytes = sum(len(line.encode('utf-8')) for line in lines)

    patient_id = 10000
    base_date = datetime.now() - timedelta(days=3650)  # 10 years

    while current_bytes < target_bytes:
        encounter_date = base_date + timedelta(days=random.randint(0, 3650))

        lines.append(f"\n========== PATIENT MRN: {patient_id} | ENCOUNTER: {encounter_date.strftime('%Y-%m-%d')} ==========")
        lines.append(f"Age: {random.randint(45, 85)} | Gender: {random.choice(['M', 'F'])}")
        lines.append("")

        lines.append("CHIEF COMPLAINT:")
        lines.append(random.choice([
            "Shortness of breath, worsening over past 3 days",
            "Chest pain, substernal, radiating to left arm",
            "Elevated blood glucose readings at home",
            "Chronic cough with purulent sputum",
            "Dizziness and palpitations"
        ]))
        lines.append("")

        lines.append("HISTORY OF PRESENT ILLNESS:")
        lines.append(f"Patient is a {random.randint(45, 85)}-year-old {random.choice(['male', 'female'])} with history of {', '.join(random.sample(MEDICAL_CONDITIONS, random.randint(2, 4)))}. ")
        lines.append("Presents with acute exacerbation of chronic condition. Symptoms began 3 days prior to admission.")
        lines.append("")

        lines.append("MEDICATIONS:")
        for med in random.sample(MEDICATIONS, random.randint(3, 6)):
            lines.append(f"  - {med} {random.choice(['10mg', '20mg', '40mg', '80mg'])} {random.choice(['daily', 'BID', 'TID'])}")
        lines.append("")

        lines.append("ALLERGIES:")
        if random.random() > 0.5:
            lines.append(f"  - {random.choice(ALLERGIES)} (reaction: {random.choice(['rash', 'anaphylaxis', 'GI upset'])})")
        else:
            lines.append("  - NKDA (No Known Drug Allergies)")
        lines.append("")

        lines.append("PHYSICAL EXAM:")
        lines.append(f"  Vitals: BP {random.randint(110, 160)}/{random.randint(70, 100)}, HR {random.randint(60, 110)}, RR {random.randint(12, 24)}, O2 sat {random.randint(88, 100)}%")
        lines.append(f"  General: {random.choice(['Alert and oriented x3', 'Appears uncomfortable', 'Mild distress'])}")
        lines.append("")

        lines.append("ASSESSMENT & PLAN:")
        lines.append(f"1. {random.choice(MEDICAL_CONDITIONS)} - Continue current medications, monitor closely")
        lines.append(f"2. {random.choice(['Admit for observation', 'Discharge home with follow-up', 'Transfer to ICU'])}")
        lines.append("")

        patient_id += random.randint(1, 5)
        current_bytes = sum(len(line.encode('utf-8')) for line in lines)

    return lines


# ============================================================================
# LEGAL DOCUMENT GENERATORS
# ============================================================================

CONTRACT_CLAUSES = [
    "INDEMNIFICATION: Party A agrees to indemnify and hold harmless Party B from any claims, damages, or losses arising from breach of this agreement.",
    "FORCE MAJEURE: Neither party shall be liable for failure to perform obligations due to circumstances beyond reasonable control, including acts of God, war, or government action.",
    "CONFIDENTIALITY: All information disclosed pursuant to this agreement shall remain confidential and shall not be disclosed to third parties without prior written consent.",
    "TERMINATION: Either party may terminate this agreement with 30 days written notice. Upon termination, all obligations under Section 4 shall survive.",
    "GOVERNING LAW: This agreement shall be governed by and construed in accordance with the laws of the State of Delaware, without regard to conflict of law provisions.",
    "DISPUTE RESOLUTION: Any disputes arising under this agreement shall be resolved through binding arbitration in accordance with AAA rules.",
    "LIMITATION OF LIABILITY: In no event shall either party's liability exceed the total fees paid under this agreement in the preceding 12 months.",
]

def generate_legal_documents(target_mb: int) -> List[str]:
    """
    Generate mock legal document corpus (contracts, emails).

    Args:
        target_mb: Target size in MB

    Returns:
        List of document lines
    """
    lines = []
    lines.append("# LEGAL DOCUMENT CORPUS (Mock Enron-style)")
    lines.append(f"# Generated: {datetime.now().isoformat()}")
    lines.append("")

    target_bytes = target_mb * 1024 * 1024
    current_bytes = sum(len(line.encode('utf-8')) for line in lines)

    doc_id = 1
    base_date = datetime.now() - timedelta(days=1825)  # 5 years

    while current_bytes < target_bytes:
        doc_date = base_date + timedelta(days=random.randint(0, 1825))

        if random.random() > 0.3:
            # Email
            lines.append(f"\n========== EMAIL DOC-{doc_id} ==========")
            lines.append(f"Date: {doc_date.strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"From: {random.choice(['john.doe', 'jane.smith', 'bob.johnson', 'alice.williams'])}@company.com")
            lines.append(f"To: {random.choice(['legal', 'finance', 'operations', 'executive'])}@company.com")
            lines.append(f"Subject: {random.choice(['Project Falcon Update', 'Q4 Financial Review', 'Contract Amendment Discussion', 'Compliance Matter'])}")
            lines.append("")
            lines.append(random.choice([
                "Per our discussion this morning, I'm forwarding the latest contract amendments for your review. Please note the changes to Section 3.2 regarding indemnification.",
                "Following up on the Project Falcon meeting. We need to address the force majeure clause before proceeding with vendor negotiations.",
                "The compliance team has flagged several issues with the existing agreements. See attached memo for details.",
                "Quick question on the termination provisions in the ABC Corp contract. Do we have standard 30-day notice language?"
            ]))
            lines.append("")
        else:
            # Contract
            lines.append(f"\n========== CONTRACT DOC-{doc_id} ==========")
            lines.append(f"Date: {doc_date.strftime('%Y-%m-%d')}")
            lines.append(f"Parties: Company Inc. AND {random.choice(['Vendor Corp', 'Supplier LLC', 'Partner Industries', 'Service Provider Ltd'])}")
            lines.append(f"Type: {random.choice(['Master Service Agreement', 'Purchase Agreement', 'License Agreement', 'Non-Disclosure Agreement'])}")
            lines.append("")

            for _ in range(random.randint(3, 6)):
                lines.append(random.choice(CONTRACT_CLAUSES))
                lines.append("")

        doc_id += 1
        current_bytes = sum(len(line.encode('utf-8')) for line in lines)

    return lines


# ============================================================================
# RESEARCH PAPER GENERATORS
# ============================================================================

PAPER_TITLES = [
    "Attention Is All You Need: Transformer Architecture for Neural Machine Translation",
    "Deep Residual Learning for Image Recognition with Skip Connections",
    "Generative Adversarial Networks: A Framework for Estimating Generative Models",
    "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks",
    "Vision Transformer: An Image is Worth 16x16 Words",
    "GPT-3: Language Models are Few-Shot Learners",
    "Self-Supervised Learning: A Survey of Recent Advances"
]

PAPER_SECTIONS = [
    ("Abstract", "We propose a novel architecture that achieves state-of-the-art results on multiple benchmarks. Our approach outperforms previous methods by significant margins while reducing computational requirements."),
    ("Introduction", "Recent advances in deep learning have enabled remarkable progress in computer vision and natural language processing. However, existing approaches face challenges in scalability and generalization."),
    ("Related Work", "Prior work in this domain includes [Vaswani et al., 2017] who introduced the Transformer architecture, and [Dosovitskiy et al., 2020] who applied self-attention to vision tasks."),
    ("Methodology", "Our approach consists of three main components: (1) a multi-head attention mechanism, (2) position-wise feed-forward networks, and (3) residual connections with layer normalization."),
    ("Experiments", "We evaluate our method on ImageNet-1K, COCO detection, and ADE20K segmentation. Results demonstrate 2.3% improvement in top-1 accuracy over previous state-of-the-art."),
    ("Conclusion", "We presented a novel architecture that achieves competitive results with fewer parameters. Future work includes extending this approach to multimodal settings.")
]

def generate_research_papers(target_mb: int) -> List[str]:
    """
    Generate mock research paper corpus (ArXiv style).

    Args:
        target_mb: Target size in MB

    Returns:
        List of paper lines
    """
    lines = []
    lines.append("# RESEARCH PAPERS CORPUS (Mock ArXiv Format)")
    lines.append(f"# Generated: {datetime.now().isoformat()}")
    lines.append("")

    target_bytes = target_mb * 1024 * 1024
    current_bytes = sum(len(line.encode('utf-8')) for line in lines)

    paper_id = 2017

    while current_bytes < target_bytes:
        year = paper_id
        month = random.randint(1, 12)

        lines.append(f"\n========== PAPER arXiv:{year}.{month:02d}.{random.randint(1000, 9999)} ==========")
        lines.append(f"Title: {random.choice(PAPER_TITLES)}")
        lines.append(f"Authors: {', '.join([f'Author{i}' for i in range(random.randint(2, 5))])}")
        lines.append(f"Published: {year}-{month:02d}")
        lines.append(f"Citations: {random.randint(10, 5000)}")
        lines.append("")

        for section, content in PAPER_SECTIONS:
            lines.append(f"## {section}")
            lines.append(content)
            lines.append("")

        lines.append("## References")
        for i in range(random.randint(10, 30)):
            lines.append(f"[{i+1}] Author et al. ({year - random.randint(0, 10)}). Title of Referenced Paper. Conference/Journal.")
        lines.append("")

        paper_id += 1
        current_bytes = sum(len(line.encode('utf-8')) for line in lines)

    return lines


# ============================================================================
# SYSTEM LOG GENERATORS
# ============================================================================

LOG_SERVICES = ["api-gateway", "payment-service", "auth-service", "user-service", "notification-service"]
LOG_LEVELS = ["INFO", "WARN", "ERROR", "DEBUG"]
LOG_MESSAGES = [
    "Request processed successfully",
    "Database query timeout after 5000ms",
    "Rate limit exceeded for client {client_id}",
    "Authentication token expired",
    "Payment processing failed with error code 21012",
    "High latency detected: p95=2340ms",
    "Cache miss for key user:{user_id}",
    "Upstream service unavailable: connection refused"
]

def generate_system_logs(target_mb: int) -> List[str]:
    """
    Generate mock system logs corpus (Kubernetes/microservices style).

    Args:
        target_mb: Target size in MB

    Returns:
        List of log lines
    """
    lines = []
    lines.append("# SYSTEM LOGS CORPUS (Mock Kubernetes Format)")
    lines.append(f"# Generated: {datetime.now().isoformat()}")
    lines.append("")

    target_bytes = target_mb * 1024 * 1024
    current_bytes = sum(len(line.encode('utf-8')) for line in lines)

    base_time = datetime.now() - timedelta(days=7)

    while current_bytes < target_bytes:
        log_time = base_time + timedelta(seconds=random.randint(0, 7*24*3600))
        service = random.choice(LOG_SERVICES)
        level = random.choice(LOG_LEVELS)
        message = random.choice(LOG_MESSAGES).format(
            client_id=f"client-{random.randint(1000, 9999)}",
            user_id=random.randint(10000, 99999)
        )

        lines.append(f"{log_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} [{service}] {level}: {message} | request_id={random.randint(100000, 999999)} trace_id={random.randint(10000000, 99999999)}")

        current_bytes = sum(len(line.encode('utf-8')) for line in lines)

    return lines


# ============================================================================
# MULTILINGUAL DOCUMENTATION GENERATORS
# ============================================================================

TRANSLATIONS = {
    "en": {
        "title": "User Guide: Getting Started",
        "intro": "Welcome to our product documentation. This guide will help you get started quickly.",
        "section1": "Installation: Follow these steps to install the software on your system.",
        "section2": "Configuration: Configure the application using the settings panel.",
        "section3": "Authentication: Set up user authentication and access controls."
    },
    "es": {
        "title": "Guía del Usuario: Primeros Pasos",
        "intro": "Bienvenido a la documentación de nuestro producto. Esta guía le ayudará a comenzar rápidamente.",
        "section1": "Instalación: Siga estos pasos para instalar el software en su sistema.",
        "section2": "Configuración: Configure la aplicación usando el panel de configuración.",
        "section3": "Autenticación: Configure la autenticación de usuarios y controles de acceso."
    },
    "zh": {
        "title": "用户指南：入门",
        "intro": "欢迎使用我们的产品文档。本指南将帮助您快速入门。",
        "section1": "安装：按照以下步骤在系统上安装软件。",
        "section2": "配置：使用设置面板配置应用程序。",
        "section3": "身份验证：设置用户身份验证和访问控制。"
    },
    "ja": {
        "title": "ユーザーガイド：はじめに",
        "intro": "製品ドキュメントへようこそ。このガイドは、すぐに始めるのに役立ちます。",
        "section1": "インストール：システムにソフトウェアをインストールする手順に従ってください。",
        "section2": "設定：設定パネルを使用してアプリケーションを設定します。",
        "section3": "認証：ユーザー認証とアクセス制御を設定します。"
    }
}

def generate_multilingual_docs(target_mb: int) -> List[str]:
    """
    Generate mock multilingual documentation corpus.

    Args:
        target_mb: Target size in MB

    Returns:
        List of documentation lines
    """
    lines = []
    lines.append("# MULTILINGUAL DOCUMENTATION CORPUS")
    lines.append(f"# Generated: {datetime.now().isoformat()}")
    lines.append("# Languages: EN, ES, ZH, JA")
    lines.append("")

    target_bytes = target_mb * 1024 * 1024
    current_bytes = sum(len(line.encode('utf-8')) for line in lines)

    doc_num = 1

    while current_bytes < target_bytes:
        for lang, content in TRANSLATIONS.items():
            lines.append(f"\n========== DOC-{doc_num} | LANGUAGE: {lang.upper()} ==========")
            lines.append(f"# {content['title']}")
            lines.append("")
            lines.append(content['intro'])
            lines.append("")
            lines.append(f"## Section 1")
            lines.append(content['section1'])
            lines.append("")
            lines.append(f"## Section 2")
            lines.append(content['section2'])
            lines.append("")
            lines.append(f"## Section 3")
            lines.append(content['section3'])
            lines.append("")

        doc_num += 1
        current_bytes = sum(len(line.encode('utf-8')) for line in lines)

    return lines


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def save_corpus_to_file(lines: List[str], filename: str) -> None:
    """Save generated corpus to file."""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def get_corpus_stats(lines: List[str]) -> Dict[str, any]:
    """Calculate corpus statistics."""
    text = '\n'.join(lines)
    return {
        'lines': len(lines),
        'bytes': len(text.encode('utf-8')),
        'mb': len(text.encode('utf-8')) / (1024 * 1024),
        'avg_line_length': sum(len(line) for line in lines) / len(lines) if lines else 0
    }
