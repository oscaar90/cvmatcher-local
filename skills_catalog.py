# skills_catalog.py

import re
from collections import defaultdict

SKILLS_CATALOG = {
    # Lenguajes
    "python": ["python", "django", "flask", "pandas", "numpy", "pytest"],
    "javascript": ["javascript", "js", "node.js", "nodejs", "react", "vue", "angular"],
    "bash": ["bash", "shell", "scripting", "scripts"],
    "powershell": ["powershell", "ps1", "windows scripting"],
    "sql": ["sql", "mysql", "postgresql", "postgres", "oracle", "mariadb", "t-sql"],
    "php": ["php", "laravel", "symfony"],
    "html": ["html", "html5"],
    "css": ["css", "css3", "sass", "scss", "bootstrap"],
    "cobol": ["cobol", "mainframe"],
    "jcl": ["jcl", "job control language"],

    # DB / observabilidad puntuales
    "grafana": ["grafana"],
    "influxdb": ["influxdb"],
    "telegraf": ["telegraf"],

    # DevOps / Cloud / IaC (alias NO genéricos)
    "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s"],
    "jenkins": ["jenkins", "jenkins pipeline"],
    "aws": ["aws", "amazon web services", "ec2", "s3"],
    "azure": ["azure", "microsoft azure"],
    "terraform": ["terraform"],
    "ansible": ["ansible"],  # ← quitamos "automation", "configuración"

    # Sistemas
    "linux": ["linux", "ubuntu", "centos", "redhat", "debian", "suse"],
    "windows": ["windows", "windows server"],
    "vmware": ["vmware", "vsphere"],
    "hyper_v": ["hyper-v", "hyper v"],

    # Herramientas varias
    "n8n": ["n8n"],
    "sccm": ["sccm", "system center"],
    "git": ["git", "github", "gitlab", "bitbucket"],
    "confluence": ["confluence"],   # ← quitamos "documentación", "wiki"
    "servicenow": ["servicenow"],

    # Seguridad
    "security": ["cybersecurity", "seguridad"],
    "ethical_hacking": ["ethical hacking", "penetration testing"],

    # Metodologías
    "agile": ["agile", "scrum", "kanban"],
    "itil": ["itil"],

    # IA
    "ai": ["ai", "inteligencia artificial", "llm", "machine learning"],

    # Redes / fabricantes / conceptos
    "cisco": ["cisco"],
    "aruba": ["aruba"],
    "juniper": ["juniper"],
    "f5": ["f5", "big-ip", "bigip"],
    "checkpoint": ["checkpoint"],
    "palo_alto": ["palo alto", "paloalto"],
    "arista": ["arista"],
    "infoblox": ["infoblox"],
    "allot": ["allot"],
    "aerohive": ["aerohive"],
    "lan": ["lan"],
    "wan": ["wan"],
    "wifi": ["wifi", "wi-fi"],
    "load_balancer": ["load balancer", "balanceo de carga"],
    "routing": ["routing", "enrutamiento"],
    "switching": ["switching", "conmutación"],
    "firewall": ["firewall", "cortafuegos"],
}

# Build reverse index alias -> canon
ALIAS2CANON = {}
for canon, aliases in SKILLS_CATALOG.items():
    for a in aliases:
        ALIAS2CANON[a.lower()] = canon

# Aliases DEMASIADO genéricos que NO debemos mapear a herramientas en modo estricto
STOP_ALIASES = {
    "automation", "automatización",
    "configuración", "configuracion",
    "monitoring", "monitorizacion", "monitorización",
    "documentacion", "documentación", "wiki",
    "workflow", "flujos", "procesos",
    "database", "db", "web", "frontend", "backend", "cloud",
    "scripting", "scripts"
}

def _norm_text(s: str) -> str:
    s = (s or "").lower()
    s = s.replace("·", ",").replace("/", " ")
    s = re.sub(r"[()\[\]{}|]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def canonicalize_token(tok: str, strict: bool = False) -> str | None:
    t = tok.strip().lower()
    if not t:
        return None
    if strict and t in STOP_ALIASES:
        return None
    # alias exacto
    if t in ALIAS2CANON:
        return ALIAS2CANON[t]
    t2 = t.replace("–", "-")
    if strict and t2 in STOP_ALIASES:
        return None
    if t2 in ALIAS2CANON:
        return ALIAS2CANON[t2]
    return None

def extract_skills_from_text(text: str, max_ngrams: int = 2, strict: bool = False) -> set[str]:
    """
    Extrae skills canónicas desde texto.
    - strict=True: ignora alias genéricos para evitar falsos positivos (JD).
    """
    text_n = _norm_text(text)
    tokens = re.split(r"[,\;\|\-•\n\. ]", text_n)
    tokens = [t for t in tokens if t]

    hits = set()
    # unigrama
    for t in tokens:
        c = canonicalize_token(t, strict=strict)
        if c:
            hits.add(c)
    # bigrama simple
    if max_ngrams >= 2:
        for i in range(len(tokens) - 1):
            big = canonicalize_token(tokens[i] + " " + tokens[i+1], strict=strict)
            if big:
                hits.add(big)
    return hits
