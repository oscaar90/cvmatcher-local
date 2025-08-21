from reportlab.lib.units import mm
import time, os, psutil
try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_ENABLED = True
except ImportError:
    PROMETHEUS_ENABLED = False
try:
    import GPUtil
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
# Standard library
import io
import re
import json
import uuid
from datetime import datetime
import unicodedata, re

# Third-party
import requests
import pdfplumber
import docx2txt
import ollama
from flask import (
    Flask, request, render_template, redirect,
    url_for, flash, jsonify, session, send_file
)
from werkzeug.utils import secure_filename
from pypdf import PdfReader, PdfWriter, PageObject
from weasyprint import HTML

# Local modules
from skills_catalog import SKILLS_CATALOG, ALIAS2CANON, extract_skills_from_text


app = Flask(__name__)
app.secret_key = 'cvmatcher_secret_key_2024'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['JOB_CACHE_DIR'] = os.path.join(os.path.dirname(__file__), '.cache_jobs')
process = psutil.Process(os.getpid())

os.makedirs(app.config['JOB_CACHE_DIR'], exist_ok=True)


def _job_path(job_id: str) -> str:
    return os.path.join(app.config['JOB_CACHE_DIR'], f'{job_id}.json')

def save_job_payload(payload: dict) -> str:
    job_id = str(uuid.uuid4())
    with open(_job_path(job_id), 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False)
    return job_id

def load_job_payload(job_id: str) -> dict | None:
    try:
        with open(_job_path(job_id), 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

# Crear directorio de uploads si no existe
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configuración Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"

def _canonize_list(xs):
    out = set()
    for x in xs or []:
        t = (x or "").strip().lower()
        if not t:
            continue
        if t in SKILLS_CATALOG:
            out.add(t); continue
        if t in ALIAS2CANON:
            out.add(ALIAS2CANON[t]); continue
        out |= extract_skills_from_text(t, strict=False)
    return out

def _norm(x: str) -> str:
    return (x or "").strip().lower()
BASE_FONT = "Helvetica"

def _mm(x): 
    return x * mm

def _wrap_text(text, font_name, font_size, max_width):
    from reportlab.pdfbase.pdfmetrics import stringWidth
    words = str(text or "").split()
    lines, cur = [], []
    for w in words:
        probe = " ".join(cur + [w])
        if stringWidth(probe, font_name, font_size) <= max_width:
            cur.append(w)
        else:
            if cur:
                lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return lines

def _draw_lines(c, x, y, lines, font_size=10, leading=13, max_lines=None):
    c.setFont(BASE_FONT, font_size)
    n = 0
    for line in lines or []:
        if max_lines and n >= max_lines:
            break
        c.drawString(x, y, line)
        y -= leading
        n += 1
    return y

def _draw_bullets(c, x, y, bullets, font_size=10, leading=14, max_width=_mm(170), bullet_char="•"):
    from reportlab.pdfbase.pdfmetrics import stringWidth
    c.setFont(BASE_FONT, font_size)
    for b in bullets or []:
        c.drawString(x, y, bullet_char)
        tw = max_width - stringWidth("  ", BASE_FONT, font_size)
        lines = _wrap_text(str(b), BASE_FONT, font_size, tw)
        if not lines:
            y -= leading
            continue
        c.drawString(x + 10, y, lines[0]); y -= leading
        for ln in lines[1:]:
            c.drawString(x + 10, y, ln); y -= leading
    return y

def _pick(arr, n):
    return [x for x in (arr or []) if x][:n]

def _role_title(role: str) -> str:
    """No mapear nada: usar exactamente lo que venga del LLM/analysis."""
    role = (role or "").strip()
    return role or "Profesional IT"



def _resolve_template_path() -> str:
    import os

    # Rutas candidatas
    candidates = [
        os.getenv("CV_TEMPLATE_PATH"),
        os.path.join("assets", "cv_base.pdf"),
        os.path.join("assets", "cv base.pdf"),
    ]

    # Filtra solo las que existen
    valid = [p for p in candidates if p and os.path.exists(p)]
    if valid:
        return valid[0]

    # Si ninguna existe, error claro
    raise FileNotFoundError(
        f"""No se encontró el template de CV.
Define CV_TEMPLATE_PATH o coloca 'cv_base.pdf' (o 'cv base.pdf') en ./assets/.
Probadas: {', '.join([p for p in candidates if p])}"""
    )
def _pick_n(seq, n):
    return [x for x in (seq or []) if x][:n]

def _profile_from_payload_for_role(payload: dict, role: str) -> dict:
    """Extrae datos reales del payload para el CV, sin inventar nada."""
    a  = (payload or {}).get("analysis") or {}
    ui = (payload or {}).get("ui") or {}
    fb = (payload or {}).get("feedback") or {}

    def first(*opts, default=""):
        for o in opts:
            if isinstance(o, str) and o.strip(): return o.strip()
            if isinstance(o, (list, tuple)) and o: return o
        return default

    nombre = first(a.get("nombre"), a.get("nombre_completo"), ui.get("headline", {}).get("nombre"), default="Nombre Apellidos")
    anios  = first(ui.get("headline", {}).get("anios_total"), a.get("headline", {}).get("anios_total"), a.get("anios_total"), default="")
    role_title = (role or "Profesional IT").strip()
    titulo = f"{role_title} · {anios} años" if str(anios) else role_title

    fortalezas = first(ui.get("headline", {}).get("fortalezas"), a.get("fortalezas"), fb.get("fortalezas"), default=[])
    resumen_bullets = _pick_n(fortalezas, 4) or _pick_n(a.get("highlights") or [], 4)

    exp_src = first(ui.get("experiencia"), a.get("experiencia"), default=[]) or []
    experiencia = []
    for e in exp_src[:6]:   # hasta 6 entradas, paginado automático
        if not isinstance(e, dict): 
            continue
        experiencia.append({
            "empresa": e.get("empresa") or e.get("org") or "",
            "pais":    e.get("pais") or e.get("ubicacion") or "",
            "puesto":  e.get("rol") or e.get("puesto") or "",
            "fecha":   e.get("rango_fechas") or f"{(e.get('inicio') or '')} – {(e.get('fin') or 'Actual')}",
            "logros":  _pick_n(e.get("logros") or ([e.get("descripcion")] if e.get("descripcion") else []), 5),
        })

    edu_src = first(ui.get("educacion"), a.get("educacion"), default=[]) or []
    educacion = []
    for ed in edu_src[:3]:
        if not isinstance(ed, dict): 
            continue
        educacion.append({
            "titulo": ed.get("titulo") or ed.get("grado") or "",
            "centro": ed.get("centro") or ed.get("escuela") or "",
            "fecha":  ed.get("fecha") or ed.get("rango_fechas") or "",
            "estado": ed.get("estado") or "",
        })

    skills = first(ui.get("headline", {}).get("fortalezas"), a.get("full_skills"), default=[])
    if not isinstance(skills, list): skills = []
    skills = skills[:16]  # más que en la cabecera del PDF-plantilla para dar contexto

    idiomas = []
    for i in first(ui.get("idiomas"), a.get("idiomas"), default=[]):
        if isinstance(i, dict): idiomas.append(f"{i.get('nombre')}: {i.get('nivel')}")
        else: idiomas.append(str(i))

    contact = first(ui.get("contacto"), a.get("contacto"), default={}) or {}
    email = first(contact.get("email"), a.get("email"), default="")
    web   = first(contact.get("web"),   a.get("web"),   default="")
    city  = first(contact.get("ciudad"),a.get("ciudad"),default="")

    proyectos = first(ui.get("proyectos"), a.get("proyectos"), default=[])
    if not isinstance(proyectos, list): proyectos = []

    return {
        "nombre": nombre, "titulo": titulo,
        "resumen_bullets": resumen_bullets,
        "experiencia": experiencia,
        "educacion": educacion,
        "skills": skills,
        "idiomas": idiomas,
        "email": email, "web": web, "ciudad": city,
        "proyectos": proyectos,
    }

def _get_template_pagesize(path):
    """Lee el tamaño real de la página 1 del template (en puntos)."""
    r = PdfReader(path)
    box = r.pages[0].mediabox
    width  = float(box.width)
    height = float(box.height)
    return (width, height), r

def _render_cv_on_overlay(data: dict, page_size_pts) -> bytes:
    """Genera overlay con el tamaño del template (portable)."""
    page_w, page_h = page_size_pts
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(page_w, page_h))

    right_x = page_w - _mm(20)

    # ----- cabecera -----
    c.setFont(BASE_FONT, 16)
    c.drawString(_mm(20), _mm(280), data["nombre"])
    c.setFont(BASE_FONT, 12); c.setFillGray(0.2)
    c.drawString(_mm(20), _mm(272), data["titulo"])
    c.setFillGray(0.0)

    # contacto (derecha)
    c.setFont(BASE_FONT, 9)
    if data.get("web"):    c.drawRightString(right_x, _mm(284), str(data["web"]))
    if data.get("email"):  c.drawRightString(right_x, _mm(280), str(data["email"]))
    if data.get("ciudad"): c.drawRightString(right_x, _mm(276), str(data["ciudad"]))

    # ----- Resumen -----
    y = _mm(260)
    y = _draw_bullets(c, _mm(20), y, data.get("resumen_bullets"), font_size=10, leading=13, max_width=_mm(170))

    # ----- Experiencia -----
    y -= _mm(4)
    c.setFont(BASE_FONT, 12); c.drawString(_mm(20), y, "Experiencia Laboral"); y -= _mm(6)

    for e in data.get("experiencia") or []:
        if y < _mm(70):
            c.showPage(); y = _mm(280)
        c.setFont(BASE_FONT, 10)
        cab = " — ".join([x for x in [e.get("empresa"), e.get("pais"), e.get("puesto")] if x])
        y = _draw_lines(c, _mm(20), y, _wrap_text(cab, BASE_FONT, 10, _mm(170)), font_size=10, leading=12)
        if e.get("fecha"):
            c.setFont(BASE_FONT, 9)
            c.drawRightString(right_x, y+12, str(e["fecha"]))
        y = _draw_bullets(c, _mm(24), y, e.get("logros"), font_size=9, leading=12, max_width=_mm(166))
        y -= _mm(2)

    # ----- Educación / Skills / Idiomas -----
    if y < _mm(90):
        c.showPage(); y = _mm(280)

    c.setFont(BASE_FONT, 12); c.drawString(_mm(20), y, "Educación"); y -= _mm(6)
    for ed in data.get("educacion") or []:
        linea = " | ".join([x for x in [ed.get("titulo"), ed.get("centro"), ed.get("fecha"), ed.get("estado")] if x])
        y = _draw_lines(c, _mm(20), y, _wrap_text(linea, BASE_FONT, 10, _mm(170)), font_size=10, leading=12)
    y -= _mm(3)

    c.setFont(BASE_FONT, 12); c.drawString(_mm(20), y, "Habilidades"); y -= _mm(6)
    if data.get("skills"):
        y = _draw_lines(c, _mm(20), y, _wrap_text(", ".join(data["skills"]), BASE_FONT, 10, _mm(170)), font_size=10, leading=12)
    y -= _mm(3)

    if data.get("idiomas"):
        c.setFont(BASE_FONT, 12); c.drawString(_mm(20), y, "Idiomas"); y -= _mm(6)
        y = _draw_lines(c, _mm(20), y, _wrap_text(", ".join(data["idiomas"]), BASE_FONT, 10, _mm(170)), font_size=10, leading=12)

    # ----- Proyectos (opcional) -----
    if data.get("proyectos"):
        if y < _mm(70):
            c.showPage(); y = _mm(280)
        c.setFont(BASE_FONT, 12); c.drawString(_mm(20), y, "Proyectos clave"); y -= _mm(6)
        bullets = []
        for p in data["proyectos"]:
            if isinstance(p, dict):
                bullets.append(" — ".join([x for x in [p.get('titulo'), p.get('desc')] if x]))
            else:
                bullets.append(str(p))
        _draw_bullets(c, _mm(20), y, bullets, font_size=10, leading=13, max_width=_mm(170))

    c.save()
    return buf.getvalue()

def _merge_overlay_on_template(overlay_pdf_bytes: bytes, template_reader: PdfReader) -> bytes:
    over = PdfReader(io.BytesIO(overlay_pdf_bytes))
    writer = PdfWriter()
    # Página 1: mezclar
    base_page: PageObject = template_reader.pages[0]
    base_page.merge_page(over.pages[0])
    writer.add_page(base_page)
    # Páginas extra del overlay
    for i in range(1, len(over.pages)):
        writer.add_page(over.pages[i])
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()

def extract_text_from_pdf(file_path):
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error extrayendo texto del PDF: {e}")
    return text

def extract_total_experience(cv_text):
    """Extrae años totales de experiencia del CV"""
    patterns = [
        r'(\d+)\s*años?\s*(?:de\s*)?(?:experiencia|en\s*sistemas|en\s*el\s*sector)',
        r'experiencia\s*(?:de\s*)?(\d+)\s*años?',
        r'(\d+)\s*years?\s*(?:of\s*)?experience'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, cv_text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    
    # Calcular por fechas de trabajo si no encuentra explícito
    return calculate_years_from_dates(cv_text)

def shortcircuit_if_non_tech(cv_text: str, filename: str):
    """Si no es perfil TECH, renderiza directamente salida mínima y evita gastar recursos."""
    dom = detect_domain(cv_text, llm_skills=None)  # ya lo tienes definido
    if dom.get("domain") != "tech":
        analysis_min = {
            "especializacion_principal": "No técnica",
            "años_experiencia_total": 0,
            "fortalezas_principales": [],
            "enfoque_busqueda": {"buscar": []},
            "balance_skills": {"status": "no_aplica", "mensaje": "App solo TECH"},
            "informe_rrhh": {"conclusion": "App solo TECH"},
            "dominio": dom.get("domain"),
            "tech_score": dom.get("score"),
            "tech_evidence": dom.get("evidence", []),
            "skills": [],  # vacío
        }
        # UI mínimo para que la plantilla no falle
        ui_min = {
            "headline": {
                "especializacion": "No técnica",
                "anios_total": 0,
                "top_roles": [],
                "fit_badge": None,
                "fortalezas": [],
                "tech_score": dom.get("score"),
            },
            "skills_levels": {"EXPERTO": [], "AVANZADO": [], "INTERMEDIO": [], "JUNIOR": []},
            "skills_overflow": {"EXPERTO": 0, "AVANZADO": 0, "INTERMEDIO": 0, "JUNIOR": 0},
            "balance": {"label": "N/A", "mensaje": "App solo TECH"},
            "full_skills": [],
        }
        feedback_min = {
            "fortalezas": ["App solo TECH"],
            "mejoras": [],
            "roles": [],
            "keywords_ats": []
        }
        return render_template(
            'results_nontech.html',
            analysis=analysis_min,
            ui=ui_min,
            feedback=feedback_min,
            filename=filename
        )
    return None  # es TECH → continuar flujo normal

def calculate_years_from_dates(cv_text):
    """Calcula experiencia por fechas de trabajo"""
    from datetime import datetime
    
    # Buscar patrones como "2019 - 2025", "Marzo 2022 – Actualidad"
    date_patterns = [
        r'(\d{4})\s*[-–]\s*(\d{4}|\w+)',
        r'(\w+\s+\d{4})\s*[-–]\s*(\w+\s+\d{4}|\w+)'
    ]
    
    current_year = datetime.now().year
    total_years = 0
    
    for pattern in date_patterns:
        matches = re.findall(pattern, cv_text)
        for match in matches:
            start, end = match
            try:
                if start.isdigit():
                    start_year = int(start)
                else:
                    # Extraer año de "Marzo 2022"
                    year_match = re.search(r'\d{4}', start)
                    start_year = int(year_match.group()) if year_match else current_year
                
                if end.lower() in ['actualidad', 'presente', 'current']:
                    end_year = current_year
                elif end.isdigit():
                    end_year = int(end)
                else:
                    year_match = re.search(r'\d{4}', end)
                    end_year = int(year_match.group()) if year_match else current_year
                
                years_diff = end_year - start_year
                if years_diff > 0:
                    total_years += years_diff
            except:
                continue
    
    return max(total_years, 1)  # Mínimo 1 año

def extract_text_from_docx(file_path):
    """Extrae texto de archivos DOCX"""
    try:
        return docx2txt.process(file_path)
    except Exception as e:
        print(f"Error extrayendo texto del DOCX: {e}")
        return ""

def extract_text_from_txt(file_path):
    """Extrae texto de archivos TXT"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error extrayendo texto del TXT: {e}")
        return ""

def extract_cv_text(file_path):
    """Extrae texto del CV según su extensión"""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension == '.docx':
        return extract_text_from_docx(file_path)
    elif file_extension == '.txt':
        return extract_text_from_txt(file_path)
    else:
        return ""


# === TECH DOMAIN DETECTION (determinista) ===
TECH_SCORE_THRESHOLD = 0.60  # si score < 0.60 => non-tech
def detect_domain(cv_text: str, llm_skills: list[str] | None = None):
    """
    Devuelve: {'domain': 'tech'|'non_tech', 'score': 0..1, 'confidence': 'high'|'none', 'evidence': [...]}
    Basado únicamente en SKILLS_CATALOG/ALIAS2CANON y un umbral TECH_SCORE_THRESHOLD.
    """
    text = (cv_text or "").lower()
    hits = set()

    # Busca directamente en el texto
    for kw in SKILLS_CATALOG:
        if kw.lower() in text:
            hits.add(kw.lower())

    # Añade skills detectadas por LLM si están en el catálogo o alias
    if llm_skills:
        for sk in llm_skills:
            s = (sk or "").lower()
            if s in SKILLS_CATALOG or s in ALIAS2CANON:
                hits.add(s)

    # Score relativo
    score_raw = len(hits)
    score = min(1.0, score_raw / 6.0)

    if score >= TECH_SCORE_THRESHOLD:
        domain, confidence = "tech", "high"
    else:
        domain, confidence = "non_tech", "none"

    return {
        "domain": domain,
        "score": round(score, 3),
        "confidence": confidence,
        "evidence": sorted(hits),
    }

# adapters.py (o similar)
job_mappings = {
    "Tech Generalist": {
        "buscar": ["Software Engineer", "Full-Stack Developer", "Technical Analyst", "IT Consultant"]
    }
}


def clean_json_response(response_text):
    """Limpia la respuesta de Ollama para extraer solo el JSON"""
    response_text = response_text.strip()
    
    # Buscar JSON entre las líneas
    lines = response_text.split('\n')
    json_start = -1
    json_end = -1
    
    for i, line in enumerate(lines):
        if line.strip().startswith('{'):
            json_start = i
            break
    
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip().endswith('}'):
            json_end = i
            break
    
    if json_start != -1 and json_end != -1:
        json_lines = lines[json_start:json_end + 1]
        return '\n'.join(json_lines)
    
    # Si no encuentra JSON estructurado, devolver tal como está
    return response_text

def generate_job_focus(
    skills: list[dict],
    specialization: str | None = None,
    *,
    total_years: int | None = None,
    cv_text: str | None = None,
    **_unused,
) -> dict:
    """
    Única API pública. No inventa posiciones.
    Delegamos en lo que exista en skills_catalog:
      - generate_job(skills, specialization, ...)  (preferido)
      - get_job_focus_map(specialization)          (alternativo)
    """
    spec = (specialization or "Tech Generalist").strip()

    # 1) Prefiere generate_job(...) si existe
    gen = getattr(sc, "generate_job", None)
    if callable(gen):
        try:
            return gen(
                skills=skills,
                specialization=spec,
                total_years=total_years,
                cv_text=cv_text,
            )
        except TypeError:
            # Degrada si la firma del catálogo es más simple
            try:
                return gen(skills, spec)
            except TypeError:
                return gen(skills)

    # 2) Alternativa: get_job_focus_map(spec) si existe
    gjfm = getattr(sc, "get_job_focus_map", None)
    if callable(gjfm):
        try:
            return gjfm(spec)
        except TypeError:
            # por si acepta args extra accidentalmente
            return gjfm(spec)

    # 3) Failsafe: estructura vacía para no romper la UI
    return {"buscar": [], "evitar": [], "keywords": [], "mensaje": ""}

def estimate_years_enhanced(cv_text, skill_name, has_work_context):
    """Estima años con lógica mejorada y conversión segura"""    
    # 1. Buscar menciones explícitas de años
    patterns = [
        rf"{re.escape(skill_name)}.*?(\d+)\s*años?",
        rf"(\d+)\s*años?.*?{re.escape(skill_name)}",
        rf"{re.escape(skill_name)}.*?(\d+)\s*year"
    ]
    
    for pattern in patterns:
        try:
            match = re.search(pattern, cv_text, re.IGNORECASE)
            if match:
                years = safe_int(match.group(1), default=1, min_val=1, max_val=15)
                return years
        except (ValueError, AttributeError):
            continue
    
    # 2. Estimación basada en contexto y posición
    cv_lower = (cv_text or "").lower()
    skill_lower = (skill_name or "").lower()
    
    # Skills de especialización reciente (IA/LLM)
    ai_skills = ["ollama", "mistral", "llama", "embeddings", "rag", "faiss", "chroma"]
    if skill_lower in ai_skills:
        return 3 if has_work_context else 1
    
    # Skills fundamentales con contexto laboral
    if has_work_context:
        fundamental_skills = ["python", "linux", "sql", "automation", "bash", "powershell", "postgresql", "flask"]
        if skill_lower in fundamental_skills:
            # Buscar indicadores de seniority
            senior_indicators = ["senior", "14 años", "experto", "especialista"]
            mid_indicators = ["administrador", "ingeniero", "lead"]
            
            if any(term in cv_lower for term in senior_indicators):
                return 8
            elif any(term in cv_lower for term in mid_indicators):
                return 5
            else:
                return 3
        else:
            return 3  # Otras skills con contexto laboral
    else:
        return 1  # Sin contexto laboral = básico
    

def build_analysis_from_llm(cv_text, llm_obj):
    """Construye el análisis completo a partir de las skills devueltas por la IA (robusto, con ranking)."""

    # 1) PROCESAR SKILLS DEVUELTAS POR EL LLM
    skills_out = []
    for item in (llm_obj.get("skills") or []):
        raw = (item.get("name") or "").strip()
        if not raw:
            continue

        # Contexto laboral (mejorado)
        has_ctx = check_work_context_enhanced(cv_text, raw.lower())

        # Estimar años con más precisión
        years = estimate_years_enhanced(cv_text, raw, has_ctx)
        years = safe_int(years, default=1, min_val=1, max_val=15)

        # Nivel en base a años y contexto
        level = get_level_from_years_enhanced(years, has_ctx)

        skills_out.append({
            "name": raw,
            "years": years,
            "level": level,
            "has_work_context": has_ctx,
            "evidence": (item.get("evidence") or "").strip()
        })

    # 1.b) ENRIQUECIMIENTO DESDE TEXTO (por si el LLM trae pocas skills)
    try:
        extra_names = extract_skills_from_text(cv_text) or []
    except Exception:
        extra_names = []

    # Normaliza y evita duplicados; respeta alias -> nombre canónico si existe
    present = {(s.get("name") or "").strip().lower() for s in skills_out}
    added = 0
    for raw in extra_names:
        cand = (raw or "").strip()
        if not cand:
            continue

        key_lower = cand.lower()
        canon = ALIAS2CANON.get(key_lower, cand).strip()
        canon_key = canon.lower()

        if canon_key in present:
            continue

        has_ctx = check_work_context_enhanced(cv_text, cand.lower())
        years = estimate_years_enhanced(cv_text, cand, has_ctx)
        years = safe_int(years, default=1, min_val=0, max_val=40)
        level = get_level_from_years_enhanced(years, has_ctx)

        skills_out.append({
            "name": canon,
            "years": years,
            "level": level,
            "has_work_context": has_ctx,
            "evidence": ""
        })
        present.add(canon_key)
        added += 1
        if added >= 25:  # límite defensivo
            break

    # === GUARDRAIL: 0 SKILLS REALES / SOLO GENÉRICAS (USANDO SOLO EL CATÁLOGO) ===
    # Consideramos "skill real" si el nombre (o su alias) está en ALIAS2CANON o en SKILLS_CATALOG.
    def _is_real(name: str) -> bool:
        n = (name or "").strip().lower()
        return (n in ALIAS2CANON) or (n in SKILLS_CATALOG)

    real_hits = [s for s in skills_out if _is_real(s.get("name"))]

    # Fallback adicional: si no hay ni una, intentamos otro pase sobre el texto bruto
    if not real_hits and extra_names:
        real_hits = [{"name": n} for n in extra_names if _is_real(n)]

    if len(real_hits) == 0:
        harsh = str(os.getenv("HARSH_MODE", "false")).strip().lower() in {"1", "true", "yes", "on"}
        msg = (
            "Este CV es una puta mierda: no aporta skills. Rehácelo desde cero."
            if harsh else
            "CV no válido: 0 skills reales. Debes rehacerlo desde cero con skills medibles."
        )
        # Métrica simple (sin dependencias nuevas)
        print("[METRIC] cvmatcher.zero_skills_triggered=1")

        # Payload mínimo/compatible para la UI
        return {
            "no_apto": True,
            "skills": [],
            "especializacion_principal": "No evaluable",
            "especializacion_secundaria": None,
            "años_experiencia_total": 0,
            "fortalezas_principales": [],
            "enfoque_busqueda": {
                "blocked": True,
                "buscar": [],
                "ranking": [],
                "mensaje": msg
            },
            "nivel_empleabilidad": "No evaluable",
            "salario_estimado": "N/D",
            "balance_skills": {"status": "rechazado", "mensaje": msg, "tipo": "error"},
            "informe_rrhh": {
                "status": "Rechazado",
                "motivo": "0 skills reales / solo genéricas",
                "conclusion": msg
            },
            "dominio": "tech",
            "tech_score": 0.0,
            "tech_evidence": [],
            "ui_flags": {
                "disable_role_cv": True,
                "hide_green_check": True,
                "match_global": "No evaluable",
                "top_roles": []
            },
            "checklist_minima": [
                "Añade 6–10 skills técnicas específicas (p. ej., SQL, Python, Docker, Linux).",
                "Incluye 3 logros con métricas (%/€, tiempo, volumen).",
                "Lista 2–4 proyectos con stack, tu rol y resultados.",
                "Detalla herramientas concretas (Git, Ansible, FastAPI, Grafana).",
                "Añade certificaciones/cursos relevantes con fecha.",
                "Incluye links: GitHub/Portfolio/LinkedIn.",
                "Aclara años de experiencia por 3–5 skills (p. ej., Python 4 años).",
                "Elimina adjetivos vacíos sin evidencia."
            ]
        }
    # === FIN GUARDRAIL =========================================================

    # 2) ESPECIALIZACIÓN
    specialization = determine_specialization_enhanced(skills_out, cv_text)

    # 3) EXPERIENCIA TOTAL (CONVERSIÓN SEGURA)
    total_years_raw = llm_obj.get("total_experience_years", 0)
    total_years = safe_int(total_years_raw, default=0, min_val=0, max_val=40)
    print(f"🔍 LLM EXPERIENCIA RAW: {total_years_raw} → CONVERTIDO: {total_years} años")

    if total_years == 0:
        extracted_years = extract_total_experience_from_cv(cv_text)
        total_years = safe_int(extracted_years, default=0, min_val=0, max_val=40)
        print(f"🔍 EXTRAIDA DEL CV: {extracted_years} → CONVERTIDO: {total_years} años")

        if total_years == 0:
            skill_years = []
            for s in skills_out:
                if s.get("has_work_context"):
                    skill_year = safe_int(s.get("years", 0), default=0, min_val=0, max_val=15)
                    skill_years.append(skill_year)
            total_years = max(skill_years) if skill_years else 2
            print(f"🔍 CALCULADA POR SKILLS: {total_years} años")

    total_years = safe_int(total_years, default=2, min_val=1, max_val=40)
    print(f"🔍 EXPERIENCIA FINAL: {total_years} años")

    # 4) DETECTAR DOMINIO TÉCNICO
    _llm_skill_names = [s.get("name", "") for s in (llm_obj.get("skills") or [])]
    dom = detect_domain(cv_text, _llm_skill_names)

    # 5) FILTRO CV NO TÉCNICO
    if dom["domain"] != "tech":
        return {
            "skills": skills_out,
            "especializacion_principal": "No técnica",
            "especializacion_secundaria": None,
            "años_experiencia_total": 0,
            "fortalezas_principales": [],
            "enfoque_busqueda": {
                "blocked": True,
                "buscar": [],
                "mensaje": f"CV no técnico detectado (score: {dom['score']}). Esta app es solo para perfiles TECH. Sube un CV técnico.",
                "tech_score": dom["score"],
                "evidence": dom["evidence"]
            },
            "nivel_empleabilidad": "N/D",
            "salario_estimado": "N/D",
            "balance_skills": {"status": "no_aplica", "mensaje": "App solo TECH", "tipo": "info"},
            "informe_rrhh": {"conclusion": "No se genera informe para CV no técnico."},
            "dominio": dom["domain"],
            "tech_score": dom["score"],
            "tech_evidence": dom["evidence"]
        }

    # 6) MÉTRICAS CON EXPERIENCIA TOTAL
    employability = calculate_employability(skills_out, total_years)

    # 7) ENFOQUE DE BÚSQUEDA + RANKING
    base_focus = generate_job_focus(skills_out, specialization, cv_text=cv_text)
    base_focus = {"buscar": base_focus.get("buscar", [])}

    candidates = base_focus.get("buscar", [])
    ranking = rank_roles_with_llm(cv_text, skills_out, candidates)
    job_focus = {"buscar": candidates}
    if ranking:
        job_focus["ranking"] = ranking

    # 8) INFORMES
    rrhh_report = generate_rrhh_report(skills_out, cv_text)
    balance = analyze_skills_balance(skills_out)

    # 9) FORTALEZAS (top 3 con contexto)
    strong_skills = [s for s in skills_out if s.get("has_work_context") and s["level"] in ["EXPERTO", "AVANZADO"]]
    strong_skills.sort(key=lambda x: safe_int(x.get("years", 0)), reverse=True)
    fortalezas = [s["name"] for s in strong_skills[:3]]

    if len(fortalezas) < 3:
        all_with_context = [s for s in skills_out if s.get("has_work_context")]
        all_with_context.sort(key=lambda x: safe_int(x.get("years", 0)), reverse=True)
        for skill in all_with_context:
            if skill["name"] not in fortalezas and len(fortalezas) < 3:
                fortalezas.append(skill["name"])

    return {
        "skills": skills_out,
        "especializacion_principal": specialization,
        "especializacion_secundaria": "No detectada",
        "años_experiencia_total": total_years,
        "fortalezas_principales": fortalezas,
        "enfoque_busqueda": job_focus,
        "nivel_empleabilidad": employability,
        "salario_estimado": "N/D",
        "balance_skills": balance,
        "informe_rrhh": rrhh_report,
        "dominio": dom["domain"],
        "tech_score": dom["score"],
        "tech_evidence": dom["evidence"]
    }


def _cap_list(xs, limit=6):
    xs = [x for x in xs if x]
    overflow = max(0, len(xs) - limit)
    return xs[:limit], overflow

def _skills_by_level(skills):
    out = {"EXPERTO": [], "AVANZADO": [], "INTERMEDIO": [], "JUNIOR": []}
    for s in skills:
        lvl = s.get("level", "JUNIOR")
        out.setdefault(lvl, []).append(s["name"])
    return out

def _top_roles(enfoque):
    # intenta ranking → si no, usa 'buscar'
    ranking = (enfoque or {}).get("ranking") or []
    if ranking:
        return [f"{r['role']} ({r['fit_percent']}%)" for r in ranking[:3]]
    # fallback
    buscar = (enfoque or {}).get("buscar") or []
    return buscar[:3]
def build_ui_viewmodel(analysis: dict, limit_per_level: int = 6) -> dict:
    """
    ViewModel para la UI:
    - headline: especialización, años, fit_badge, top_roles (texto), roles_pct (dict), fortalezas
    - skills_levels: skills por nivel (capadas) + skills_overflow
    - balance: label + mensaje (RRHH desde backend)
    - full_skills: lista completa para "ver más"
    """
    esp   = analysis.get("especializacion_principal") or "N/D"
    years = analysis.get("años_experiencia_total") or 0
    enfoque = analysis.get("enfoque_busqueda") or {}
    fortalezas = (analysis.get("fortalezas_principales") or [])[:3]

    # 1) titulares
    top_roles = [] if analysis.get("no_apto") else _top_roles(enfoque)
    fit_badge = None
    ranking = (enfoque.get("ranking") or [])
    if ranking and not analysis.get("no_apto"):
        best = ranking[0]
        fit_badge = f"{int(best.get('fit_percent', 0))}% → {best.get('role', '')}"

    # Barras de compatibilidad por rol (top-3)
    roles_pct = {}
    for item in ranking[:3]:
        role = item.get("role", "")
        pct  = int(item.get("fit_percent", 0) or 0)
        if role:
            roles_pct[role] = max(0, min(100, pct))

    # 2) skills por nivel (capadas)
    levels = _skills_by_level(analysis.get("skills", []))
    ui_levels, overflow_map = {}, {}
    for lvl, arr in levels.items():
        capped, overflow = _cap_list(arr, limit=limit_per_level)
        ui_levels[lvl] = capped
        overflow_map[lvl] = overflow

    # 3) Mensaje RRHH: prioriza informe_rrhh.conclusion; fallback balance_skills.mensaje
    rrhh = (analysis.get("informe_rrhh") or {})
    rrhh_msg = rrhh.get("conclusion") or (analysis.get("balance_skills", {}) or {}).get("mensaje") or "Análisis disponible."

    # 4) Label del alert
    status = ((analysis.get("balance_skills") or {}).get("status") or "").lower()
    if status:
        label = status.upper()
    else:
        msgl = rrhh_msg.lower()
        if "excelente" in msgl or "excepcional" in msgl:
            label = "EXCELENTE"
        elif "warning" in msgl or "desbalanceado" in msgl or "mejorar" in msgl:
            label = "ALERTA"
        else:
            label = "OK"

    # 5) full skills
    full_skills = sorted({s.get("name") for s in analysis.get("skills", []) if s.get("name")})

    # Construir el ViewModel (primero crear `ui`, luego tocar flags)
    ui = {
        "headline": {
            "especializacion": esp,
            "anios_total": years,
            "top_roles": top_roles,      # ya condicionado arriba
            "roles_pct": roles_pct,      # calculado arriba
            "fit_badge": fit_badge,      # ya condicionado arriba
            "fortalezas": fortalezas,
            "tech_score": analysis.get("tech_score"),
        },
        "skills_levels": ui_levels,
        "skills_overflow": overflow_map,
        "balance": {
            "label": label,
            "mensaje": rrhh_msg,
        },
        "full_skills": full_skills,
    }

    # Flags de NO APTO (ahora `ui` ya existe)
    if analysis.get("no_apto"):
        ui["disable_role_cv"] = True
        ui["match_global"] = "No evaluable"
        ui["headline"]["top_roles"] = []
        ui["headline"]["fit_badge"] = None

    return ui

def generar_prioridades(cv_text: str, skills_detectadas: list[str] | None = None) -> list[dict]:
    """
    Devuelve 1..3 acciones priorizadas como lista de dicts:
    [
      {"accion": "...", "impacto": "...", "evidencia": "...", "esfuerzo": "bajo|medio|alto", "eta_dias": 1, "check": false}
    ]
    SIEMPRE intenta que lo genere el LLM en JSON. Aplica validación y limpieza mínima.
    """
    skills_detectadas = skills_detectadas or []
    skills_join = ", ".join(sorted({s for s in (skills_detectadas or []) if s}))

    prompt = (
        "Eres Director Técnico y Responsable de RRHH. Responde ÚNICAMENTE con JSON VÁLIDO.\n"
        "Objetivo: generar 1–3 acciones priorizadas, específicas del CV, listas para ejecutar.\n"
        "REQUISITOS de cada acción:\n"
        "- Debe referirse a algo verificable del CV (título, proyecto, skill, logro, métrica, periodo, stack).\n"
        "- Debe incluir una evidencia breve (frase/fragmento exacto del CV o elemento inequívoco: 'Sección 2011–2018: COBOL/JCL').\n"
        "- Evita generalidades (p.ej., 'mejorar procesos', 'documentar', 'investigar').\n"
        "- Máximo 3 acciones.\n\n"
        "FORMATO JSON ESTRICTO (y solo esto):\n"
        "{\n"
        "  \"acciones\": [\n"
        "    {\n"
        "      \"accion\": \"...\",\n"
        "      \"impacto\": \"...\",\n"
        "      \"evidencia\": \"...\",\n"
        "      \"esfuerzo\": \"bajo|medio|alto\",\n"
        "      \"eta_dias\": 1,\n"
        "      \"check\": false\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"Skills detectadas: {skills_join or '(no detectadas)'}\n\n"
        "CV (texto):\n\"\"\"" + (cv_text or "") + "\"\"\"\n"
    )

    # 1) LLM (JSON estricto)
    resp = query_ollama(prompt, expect_json=True)
    cleaned = clean_json_response(resp)

    acciones = []
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict) and isinstance(data.get("acciones"), list):
            acciones = data["acciones"]
    except Exception:
        acciones = []

    # 2) Validación + limpieza
    def es_generica(txt: str) -> bool:
        if not txt:
            return True
        t = txt.lower()
        prohibidas = [
            "mejorar procesos","optimizar procesos","documentar","resolver problemas",
            "investigar","aplicar técnicas de diagnóstico","comunicación","trabajo en equipo"
        ]
        if any(p in t for p in prohibidas):
            return True
        # Debe mencionar algo del CV (skill/proyecto/stack) para no ser genérica
        hints = [
            "flask","fastapi","ollama","mistral","llama","embeddings","faiss","chroma",
            "pytest","grafana","servicenow","terraform","postgres","sqlite","cobol","jcl",
            "passforge","idea tracker","ai tools engineer","automation"
        ]
        has_hint = any(h in t for h in hints) or any((s or "").lower() in t for s in skills_detectadas)
        return not has_hint

    cleaned_acc = []
    for a in acciones:
        if not isinstance(a, dict):
            continue
        accion   = (a.get("accion") or "").strip()
        impacto  = (a.get("impacto") or "").strip()
        evidencia = (a.get("evidencia") or "").strip()
        esfuerzo = (a.get("esfuerzo") or "medio").strip().lower()
        eta      = a.get("eta_dias", 3)

        if not accion or es_generica(accion):
            continue
        if esfuerzo not in {"bajo","medio","alto"}:
            esfuerzo = "medio"
        try:
            eta = int(eta); eta = max(1, min(30, eta))
        except Exception:
            eta = 3

        cleaned_acc.append({
            "accion": accion[:180],
            "impacto": (impacto or "Impacto directo en encaje/visibilidad.")[:180],
            "evidencia": evidencia[:200],
            "esfuerzo": esfuerzo,
            "eta_dias": eta,
            "check": False
        })

    # 3) Fallback mínimo si el modelo patina
    if not cleaned_acc:
        text = (cv_text or "").lower()
        fb = []
        def push(accion, impacto, evidencia, esfuerzo="bajo", eta=1):
            fb.append({"accion": accion, "impacto": impacto, "evidencia": evidencia, "esfuerzo": esfuerzo, "eta_dias": eta, "check": False})

        if "cobol" in text or "jcl" in text:
            push("Eliminar COBOL/JCL del resumen y moverlo a 'histórico'",
                 "Evita ruido para roles actuales",
                 "Sección 2011–2018: COBOL/JCL", "bajo", 1)
        if "flask" in text or "fastapi" in text:
            push("Añadir métrica de uso en APIs Flask/FastAPI",
                 "Aumenta credibilidad técnica",
                 "Stack: Flask/FastAPI", "medio", 2)
        if any(k in text for k in ["ollama","mistral","llama"]):
            push("Añadir benchmark de latencia/VRAM en inferencia local",
                 "Refuerza seniority en IA local",
                 "IA local: Ollama/Mistral", "medio", 2)

        cleaned_acc = fb[:3] if fb else [{
            "accion": "Ajustar el título al target (p.ej., 'AI Tools Engineer — Python-First, Local AI')",
            "impacto": "Mejor encaje en ATS y primeras cribas",
            "evidencia": "Resumen del CV",
            "esfuerzo": "bajo",
            "eta_dias": 1,
            "check": False
        }]

    return cleaned_acc[:3]

def rank_roles_with_llm(cv_text, skills_out, candidate_roles):
    """Rankea roles usando LLM con fallback determinista robusto"""
    if not candidate_roles:
        return []
    
    try:
        print(f"🔍 RANKING {len(candidate_roles)} roles candidatos...")
        
        # Crear prompt para ranking
        prompt = create_roles_ranking_prompt(cv_text, skills_out, candidate_roles)
        
        # Consultar Ollama con JSON esperado
        resp = query_ollama(prompt, expect_json=True)
        
        if not resp or "error" in resp.lower():
            print("❌ Error en respuesta LLM, usando fallback determinista")
            return rank_roles_deterministic(cv_text, skills_out, candidate_roles)
        
        # Limpiar respuesta JSON
        cleaned = clean_json_response(resp)
        
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(f"❌ Error parseando JSON: {e}, usando fallback")
            return rank_roles_deterministic(cv_text, skills_out, candidate_roles)
        
        ranking = data.get("ranking", [])
        
        if not ranking:
            print("❌ Ranking vacío, usando fallback")
            return rank_roles_deterministic(cv_text, skills_out, candidate_roles)
        
        # Validar y limpiar ranking
        seen, clean_ranking = set(), []
        for item in ranking:
            if not isinstance(item, dict):
                continue
                
            role = (item.get("role") or "").strip()
            
            try:
                fit = int(item.get("fit_percent", 0))
            except (ValueError, TypeError):
                fit = 50  # Default si no se puede convertir
            
            # Solo incluir roles válidos de la lista candidatos
            if role in candidate_roles and role not in seen:
                seen.add(role)
                clean_ranking.append({
                    "role": role, 
                    "fit_percent": max(0, min(100, fit))  # Clamp 0-100
                })
        
        # Ordenar por fit_percent descendente
        clean_ranking.sort(key=lambda x: x["fit_percent"], reverse=True)
        
        # Máximo 5 roles
        final_ranking = clean_ranking[:5]
        
        if final_ranking:
            print(f"✅ LLM ranking exitoso: {len(final_ranking)} roles")
            return final_ranking
        else:
            print("❌ Ranking filtrado vacío, usando fallback")
            return rank_roles_deterministic(cv_text, skills_out, candidate_roles)
        
    except Exception as e:
        print(f"❌ Error general en LLM ranking: {e}")
        return rank_roles_deterministic(cv_text, skills_out, candidate_roles)

def get_level_from_years_enhanced(years, has_work_context):
    """Determina nivel considerando contexto laboral"""
    if not has_work_context and years <= 2:
        return "JUNIOR"
    elif years >= 8:
        return "EXPERTO" 
    elif years >= 4:
        return "AVANZADO"
    elif years >= 2:
        return "INTERMEDIO"
    else:
        return "JUNIOR"


def extract_total_experience_from_cv(cv_text):
    """Extrae años totales de experiencia del CV"""
    patterns = [
        r'(\d+)\s*años?\s*en\s*sistemas?\s*y?\s*seguridad',
        r'(\d+)\s*años?\s*de\s*experiencia',
        r'experiencia\s*de\s*(\d+)\s*años?',
        r'(\d+)\s*years?\s*(?:of\s*)?experience',
        r'(\d+)\s*años?\s*en\s*(?:el\s*)?sector'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, cv_text, re.IGNORECASE)
        if match:
            years = int(match.group(1))
            if 1 <= years <= 40:  # Rango realista
                return years
    
    return 0


def determine_specialization_enhanced(skills_out, cv_text: str = "") -> str:
    """Determina especialización con reglas simples y señales del texto."""
    skill_names = { (s.get("name") or "").lower() for s in (skills_out or []) if (s.get("name") or "").strip() }
    cv_lower = (cv_text or "").lower()

    ai_skills         = {"ollama","mistral","llama","llama.cpp","embeddings","rag","faiss","chroma","llm","transformer"}
    automation_skills = {"automation","automatización","terraform","servicenow","jenkins","ansible","powershell","bash"}
    devops_skills     = {"docker","docker compose","compose","kubernetes","k8s","terraform","jenkins","github actions","gitlab ci","prometheus","grafana","observability","ci/cd","infrastructure","monitoring"}
    backend_skills    = {"python","flask","fastapi","django","api","rest","postgresql","mysql","sqlite","pandas"}

    ai_score         = len(skill_names & ai_skills)
    automation_score = len(skill_names & automation_skills)
    devops_score     = len(skill_names & devops_skills)
    backend_score    = len(skill_names & backend_skills)

    print(f"🔍 SCORES: AI={ai_score}, Auto={automation_score}, DevOps={devops_score}, Backend={backend_score}")

    if "ai tools engineer" in cv_lower:
        return "AI Tools Engineer"
    if ai_score >= 4 and automation_score >= 2:
        return "AI Tools Engineer"
    if ai_score >= 3:
        return "AI Engineer"
    if automation_score >= 3 or "senior systems admin" in cv_lower:
        return "Automation Engineer"
    if devops_score >= 4:
        return "DevOps Engineer"
    if backend_score >= 4:
        return "Backend Developer"
    return "Tech Generalist"


def check_work_context_enhanced(cv_text, skill_keyword):
    """Detecta contexto laboral con mayor precisión"""
    cv_lower = cv_text.lower()
    skill_lower = skill_keyword.lower()
    
    # Indicadores FUERTES de experiencia profesional
    strong_indicators = [
        "años de experiencia", "implementación de", "administración de", 
        "gestión de", "optimización", "automatización", "despliegues",
        "métricas", "estandarizar", "proyectos", "desarrollo de",
        "mejoras de", "continuidad", "stack:", "utilizado en",
        "experiencia con", "trabajado con", "especialista en",
        "senior", "administrador", "ingeniero"
    ]
    
    # Buscar skill cerca de indicadores fuertes (dentro de 150 caracteres)
    skill_positions = [m.start() for m in re.finditer(re.escape(skill_lower), cv_lower)]
    
    for pos in skill_positions:
        # Contexto amplio alrededor de la skill
        start = max(0, pos - 150)
        end = min(len(cv_lower), pos + 150)
        context = cv_lower[start:end]
        
        # Verificar indicadores fuertes
        for indicator in strong_indicators:
            if indicator in context:
                return True
    
    return False

def create_roles_ranking_prompt(cv_text, skills_out, candidate_roles):
    """
    Pide a Ollama un ranking de 3–5 roles con fit_percent (0–100).
    SOLO puede elegir de candidate_roles. Respuesta: JSON puro.
    """
    skills_lines = "\n".join(
        f"- {s.get('name','')} (años={s.get('years',0)}, nivel={s.get('level','')}, ctx={s.get('has_work_context',False)})"
        for s in (skills_out or [])
    )
    roles_lines = "\n".join(f"- {r}" for r in (candidate_roles or []))
    return (
        "Eres un evaluador de encaje de roles. Devuelve ÚNICAMENTE JSON válido y nada más.\n"
        "Elige de 3 a 5 roles de la lista CANDIDATOS (prohibido inventar roles nuevos).\n"
        "Asigna fit_percent (0..100) según el CV y sus skills/evidencias. No es necesario sumar 100.\n"
        "Ordena de mayor a menor encaje.\n\n"
        "CV:\n\"\"\"" + (cv_text or "") + "\"\"\"\n\n"
        "SKILLS:\n" + skills_lines + "\n\n"
        "CANDIDATOS:\n" + roles_lines + "\n\n"
        "SALIDA JSON OBLIGATORIA:\n"
        "{\n"
        "  \"ranking\": [\n"
        "    {\"role\": \"DevOps Engineer\", \"fit_percent\": 92},\n"
        "    {\"role\": \"Site Reliability Engineer\", \"fit_percent\": 88}\n"
        "  ]\n"
        "}\n"
    )

def rank_roles_deterministic(cv_text, skills_out, candidate_roles):
    """Ranking determinista con señales de skills + texto del CV."""
    text = (cv_text or "").lower()
    skill_tokens = { (s.get("name") or "").lower() for s in (skills_out or []) if (s.get("name") or "").strip() }

    role_signals = {
        # IA/ML
        "AI Tools Engineer": ["python","ollama","mistral","llama","llama.cpp","embeddings","rag","faiss","chroma","llm","ai","machine learning","automation","local ai","on-premises ai"],
        "AI Infrastructure Engineer": ["ollama","llm","ai infrastructure","model deployment","gpu","kubernetes","docker","terraform","ai platform","ml ops"],
        "Machine Learning Engineer": ["python","tensorflow","pytorch","scikit-learn","pandas","numpy","ml","machine learning","model training","data science"],
        "LLM Engineer": ["ollama","mistral","llama","gpt","transformer","huggingface","embeddings","rag","prompt engineering","fine-tuning"],
        "AI Platform Engineer": ["ai platform","ml platform","model serving","kubernetes","docker","terraform","ci/cd","automation","infrastructure"],
        # Automation/DevOps
        "Automation Engineer": ["python","bash","powershell","automation","scripting","terraform","ansible","jenkins","ci/cd","infrastructure automation"],
        "DevOps Engineer": ["python","bash","docker","kubernetes","terraform","ansible","jenkins","ci/cd","aws","azure","infrastructure","monitoring","github actions","gitlab ci","prometheus","grafana"],
        "Site Reliability Engineer": ["sre","reliability","monitoring","prometheus","grafana","kubernetes","incident response","automation","observability"],
        "Platform Engineer": ["platform","kubernetes","terraform","ci/cd","developer experience","internal tools","automation","infrastructure"],
        "Infrastructure Engineer": ["infrastructure","terraform","ansible","kubernetes","docker","monitoring","linux","networking","cloud"],
        # Backend
        "Backend Developer": ["python","flask","fastapi","django","api","rest","postgresql","mysql","microservices","database"],
        "Python Developer": ["python","flask","django","fastapi","pandas","requests","api development"],
        "Software Engineer": ["python","software development","programming","algorithms","system design"],
    }

    scores = []
    for role in (candidate_roles or []):
        base = 0
        for signal in role_signals.get(role, []):
            s = signal.lower()
            if s in skill_tokens:
                base += 3
            elif s in text:
                base += 1
        scores.append((role, base))

    scores = [(r, s) for r, s in scores if s > 0]
    scores.sort(key=lambda x: x[1], reverse=True)

    n = len(scores)
    top = scores[:5] if n >= 5 else scores[:max(3, n)]
    if not top:
        base = (candidate_roles or [])[:3]
        return [{"role": r, "fit_percent": p} for r, p in zip(base, [60, 50, 40])]

    max_score = top[0][1] if top else 1
    out = []
    for r, s in top:
        pct = min(99, max(30, int((s / max_score) * 100))) if max_score > 0 else 50
        out.append({"role": r, "fit_percent": pct})
    return out

def safe_int(value, default=0, min_val=None, max_val=None):
    """Convierte cualquier valor a entero de forma segura"""
    try:
        # Intentar conversión directa
        if isinstance(value, (int, float)):
            result = int(value)
        elif isinstance(value, str):
            # Limpiar string y convertir
            clean_value = value.strip()
            if clean_value.isdigit():
                result = int(clean_value)
            else:
                numbers = re.findall(r'\d+', clean_value)
                result = int(numbers[0]) if numbers else default
        else:
            result = default
            
        # Aplicar límites si se especifican
        if min_val is not None:
            result = max(min_val, result)
        if max_val is not None:
            result = min(max_val, result)
            
        return result
        
    except (ValueError, TypeError, IndexError):
        return default

def manual_skills_extraction(cv_text: str) -> dict:
    """
    Fallback robusto cuando el LLM falla: detecta skills por catálogo/keywords,
    estima años/contexto y devuelve un análisis coherente.
    """
    cv_text = cv_text or ""
    cv_lower = cv_text.lower()

    # --- helpers locales ---
    import re
    def _word_in_text(text: str, token: str) -> bool:
        if not token:
            return False
        # soporta tokens con puntos (p.ej., "llama.cpp") y guiones
        if "." in token:
            return token.lower() in text
        pat = r"(?<!\w)" + re.escape(token.lower()) + r"(?!\w)"
        return re.search(pat, text) is not None

    def _safe_int(val, default=1, min_val=0, max_val=50) -> int:
        try:
            x = int(round(float(val)))
        except Exception:
            x = default
        return max(min_val, min(max_val, x))

    # --- 1) Candidatos por catálogo/keywords ---
    detected: dict[str, dict] = {}

    # Preferimos keywords del catálogo para encontrar menciones robustas
    for skill_key, keywords in (SKILLS_CATALOG or {}).items():
        if not keywords:
            continue
        match = None
        for kw in keywords:
            if _word_in_text(cv_lower, (kw or "").strip().lower()):
                match = kw
                break
        # si no hay match por keyword, probamos por el nombre de la skill
        if not match and _word_in_text(cv_lower, (skill_key or "").strip().lower()):
            match = skill_key

        if not match:
            continue

        # Contexto laboral + años
        has_ctx = False
        years = 0
        try:
            has_ctx = check_work_context(cv_text, match)
        except Exception:
            pass

        try:
            years = estimate_years_from_context(cv_text, match)
        except Exception:
            years = 0

        if not has_ctx:
            years = min(years, 2)  # sin evidencia laboral → cap 2 años

        years = _safe_int(years, default=1, min_val=0, max_val=40)

        # Nivel
        try:
            level = get_level_from_years(years)
        except Exception:
            level = "INTERMEDIO" if years >= 2 else "JUNIOR"

        # Nombre visual
        try:
            disp = format_skill_name(skill_key)
        except Exception:
            disp = str(skill_key).strip()

        key = disp.lower()
        if key not in detected:
            detected[key] = {
                "name": disp,
                "years": years,
                "level": level,
                "has_work_context": bool(has_ctx),
            }

    skills_out = list(detected.values())

    # --- 2) Especialización, métricas y reportes con funciones existentes ---
    try:
        specialization = determine_specialization(skills_out)
    except Exception:
        specialization = "No detectada"

    total_years = 1
    try:
        total_years = max([s["years"] for s in skills_out if s.get("has_work_context")], default=1)
    except Exception:
        total_years = 1

    try:
        job_focus = generate_job_focus([s["name"] for s in skills_out], total_years, cv_text=cv_text)
    except Exception:
        job_focus = {}

    try:
        rrhh_report = generate_rrhh_report(skills_out, cv_text)
    except Exception:
        rrhh_report = {}

    try:
        empleabilidad = calculate_employability(skills_out, total_years)
    except Exception:
        empleabilidad = {}

    try:
        balance = analyze_skills_balance(skills_out)
    except Exception:
        balance = {}

    return {
        "skills": skills_out,
        "especializacion_principal": specialization,
        "especializacion_secundaria": "No detectada",
        "años_experiencia_total": total_years,
        "fortalezas_principales": [s["name"] for s in skills_out[:3] if s.get("has_work_context")],
        "enfoque_busqueda": job_focus,
        "nivel_empleabilidad": empleabilidad,
        "balance_skills": balance,
        "informe_rrhh": rrhh_report,
    }
def generate_rrhh_report(skills, cv_text):
    """Genera un informe detallado desde la perspectiva de RRHH"""
    
    # Agrupar skills por nivel
    experto_skills = [s for s in skills if s["level"] == "EXPERTO" and s["has_work_context"]]
    avanzado_skills = [s for s in skills if s["level"] == "AVANZADO" and s["has_work_context"]]
    intermedio_skills = [s for s in skills if s["level"] == "INTERMEDIO"]
    junior_skills = [s for s in skills if s["level"] == "JUNIOR" or not s["has_work_context"]]
    
    report = {}
    
    # Skills EXPERTO
    if experto_skills:
        positions = extract_work_positions(cv_text)
        experto_names = [s["name"] for s in experto_skills]
        
        report["experto"] = {
            "skills": experto_names,
            "motivo": f"Has trabajado con estas tecnologías durante más de 8 años en posiciones como {', '.join(positions[:2])}. Experiencia sólida y demostrable en entornos profesionales."
        }
    
    # Skills AVANZADO
    if avanzado_skills:
        avanzado_names = [s["name"] for s in avanzado_skills]
        
        report["avanzado"] = {
            "skills": avanzado_names,
            "motivo": f"Experiencia de 4-8 años con estas tecnologías en proyectos reales. Nivel competente que permite trabajar de forma independiente."
        }
    
    # Skills INTERMEDIO
    if intermedio_skills:
        intermedio_names = [s["name"] for s in intermedio_skills]
        
        report["intermedio"] = {
            "skills": intermedio_names,
            "motivo": "Experiencia de 2-4 años o uso ocasional en proyectos. Conocimiento funcional pero requiere supervisión en tareas complejas."
        }
    
    # Skills JUNIOR/Inexistentes
    if junior_skills:
        junior_names = [s["name"] for s in junior_skills]
        
        report["junior"] = {
            "skills": junior_names,
            "motivo": "Las mencionas en tu CV pero no hay evidencia de uso profesional significativo. RRHH podría cuestionar tu nivel real con estas tecnologías."
        }
    
    # Conclusión del mentor
    total_skills = len(skills)
    strong_skills = len(experto_skills) + len(avanzado_skills)
    weak_skills = len(junior_skills)
    
    if weak_skills > strong_skills:
        conclusion = f"⚠️ Tienes {weak_skills} skills sin evidencia laboral vs {strong_skills} skills sólidas. Considera eliminar skills sin experiencia real o añadir contexto específico de dónde las has usado."
    elif strong_skills >= total_skills * 0.7:
        conclusion = f"✅ Excelente perfil: {strong_skills} de {total_skills} skills tienen base sólida. Tu CV pasará filtros de RRHH sin problemas."
    else:
        conclusion = f"🔄 Perfil balanceado: {strong_skills} skills fuertes de {total_skills} totales. Considera potenciar algunas skills junior con ejemplos específicos."
    
    report["conclusion"] = conclusion
    
    return report

def extract_work_positions(cv_text):
    """Extrae posiciones de trabajo del CV"""
    positions = []
    
    # Buscar patrones de títulos de trabajo
    position_patterns = [
        r"ingeniero[a-z\s]*",
        r"administrador[a-z\s]*",
        r"desarrollador[a-z\s]*",
        r"analista[a-z\s]*",
        r"especialista[a-z\s]*",
        r"técnico[a-z\s]*"
    ]
    
    cv_lower = cv_text.lower()
    for pattern in position_patterns:
        matches = re.findall(pattern, cv_lower)
        for match in matches:
            clean_position = match.strip().title()
            if clean_position and len(clean_position) > 5:
                positions.append(clean_position)
    
    # Eliminar duplicados y devolver max 3
    unique_positions = list(dict.fromkeys(positions))
    return unique_positions[:3] if unique_positions else ["Técnico IT"]

def check_work_context(cv_text, skill_keyword):
    """Verifica si una skill tiene contexto laboral real en el CV"""
    cv_lower = cv_text.lower()
    
    # Secciones que indican experiencia laboral
    work_sections = [
        "experiencia profesional",
        "experiencia laboral", 
        "trabajo",
        "empleado",
        "ingeniero",
        "administrador",
        "desarrollador",
        "analista"
    ]
    
    # Palabras que indican uso real/profesional
    professional_indicators = [
        "años de experiencia",
        "proyectos",
        "automatización de",
        "desarrollo de",
        "implementación de",
        "gestión de",
        "administración de",
        "scripts",
        "sistemas",
        "infraestructura"
    ]
    
    # Buscar el skill cerca de indicadores profesionales
    skill_contexts = []
    lines = cv_text.split('\n')
    
    for i, line in enumerate(lines):
        if skill_keyword in line.lower():
            # Verificar líneas adyacentes para contexto
            context_lines = lines[max(0, i-2):min(len(lines), i+3)]
            context_text = ' '.join(context_lines).lower()
            
            # Verificar si está en contexto profesional
            for indicator in professional_indicators:
                if indicator in context_text:
                    return True
            
            # Verificar si está en sección de experiencia
            for section in work_sections:
                if section in context_text:
                    return True
    
    return False

def estimate_years_from_context(cv_text, keyword):
    """Estima años de experiencia basado en contexto muy básico"""
    # Buscar patrones como "3 años de Python", "Python (2 años)"
    patterns = [
        rf"{keyword}.*?(\d+)\s*años?",
        rf"(\d+)\s*años?.*?{keyword}",
        rf"{keyword}.*?(\d+)\s*year",
        rf"(\d+)\s*year.*?{keyword}"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, cv_text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    
    # Si no encuentra años específicos, estimación básica
    cv_lower = cv_text.lower()
    if "senior" in cv_lower or "experto" in cv_lower:
        return 5
    elif "junior" in cv_lower or "principiante" in cv_lower:
        return 1
    else:
        return 2  # Default

def get_level_from_years(years):
    """Convierte años en nivel de competencia"""
    if years >= 5:
        return "EXPERTO"
    elif years >= 2:
        return "AVANZADO"
    else:
        return "JUNIOR"

def analyze_skills_balance(skills):
    """Analiza el balance entre skills con/sin contexto laboral"""
    total_skills = len(skills)
    skills_with_context = sum(1 for s in skills if s.get("has_work_context", False))
    skills_without_context = total_skills - skills_with_context
    
    if total_skills == 0:
        return {
            "status": "sin_skills",
            "mensaje": "No se detectaron skills técnicas específicas.",
            "tipo": "warning"
        }
    
    percentage_without_context = (skills_without_context / total_skills) * 100
    
    if percentage_without_context > 60:
        return {
            "status": "desbalanceado",
            "mensaje": f"Detecté {skills_without_context} skills técnicas que no aparecen en tu experiencia profesional. Considera añadir ejemplos específicos de dónde las has usado laboralmente para demostrar tu experiencia real.",
            "tipo": "warning",
            "skills_sin_contexto": [s["name"] for s in skills if not s.get("has_work_context", False)]
        }
    elif percentage_without_context < 20:
        return {
            "status": "bien_balanceado",
            "mensaje": "Excelente organización de skills. Todas tus competencias técnicas están respaldadas por experiencia profesional demostrable.",
            "tipo": "success"
        }
    else:
        return {
            "status": "aceptable",
            "mensaje": "Buen balance entre skills mencionadas y experiencia laboral. Podrías considerar añadir más contexto a algunas herramientas.",
            "tipo": "info"
        }

def format_skill_name(skill_key):
    """Formatea el nombre de la skill para mostrar de forma más legible"""
    skill_names = {
        "always_on": "SQL Always On",
        "hyper_v": "Hyper-V",
        "ethical_hacking": "Ethical Hacking",
        "ai": "Inteligencia Artificial",
        "observability": "Observabilidad",
        "csharp": "C#",
        "sql_server": "SQL Server",
        "react_native": "React Native",
        "scikit_learn": "Scikit-Learn"
    }
    
    return skill_names.get(skill_key, skill_key.replace('_', ' ').title())

def check_work_context(cv_text, skill_keyword):
    """Verifica si una skill tiene contexto laboral real en el CV"""
    cv_lower = cv_text.lower()
    
    # Secciones que indican experiencia laboral
    work_sections = [
        "experiencia profesional",
        "experiencia laboral", 
        "trabajo",
        "empleado",
        "ingeniero",
        "administrador",
        "desarrollador",
        "analista"
    ]
    
    # Palabras que indican uso real/profesional
    professional_indicators = [
        "años de experiencia",
        "proyectos",
        "automatización de",
        "desarrollo de",
        "implementación de",
        "gestión de",
        "administración de",
        "scripts",
        "sistemas",
        "infraestructura"
    ]
    
    # Buscar el skill cerca de indicadores profesionales
    skill_contexts = []
    lines = cv_text.split('\n')
    
    for i, line in enumerate(lines):
        if skill_keyword in line.lower():
            # Verificar líneas adyacentes para contexto
            context_lines = lines[max(0, i-2):min(len(lines), i+3)]
            context_text = ' '.join(context_lines).lower()
            
            # Verificar si está en contexto profesional
            for indicator in professional_indicators:
                if indicator in context_text:
                    return True
            
            # Verificar si está en sección de experiencia
            for section in work_sections:
                if section in context_text:
                    return True
    
    return False
# ---------------------------
# Métricas de rendimiento
# ---------------------------
proc = psutil.Process(os.getpid())

def log_metrics(endpoint: str = "", method: str = "", status: int | None = None,
                start_time: float | None = None, *, stage: str | None = None, **_unused) -> None:
    """
    Métricas ligeras de proceso/sistema. Soporta:
      - log_metrics(request.path, request.method, response.status_code, start_time)
      - log_metrics(stage='cv_html:start')
    No requiere variables globales previas; se auto-inicializa en el primer uso.
    """
    # Lazy init (una sola vez)
    if not hasattr(log_metrics, "_inited"):
        try:
            import psutil as _ps
            log_metrics._psutil = _ps
            log_metrics._proc = _ps.Process(os.getpid())
        except Exception:
            log_metrics._psutil = None
            log_metrics._proc = None
        try:
            import GPUtil as _gpu
            log_metrics._gputil = _gpu
        except Exception:
            log_metrics._gputil = None
        log_metrics._inited = True

    _psutil = getattr(log_metrics, "_psutil", None)
    _proc   = getattr(log_metrics, "_proc", None)
    _gputil = getattr(log_metrics, "_gputil", None)

    now = time.perf_counter()
    dt_ms = ((now - start_time) * 1000) if isinstance(start_time, (int, float)) else None

    # Proceso
    try:
        rss_mb = _proc.memory_info().rss / (1024 * 1024) if _proc else 0.0
    except Exception:
        rss_mb = 0.0
    try:
        cpu_p = _proc.cpu_percent(interval=0.0) if _proc else 0.0
    except Exception:
        cpu_p = 0.0

    # Sistema
    try:
        sys_cpu = _psutil.cpu_percent(interval=0.0) if _psutil else 0.0
        sys_ram = _psutil.virtual_memory().percent if _psutil else 0.0
    except Exception:
        sys_cpu = 0.0
        sys_ram = 0.0

    # GPU (opcional)
    gpu_txt = ""
    if _gputil:
        try:
            gpus = _gputil.getGPUs()
            if gpus:
                g = gpus[0]
                gpu_txt = f" gpu%={g.load*100:.1f} vram%={g.memoryUtil*100:.1f}"
        except Exception:
            pass

    parts = ["[METRICS]"]
    if stage:   parts.append(f"stage={stage}")
    if endpoint:parts.append(f"endpoint={endpoint}")
    if method:  parts.append(f"method={method}")
    if status is not None: parts.append(f"status={status}")
    if dt_ms is not None:  parts.append(f"dt={dt_ms:.1f}ms")
    parts.append(f"rss_mb={rss_mb:.1f} cpu%={cpu_p:.1f} sys_cpu%={sys_cpu:.1f} sys_ram%={sys_ram:.1f}{gpu_txt}")

    print(" ".join(parts))

def analyze_skills_balance(skills):
    """Analiza el balance entre skills con/sin contexto laboral"""
    total_skills = len(skills)
    skills_with_context = sum(1 for s in skills if s["has_work_context"])
    skills_without_context = total_skills - skills_with_context
    
    if total_skills == 0:
        return {
            "status": "sin_skills",
            "mensaje": "No se detectaron skills técnicas específicas.",
            "tipo": "warning"
        }
    
    percentage_without_context = (skills_without_context / total_skills) * 100
    
    if percentage_without_context > 60:
        return {
            "status": "desbalanceado",
            "mensaje": f"Detecté {skills_without_context} skills técnicas que no aparecen en tu experiencia profesional. Considera añadir ejemplos específicos de dónde las has usado laboralmente para demostrar tu experiencia real.",
            "tipo": "warning",
            "skills_sin_contexto": [s["name"] for s in skills if not s["has_work_context"]]
        }
    elif percentage_without_context < 20:
        return {
            "status": "bien_balanceado",
            "mensaje": "Excelente organización de skills. Todas tus competencias técnicas están respaldadas por experiencia profesional demostrable.",
            "tipo": "success"
        }
    else:
        return {
            "status": "aceptable",
            "mensaje": "Buen balance entre skills mencionadas y experiencia laboral. Podrías considerar añadir más contexto a algunas herramientas.",
            "tipo": "info"
        }

def estimate_years_from_context(cv_text, keyword):
    """Estima años de experiencia basado en contexto muy básico"""
    # Buscar patrones como "3 años de Python", "Python (2 años)"
    import re
    patterns = [
        rf"{keyword}.*?(\d+)\s*años?",
        rf"(\d+)\s*años?.*?{keyword}",
        rf"{keyword}.*?(\d+)\s*year",
        rf"(\d+)\s*year.*?{keyword}"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, cv_text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    
    # Si no encuentra años específicos, estimación básica
    cv_lower = cv_text.lower()
    if "senior" in cv_lower or "experto" in cv_lower:
        return 5
    elif "junior" in cv_lower or "principiante" in cv_lower:
        return 1
    else:
        return 2  # Default

def get_level_from_years(years):
    """Convierte años en nivel de competencia"""
    if years >= 5:
        return "EXPERTO"
    elif years >= 2:
        return "AVANZADO"
    else:
        return "JUNIOR"

def determine_specialization(skills, cv_text=""):
    """Determina especialización con más precisión"""
    skill_names = {s["name"].lower() for s in skills}
    cv_lower = (cv_text or "").lower()
    
    # IA/ML Engineer
    ai_keywords = ["ollama", "mistral", "llama", "embeddings", "rag", "llm", "ai", "artificial intelligence"]
    ai_score = sum(1 for kw in ai_keywords if kw in cv_lower or kw in skill_names)
    
    # Automation Engineer  
    automation_keywords = ["automation", "automatización", "scripts", "jenkins", "terraform", ">50", "ahorro"]
    automation_score = sum(1 for kw in automation_keywords if kw in cv_lower)
    
    # DevOps Engineer
    devops_keywords = ["docker", "kubernetes", "aws", "azure", "linux", "bash", "powershell", "infrastructure"]
    devops_score = sum(1 for kw in devops_keywords if kw in skill_names)
    
    # Backend Developer
    backend_keywords = ["python", "flask", "fastapi", "api", "postgresql", "sql"]
    backend_score = sum(1 for kw in backend_keywords if kw in skill_names)
    
    # Determinar especialización principal
    if ai_score >= 3 and automation_score >= 2:
        return "AI Tools Engineer"
    elif automation_score >= 3 or "senior systems admin" in cv_lower:
        return "Senior Systems Administrator"  
    elif devops_score >= 3:
        return "DevOps Engineer"
    elif backend_score >= 3:
        return "Backend Developer"
    else:
        return "Tech Generalist"

def calculate_employability(skills, total_years):
    """Calcula nivel de empleabilidad basado en skills y experiencia"""
    skill_count = len(skills)
    
    if total_years >= 8 and skill_count >= 4:
        return "ALTO"
    elif total_years >= 3 and skill_count >= 3:
        return "MEDIO"
    else:
        return "BAJO"



def generate_rrhh_report(skills, cv_text):
    """Genera un informe detallado desde la perspectiva de RRHH"""
    
    # Agrupar skills por nivel
    experto_skills = [s for s in skills if s["level"] == "EXPERTO" and s.get("has_work_context", False)]
    avanzado_skills = [s for s in skills if s["level"] == "AVANZADO" and s.get("has_work_context", False)]
    intermedio_skills = [s for s in skills if s["level"] == "INTERMEDIO"]
    junior_skills = [s for s in skills if s["level"] == "JUNIOR" or not s.get("has_work_context", False)]
    
    report = {}
    
    # Skills EXPERTO
    if experto_skills:
        positions = extract_work_positions(cv_text)
        experto_names = [s["name"] for s in experto_skills]
        
        report["experto"] = {
            "skills": experto_names,
            "motivo": f"Has trabajado con estas tecnologías durante más de 8 años en posiciones como {', '.join(positions[:2])}. Experiencia sólida y demostrable en entornos profesionales."
        }
    
    # Skills AVANZADO
    if avanzado_skills:
        avanzado_names = [s["name"] for s in avanzado_skills]
        
        report["avanzado"] = {
            "skills": avanzado_names,
            "motivo": f"Experiencia de 4-8 años con estas tecnologías en proyectos reales. Nivel competente que permite trabajar de forma independiente."
        }
    
    # Skills INTERMEDIO
    if intermedio_skills:
        intermedio_names = [s["name"] for s in intermedio_skills]
        
        report["intermedio"] = {
            "skills": intermedio_names,
            "motivo": "Experiencia de 2-4 años o uso ocasional en proyectos. Conocimiento funcional pero requiere supervisión en tareas complejas."
        }
    
    # Skills JUNIOR/Inexistentes
    if junior_skills:
        junior_names = [s["name"] for s in junior_skills]
        
        report["junior"] = {
            "skills": junior_names,
            "motivo": "Las mencionas en tu CV pero no hay evidencia de uso profesional significativo. RRHH podría cuestionar tu nivel real con estas tecnologías."
        }
    
    # Conclusión del mentor
    total_skills = len(skills)
    strong_skills = len(experto_skills) + len(avanzado_skills)
    weak_skills = len(junior_skills)
    
    if weak_skills > strong_skills:
        conclusion = f"⚠️ Tienes {weak_skills} skills sin evidencia laboral vs {strong_skills} skills sólidas. Considera eliminar skills sin experiencia real o añadir contexto específico de dónde las has usado."
    elif strong_skills >= total_skills * 0.7:
        conclusion = f"✅ Excelente perfil: {strong_skills} de {total_skills} skills tienen base sólida. Tu CV pasará filtros de RRHH sin problemas."
    else:
        conclusion = f"🔄 Perfil balanceado: {strong_skills} skills fuertes de {total_skills} totales. Considera potenciar algunas skills junior con ejemplos específicos."
    
    report["conclusion"] = conclusion
    
    return report

def extract_work_positions(cv_text):
    """Extrae posiciones de trabajo del CV"""
    positions = []
    
    # Buscar patrones de títulos de trabajo
    position_patterns = [
        r"ingeniero[a-z\s]*",
        r"administrador[a-z\s]*",
        r"desarrollador[a-z\s]*",
        r"analista[a-z\s]*",
        r"especialista[a-z\s]*",
        r"técnico[a-z\s]*"
    ]
    
    cv_lower = cv_text.lower()
    for pattern in position_patterns:
        import re
        matches = re.findall(pattern, cv_lower)
        for match in matches:
            clean_position = match.strip().title()
            if clean_position and len(clean_position) > 5:
                positions.append(clean_position)
    
    # Eliminar duplicados y devolver max 3
    unique_positions = list(dict.fromkeys(positions))
    return unique_positions[:3] if unique_positions else ["Técnico IT"]

def calculate_employability(skills, total_years):
    """Calcula nivel de empleabilidad basado en skills y experiencia"""
    skill_count = len(skills)
    
    if total_years >= 8 and skill_count >= 4:
        return "ALTO"
    elif total_years >= 3 and skill_count >= 3:
        return "MEDIO"
    else:
        return "BAJO"


def create_feedback_prompt(analysis_data: dict, cv_text: str) -> str:
    esp   = analysis_data.get("especializacion_principal", "N/D")
    years = analysis_data.get("años_experiencia_total", 0)
    skills = [s.get("name","") for s in analysis_data.get("skills", [])]
    skills_join = ", ".join(sorted({s for s in skills if s}))

    return (
        "Eres Director Técnico y Responsable de RRHH. Responde ÚNICAMENTE con JSON válido en castellano.\n"
        "Devuelve exactamente esta forma (sin texto extra):\n"
        "{\n"
        "  \"fortalezas\": [\"(máx 6, frases cortas)\"] ,\n"
        "  \"mejoras\": [\"(máx 5, acciones concretas)\"] ,\n"
        "  \"keywords_ats\": [\"(máx 20, snake_case o kebab-case)\"]\n"
        "}\n\n"
        "Reglas:\n"
        "- Siempre en castellano.\n"
        "- Frases de 1 línea, sin adornos.\n"
        "- No repitas ni inventes skills que no existan en el CV.\n"
        "- Resume el stack en componentes nucleares (p.ej., python, flask, fastapi, ollama, faiss, chroma, pandas, sqlite, postgresql, logging_estructurado, pytest).\n\n"
        f"Especialización: {esp}\n"
        f"Años experiencia: {years}\n"
        f"Skills detectadas: {skills_join}\n\n"
        "CV (texto):\n\"\"\"" + (cv_text or "")[:8000] + "\"\"\"\n"
    )



def query_ollama(
    prompt: str,
    expect_json: bool = False,
    *,
    num_predict: int | None = None,   # tokens de salida
    timeout: int = 90,                # s
    max_chars: int = 12000,           # recorte defensivo del prompt
    options: dict | None = None       # overrides finos si los necesitas
) -> str:
    """Consulta a Ollama. Devuelve SIEMPRE texto (la capa superior ya parsea si hace falta)."""
    import time
    t0 = time.perf_counter()

    p = prompt or ""
    if len(p) > max_chars:
        p = p[:max_chars]

    # Defaults pensados para JSON “largo” y robusto
    if num_predict is None:
        # Para respuestas JSON evita cortes: más tokens que en texto libre
        num_predict = 1024 if expect_json else 256

    base_opts = {
        "temperature": 0.2,       # 0.0 puede ser más frágil; 0.2 sigue siendo preciso
        "top_k": 20,
        "top_p": 0.9,
        "num_predict": num_predict,
        # Contexto amplio para CVs grandes; se puede sobreescribir vía options
        "num_ctx": 8192 if expect_json else 4096,
        # "stop": ["```", "\n\n\n"],  # opcional si el modelo insiste en fences
    }
    if options:
        base_opts.update(options)

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": p,
        "stream": False,
        "options": base_opts,
        "keep_alive": 300,  # 5 min
    }
    if expect_json:
        payload["format"] = "json"

    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        r.raise_for_status()
        try:
            data = r.json()
            out = data.get("response", "")
            return out if isinstance(out, str) else str(out or "")
        except ValueError:
            # Servidor devolvió texto plano: devolver tal cual
            return r.text or ""
    except requests.exceptions.ConnectionError:
        return "Error: No se puede conectar con Ollama. ¿Está ejecutándose?"
    except requests.exceptions.Timeout:
        return "Error: Timeout esperando respuesta de Ollama."
    except requests.exceptions.RequestException as e:
        body = getattr(e.response, "text", "")
        return f"Error consultando Ollama: {e} {body[:200]}"
    finally:
        try:
            log_metrics(stage=f"ollama:{'json' if expect_json else 'text'}", start_time=t0)
        except Exception:
            pass


def create_skills_analysis_prompt(cv_text: str) -> str:
    return (
        "Eres un extractor de skills técnicas. Devuelve SOLO JSON válido. "
        "Máximo 20 skills; cada 'evidence' ≤ 160 caracteres. "
        "Campos: total_experience_years:int, skills:[{name:str, evidence:str}].\n\n"
        f"CV:\n\"\"\"{cv_text}\"\"\"\n\n"
        "RESPUESTA JSON (sin texto adicional): "
        "{ \"total_experience_years\": 0, \"skills\": [ {\"name\":\"Python\",\"evidence\":\"...\"} ] }"
    )


def calcular_match(skills_cv, skills_oferta):
    comunes = set(skills_cv) & set(skills_oferta)
    return round((len(comunes) / len(skills_oferta)) * 100, 1), list(comunes)

def sugerir_mejoras_oferta(skills_cv, skills_oferta, titulo_oferta):
    prompt = f"""
    Eres un asesor de carrera tecnológica.
    Analiza este CV (skills: {skills_cv}) frente a esta oferta "{titulo_oferta}" (skills requeridas: {skills_oferta}).
    Devuelve EXACTAMENTE una lista en JSON con 2 o 3 acciones concretas para mejorar el encaje.
    Responde en castellano, breve y sin adornos.
    """
    try:
        respuesta = ollama.chat(
            model="mistral",
            messages=[{"role": "user", "content": prompt}]
        )
        acciones = json.loads(respuesta['message']['content'])
    except Exception:
        acciones = ["Error al procesar sugerencias"]

    return acciones
def allowed_file(filename):
    """Verifica si el archivo tiene una extensión permitida"""
    ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# === CV OPTIMIZADO ===

def build_cv_prompt(role: str, analysis: dict, cv_text: str) -> str:
    """
    Pedimos JSON ESTRICTO con el contenido del CV ya reescrito para un ROLE específico.
    """
    esp   = analysis.get("especializacion_principal", "N/D")
    years = analysis.get("años_experiencia_total", 0)
    skills = [s.get("name","") for s in analysis.get("skills", [])]
    skills_join = ", ".join(sorted({s for s in skills if s}))

    return (
        "Eres un experto en RRHH y un Hiring Manager técnico. Responde ÚNICAMENTE con JSON válido.\n"
        "Objetivo: generar los textos de un CV optimizado para el rol indicado.\n"
        "Idioma: español.\n"
        "No inventes empleadores ni fechas. Si faltan datos, escribe \"(dato no disponible)\".\n\n"
        "FORMATO JSON ESTRICTO:\n"
        "{\n"
        "  \"titulo\": \"<AI Tools Engineer / DevOps Engineer / ...>\",\n"
        "  \"resumen\": \"<párrafo 3-5 líneas, orientado al rol>\",\n"
        "  \"skills_destacadas\": [\"Python\", \"Automation\", \"Flask\", ...],\n"
        "  \"experiencia_bullets\": [\n"
        "      \"<bullet de impacto con métricas si existen>\",\n"
        "      \"...\"\n"
        "  ],\n"
        "  \"proyectos_bullets\": [\"...\"],\n"
        "  \"formacion_bullets\": [\"...\"],\n"
        "  \"extras_bullets\": [\"Certificaciones / publicaciones si existen\"]\n"
        "}\n\n"
        f"ROL OBJETIVO: {role}\n"
        f"Especialización detectada: {esp}\n"
        f"Años experiencia: {years}\n"
        f"Skills detectadas: {skills_join or '(ninguna)'}\n\n"
        "CV (texto original):\n\"\"\"" + (cv_text or "") + "\"\"\"\n"
    )
# Conjuntos para matching rápido en oferta
JD_SKILL_SET = {k.strip().lower() for k in SKILLS_CATALOG.keys()}
JD_SKILL_SYNONYMS = {kw.strip().lower()
                     for vals in SKILLS_CATALOG.values() for kw in (vals or [])
                     if isinstance(kw, str) and kw.strip()}
ALL_JD_TOKENS = JD_SKILL_SET | JD_SKILL_SYNONYMS

def jd_extract_with_llm(jd_text: str) -> dict:
    """
    Determinista (sin LLM).
    - Título = primera línea no vacía.
    - Sección core: tras 'Requisitos/Responsabilidades/Funciones/Imprescindible...'
    - Sección periféricas: tras 'Valorable/Deseable/Plus/Nice to have...'
    - Usa extract_skills_from_text(strict=True) para evitar alias genéricos.
    """
    raw_lines = [ln.strip() for ln in jd_text.splitlines()]
    titulo = next((ln for ln in raw_lines if ln), "Oferta")

    core_keys = [
        "requisitos", "requerimientos", "responsabilidades", "funciones",
        "imprescindible", "must", "qué buscamos", "que buscamos"
    ]
    per_keys = [
        "valorable", "deseable", "plus", "deseados", "nice to have"
    ]

    def _collect_after(keys):
        out = []
        lines = jd_text.splitlines()
        capture = False
        for ln in lines:
            ln_l = ln.lower()
            if any(k in ln_l for k in keys):
                capture = True
                continue
            if capture and any(k in ln_l for k in core_keys + per_keys):
                break
            if capture:
                out.append(ln)
        return "\n".join(out) if out else ""

    core_block = _collect_after(core_keys)
    per_block  = _collect_after(per_keys)

    # Extracción estricta para evitar falsos positivos
    core_hits = extract_skills_from_text(core_block or "", strict=True)
    per_hits  = extract_skills_from_text(per_block  or "", strict=True)

    # Si no hay secciones claras, intentamos todo el texto (estricto)
    if not core_hits and not per_hits:
        all_hits = extract_skills_from_text(jd_text, strict=True)
        core_hits, per_hits = all_hits, set()

    # Evitar solapes
    per_hits = per_hits - core_hits

    return {
        "titulo": titulo,
        "core_skills": sorted(core_hits),
        "peripheral_skills": sorted(per_hits)
    }

# 2) FUNCIÓN: compute_fit — (si ya la tienes igual, no toques nada)
def compute_fit(cv_skills_norm, jd_core, jd_per):
    """
    Encaje por coincidencia de skills canónicas.
    Peso: 70% core, 30% periféricas.
    ATS pass: fit>=60 y cobertura core>=50%.
    """
    cv_can   = _canonize_list(cv_skills_norm)
    core_can = _canonize_list(jd_core)
    per_can  = _canonize_list(jd_per)

    core_match   = sorted(core_can & cv_can)
    core_faltan  = sorted(core_can - cv_can)
    per_match    = sorted(per_can & cv_can)
    per_faltan   = sorted(per_can - cv_can)

    total_core = len(core_can)
    total_per  = len(per_can)

    core_score = (len(core_match) / total_core * 100.0) if total_core else 0.0
    per_score  = (len(per_match) / total_per  * 100.0) if total_per  else 0.0

    fit_percent = int(round(core_score * 0.7 + per_score * 0.3))
    ats_pass = (fit_percent >= 60) and (core_score >= 50.0 if total_core else False)

    return {
        "ats_pass": ats_pass,
        "fit_percent": fit_percent,
        "skills": {
            "core_match": core_match,
            "core_faltan": core_faltan,
            "perifericas_match": per_match,
            "perifericas_faltan": per_faltan
        }
    }


# 3) FUNCIÓN: jd_short_analysis_with_llm — fuerza consejos según umbral 75%
def jd_short_analysis_with_llm(jd_text: str, cv_skills_norm: list[str], titulo: str):
    cv_can = _canonize_list(cv_skills_norm)
    jd     = jd_extract_with_llm(jd_text)
    core   = set(_canonize_list(jd.get("core_skills", [])))
    per    = set(_canonize_list(jd.get("peripheral_skills", [])))

    core_miss = sorted(core - cv_can)
    per_miss  = sorted(per - cv_can)
    core_hit  = sorted(core & cv_can)
    per_hit   = sorted(per & cv_can)

    # Fit local para redactar los mensajes (mismo criterio que compute_fit)
    total_core = len(core)
    total_per  = len(per)
    core_score = (len(core_hit) / total_core * 100.0) if total_core else 0.0
    per_score  = (len(per_hit)  / total_per  * 100.0) if total_per  else 0.0
    fit_local  = int(round(core_score * 0.7 + per_score * 0.3))

    analisis = (
        f"Comparación de tu CV contra la oferta '{titulo}': "
        f"encaje aproximado {fit_local}%. Faltan {len(core_miss)} núcleo y {len(per_miss)} periféricas."
        if (core or per) else
        "No se detectaron requisitos técnicos claros en la oferta."
    )

    consejos = []
    if core or per:
        if fit_local >= 75:
            # 3 consejos para empujar al 85–90% (prioriza núcleo, luego periféricas)
            if core_miss:
                consejos.append(f"Refuerza 1: añade {', '.join(core_miss[:1])} con ejemplo real de uso.")
            if len(core_miss) >= 2:
                consejos.append(f"Refuerza 2: cubre {', '.join(core_miss[1:2])} con evidencia (logro/valor).")
            if per_miss:
                consejos.append(f"Completa stack: {', '.join(per_miss[:1])} (mini proyecto + README).")
            if not consejos:
                consejos = [
                    "Ajusta el título del CV al rol exacto de la oferta.",
                    "Destaca las 3 skills clave en el primer bloque del CV.",
                    "Añade 1 logro medible directamente relacionado."
                ]
            consejos = consejos[:3]
        else:
            # Encaje < 75%: recomendaciones claras y accionables según faltantes
            if core_miss:
                consejos.append(f"Cubre núcleo faltante: {', '.join(core_miss[:5])}.")
            if per_miss:
                consejos.append(f"Periféricas a considerar: {', '.join(per_miss[:5])}.")
            if not consejos:
                consejos.append("Reordena el CV destacando experiencia y resultados en tecnologías del rol.")
    else:
        consejos.append("La oferta no lista requisitos técnicos reconocibles; pega el texto completo o busca otra fuente.")

    return analisis, consejos[:5]
def _extract_cv_plaintext_from_analysis(analysis: dict) -> str:
    if not isinstance(analysis, dict):
        return ""
    raw = (analysis.get("cv_text_raw") or analysis.get("cv_text") or "").strip()
    if raw:
        return raw
    chunks = []
    titulo = (analysis.get("titulo_cv") or analysis.get("headline") or "").strip()
    if titulo:
        chunks.append(f"Título CV: {titulo}")
    exp = analysis.get("experiencia") or []
    if isinstance(exp, list) and exp:
        mini = []
        for e in exp[:6]:
            if not isinstance(e, dict): 
                continue
            rol = (e.get("rol") or e.get("puesto") or "").strip()
            comp= (e.get("empresa") or "").strip()
            desc= (e.get("descripcion") or "").strip()
            line = " • ".join([z for z in [rol, comp, desc] if z])
            if line:
                mini.append(line)
        if mini:
            chunks.append("Experiencia:\n" + "\n".join(mini))
    skills = analysis.get("full_skills") or analysis.get("skills_detectadas") or analysis.get("skills") or []
    if isinstance(skills, list) and skills:
        chunks.append("Skills: " + ", ".join([str(s) for s in skills][:80]))
    return "\n\n".join(chunks).strip()

def _parse_llm_json(text: str) -> dict:
    """
    Parser robusto para salidas del LLM:
    - Acepta ```json ... ``` o texto con el JSON embebido.
    - Normaliza comillas tipográficas y comas colgantes.
    - Reintenta con JSON, luego con literal_eval y, si todo falla, extrae por regex.
    """
    import json, re, ast

    raw = (text or "").strip()
    if not raw:
        return {}

    # 1) Si viene en fences ```json ... ```
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw, flags=re.I)
    if m:
        raw = m.group(1).strip()

    # 2) Si hay múltiples cosas, quédate con el 1er bloque {...}
    if not raw.lstrip().startswith("{"):
        m2 = re.search(r"\{[\s\S]*\}", raw)
        if m2:
            raw = m2.group(0).strip()

    # 3) Normalizaciones
    s = raw
    # comillas tipográficas → rectas
    s = (s.replace("\u201c", '"').replace("\u201d", '"').replace("\u201e", '"')
           .replace("\u2018", "'").replace("\u2019", "'"))
    # quitar comas colgantes: ,]
    s = re.sub(r",(\s*[\]}])", r"\1", s)
    # eliminar BOM o rarezas
    s = s.replace("\ufeff", "")

    # 4) Primer intento: JSON puro
    try:
        return json.loads(s)
    except Exception:
        pass

    # 5) Segundo intento: si hay más ' que " probablemente es estilo Python → cambia a "
    try2 = s
    if s.count('"') < s.count("'"):
        try2 = s.replace("'", '"')
        try:
            return json.loads(try2)
        except Exception:
            pass

    # 6) Tercer intento: literal_eval tras mapear true/false/null
    pyish = re.sub(r"\btrue\b", "True", s, flags=re.I)
    pyish = re.sub(r"\bfalse\b", "False", pyish, flags=re.I)
    pyish = re.sub(r"\bnull\b", "None", pyish, flags=re.I)
    try:
        obj = ast.literal_eval(pyish)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 7) Último recurso: extracción por claves
    out = {}

    # fit_percent
    m = re.search(r'fit_percent"\s*:\s*"?(\d{1,3})"?', s)
    if not m:
        m = re.search(r"fit_percent'\s*:\s*'?(\d{1,3})'?", s)
    if not m:
        m = re.search(r"(\d{1,3})\s*%", s)
    if m:
        out["fit_percent"] = max(0, min(100, int(m.group(1))))

    # analisis / analysis
    m = re.search(r'"analisis"\s*:\s*"(.*?)"', s, flags=re.S)
    if not m:
        m = re.search(r"'analisis'\s*:\s*'(.*?)'", s, flags=re.S)
    if not m:
        m = re.search(r'"analysis"\s*:\s*"(.*?)"', s, flags=re.S)
    if not m:
        m = re.search(r"'analysis'\s*:\s*'(.*?)'", s, flags=re.S)
    if m:
        out["analisis"] = m.group(1).strip()
    else:
        out["analisis"] = raw[:300]

    # consejos
    m = re.search(r'"consejos"\s*:\s*\[(.*?)\]', s, flags=re.S)
    if not m:
        m = re.search(r"'consejos'\s*:\s*\[(.*?)\]", s, flags=re.S)
    tips = []
    if m:
        tips = re.findall(r'"(.*?)"', m.group(1), flags=re.S)
        if not tips:
            tips = re.findall(r"'(.*?)'", m.group(1), flags=re.S)
    out["consejos"] = [t.strip() for t in tips if t.strip()]

    return out

def jd_fit_with_llm(cv_plain: str, jd_text: str) -> dict:
    """
    Pide a la LLM (Ollama) un % de encaje directo entre CV y oferta.
    Devuelve: {"fit_percent": int[0..100], "analisis": str, "consejos": list[str]}
    """
    system_prompt = (
        "Eres un reclutador técnico conciso. "
        "Lee el CV y la oferta y estima el encaje. "
        "Devuelve SOLO JSON válido sin explicaciones externas."
    )
    user_prompt = f"""
CV (texto):
\"\"\"{(cv_plain or '').strip()[:12000]}\"\"\"

OFERTA (texto):
\"\"\"{(jd_text or '').strip()[:12000]}\"\"\"

Tarea:
1) Estima el porcentaje de encaje (0–100) del CV para esta oferta.
2) Explica en 2-3 frases el porqué (analisis).
3) Da entre 3 y 6 consejos accionables para mejorar el ajuste.

Responde SOLO JSON con este esquema exacto:
{{
  "fit_percent": 0,
  "analisis": "texto breve",
  "consejos": ["...", "...", "..."]
}}
""".strip()

    prompt = f"{system_prompt}\n\n{user_prompt}".strip()

    # Intento 1: función global query_ollama(...)
    raw = ""
    try:
        qf = globals().get("query_ollama") or locals().get("query_ollama")
        if callable(qf):
            raw = (qf(prompt, expect_json=False) or "").strip()
        else:
            raise NameError("query_ollama no disponible")
    except Exception:
        # Intento 2: API HTTP de Ollama
        try:
            import requests
            model = os.environ.get("OLLAMA_MODEL", "mistral")
            url = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
            payload = {"model": model, "prompt": prompt, "stream": False}
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            raw = (data.get("response") or "").strip()
        except Exception as e:
            return {"fit_percent": 0, "analisis": f"Error al consultar Ollama: {e}", "consejos": []}

    obj = _parse_llm_json(raw)
    try:
        fit = int(obj.get("fit_percent", 0))
    except Exception:
        fit = 0
    fit = max(0, min(100, fit))
    analisis = str(obj.get("analisis") or "").strip() or "Sin análisis."
    consejos = obj.get("consejos") or []
    consejos = [str(c).strip() for c in consejos if str(c).strip()][:6]
    return {"fit_percent": fit, "analisis": analisis, "consejos": consejos}


def create_roles_prompt(cv_text: str, skills_out: list[dict], n: int = 5) -> str:
    """Pide al LLM sólo roles + porcentajes en JSON estricto."""
    skills_flat = ", ".join(
        sorted({(s.get("name") or "").lower() for s in (skills_out or [])})[:40]
    )
    return (
        "Eres un selector ATS. A partir del CV y de la lista de skills, "
        "devuelve SOLO JSON con estas claves:\n"
        f"- top_roles: array con hasta {n} títulos de rol (inglés) más adecuados\n"
        "- roles_pct: objeto {rol:int} con porcentaje 0..100 para esos mismos roles\n"
        "No añadas texto fuera del JSON. No inventes skills.\n\n"
        f"CV:\n\"\"\"{(cv_text or '')[:8000]}\"\"\"\n\n"
        f"SKILLS: [{skills_flat}]\n\n"
        "Ejemplo de respuesta:\n"
        "{ \"top_roles\":[\"DevOps Engineer\",\"Automation Engineer\",\"SRE\"], "
        "\"roles_pct\": {\"DevOps Engineer\":86, \"Automation Engineer\":78, \"SRE\":65} }"
    )


def infer_roles_with_llm(cv_text: str, skills_out: list[dict], n: int = 5) -> tuple[list[dict], dict[str, int]]:
    """Obtiene roles del LLM; si no hay %, los calcula con rank_roles_deterministic."""
    raw = query_ollama(create_roles_prompt(cv_text, skills_out, n=n), expect_json=True)
    try:
        jj = json.loads(clean_json_response(raw)) if raw else {}
    except Exception:
        jj = {}

    roles_llm: list[dict] = []
    roles_pct: dict[str, int] = {}

    # top_roles (strings)
    if isinstance(jj.get("top_roles"), list):
        for r in jj["top_roles"][:n]:
            if isinstance(r, str) and r.strip():
                roles_llm.append({"role": r.strip(), "fit_percent": None})

    # roles_pct (dict)
    if isinstance(jj.get("roles_pct"), dict):
        for k, v in jj["roles_pct"].items():
            try:
                roles_pct[str(k).strip()] = int(v)
            except Exception:
                pass

    # Si hay roles pero no % → calcula % determinista
    if roles_llm and not roles_pct:
        ranked = rank_roles_deterministic(
            (cv_text or ""), skills_out or [], [r["role"] for r in roles_llm]
        )
        # ranked ya viene con fit_percent
        roles_llm = ranked or roles_llm
        for r in roles_llm:
            if isinstance(r.get("fit_percent"), (int, float)):
                roles_pct[r["role"]] = int(r["fit_percent"])

    return roles_llm, roles_pct

def create_roles_prompt(cv_text: str, skills_out: list[dict], n: int = 5) -> str:
    skills_flat = ", ".join(
        sorted({(s.get("name") or "").lower() for s in (skills_out or [])})[:40]
    )
    return (
        "Eres un selector ATS. A partir del CV y de la lista de skills, "
        f"devuelve SOLO JSON con estas claves (máximo {n} roles):\n"
        "- top_roles: array de títulos de rol más adecuados\n"
        "- roles_pct: objeto {rol:int} con porcentaje 0..100 para esos roles\n"
        "No añadas texto fuera del JSON.\n\n"
        f"CV:\n\"\"\"{(cv_text or '')[:8000]}\"\"\"\n\n"
        f"SKILLS: [{skills_flat}]\n\n"
        "Ejemplo:\n"
        "{ \"top_roles\":[\"DevOps Engineer\",\"Automation Engineer\",\"SRE\"],"
        "  \"roles_pct\": {\"DevOps Engineer\":86, \"Automation Engineer\":78, \"SRE\":65} }"
    )


def infer_roles_with_llm(cv_text: str, skills_out: list[dict], n: int = 5) -> tuple[list[dict], dict[str, int]]:
    raw = query_ollama(create_roles_prompt(cv_text, skills_out, n=n), expect_json=True)
    try:
        jj = json.loads(clean_json_response(raw)) if raw else {}
    except Exception:
        jj = {}

    roles_llm: list[dict] = []
    roles_pct: dict[str, int] = {}

    if isinstance(jj.get("top_roles"), list):
        for r in jj["top_roles"][:n]:
            if isinstance(r, str) and r.strip():
                roles_llm.append({"role": r.strip(), "fit_percent": None})

    if isinstance(jj.get("roles_pct"), dict):
        for k, v in jj["roles_pct"].items():
            try:
                roles_pct[str(k).strip()] = int(v)
            except Exception:
                pass

    # Si hay roles pero sin %, calculamos % con ranking determinista
    if roles_llm and not roles_pct:
        ranked = rank_roles_deterministic(
            (cv_text or ""), skills_out or [], [r["role"] for r in roles_llm]
        )
        roles_llm = ranked or roles_llm
        for r in roles_llm:
            if isinstance(r.get("fit_percent"), (int, float)):
                roles_pct[r["role"]] = int(r["fit_percent"])

    return roles_llm, roles_pct
def generate_cv_json_for_role(role: str, analysis: dict, cv_text: str) -> dict:
    prompt = build_cv_prompt(role, analysis, cv_text)
    resp = query_ollama(prompt, expect_json=True)
    cleaned = clean_json_response(resp)
    try:
        data = json.loads(cleaned)
        # Saneado mínimo
        out = {
            "titulo": (data.get("titulo") or role).strip()[:120],
            "resumen": (data.get("resumen") or "").strip(),
            "skills_destacadas": [s.strip() for s in (data.get("skills_destacadas") or []) if s and isinstance(s, str)][:12],
            "experiencia_bullets": [b.strip() for b in (data.get("experiencia_bullets") or []) if b and isinstance(b, str)][:8],
            "proyectos_bullets": [b.strip() for b in (data.get("proyectos_bullets") or []) if b and isinstance(b, str)][:6],
            "formacion_bullets": [b.strip() for b in (data.get("formacion_bullets") or []) if b and isinstance(b, str)][:6],
            "extras_bullets": [b.strip() for b in (data.get("extras_bullets") or []) if b and isinstance(b, str)][:6],
        }
        return out
    except Exception:
        # Fallback minimal si el LLM patina
        return {
            "titulo": role,
            "resumen": "Profesional con experiencia en automatización, sistemas y desarrollo Python. CV generado automáticamente.",
            "skills_destacadas": [s.get("name","") for s in (analysis.get("skills") or [])][:10],
            "experiencia_bullets": [],
            "proyectos_bullets": [],
            "formacion_bullets": [],
            "extras_bullets": []
        }
    
def _is_empty_profile(data: dict) -> bool:
    if not data or not isinstance(data, dict):
        return True
    # si no hay experiencia, educación ni skills => vacío a efectos de presentación
    return not any([
        data.get("experiencia"),
        data.get("educacion"),
        data.get("skills"),
    ])

# Map de slugs alternativos -> fichero base
_FALLBACK_ROLE_MAP = {
    # backend
    "backend": "backend",
    "backend-developer": "backend",
    "software-engineer-backend": "backend",
    "python-backend": "backend",
    "java-backend": "backend",
    # sysadmin
    "sysadmin": "sysadmin",
    "systems-administrator": "sysadmin",
    "system-administrator": "sysadmin",
    "it-system-administrator": "sysadmin",
    # devops
    "devops": "devops",
    "devops-engineer": "devops",
    "site-reliability-engineer": "devops",
    "sre": "devops",
    # data
    "data": "data",
    "data-analyst": "data",
    "data-engineer": "data",
    "machine-learning-engineer": "data",
    # frontend
    "frontend": "frontend",
    "frontend-developer": "frontend",
    "web-developer-frontend": "frontend",
}
def _load_fallback_profile(app, role: str) -> tuple[dict, str]:
    """
    Carga un CV de ejemplo por rol. Devuelve (data, used_name)
    used_name = nombre del fichero lógico (para logs/UI).
    """
    slug = slugify_role(role)
    fname = _FALLBACK_ROLE_MAP.get(slug, slug)
    # ruta absoluta dentro del proyecto
    examples_dir = os.path.join(app.root_path, "cv_examples")
    candidate = os.path.join(examples_dir, f"{fname}.json")
    if not os.path.exists(candidate):
        candidate = os.path.join(examples_dir, "generic.json")
        used = "generic"
    else:
        used = fname
    try:
        with open(candidate, "r", encoding="utf-8") as f:
            data = json.load(f)
        # fuerza el título a ese rol (si procede)
        if role and isinstance(data, dict):
            base_title = data.get("titulo") or role
            # no toques si ya trae el del fichero
            data["titulo"] = base_title
        return data, used
    except Exception:
        # último recurso
        return {
            "nombre": "Nombre Apellidos",
            "titulo": role or "Profesional IT",
            "resumen_bullets": ["Este es un ejemplo de CV base."],
            "experiencia": [],
            "educacion": [],
            "skills": [],
            "idiomas": [],
            "email": "",
            "web": "",
            "ciudad": "",
            "proyectos": [],
        }, "generic"
def slugify_role(s: str) -> str:
    if not s:
        return "generic"
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().strip()
    s = s.replace("/", "-")
    s = re.sub(r"[^a-z0-9\s\-]", "", s)
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"-{2,}", "-", s)
    return s or "generic"

@app.before_request
def before():
    request.start_time = time.perf_counter()

@app.after_request
def after(response):
    if hasattr(request, "start_time"):
        log_metrics(request.path, request.method, response.status_code, request.start_time)
    return response
@app.route("/metrics")
def metrics():
    if not PROMETHEUS_ENABLED:
        return "Prometheus client not installed", 503
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}

@app.route('/')
def index():
    """Página principal con formulario de upload"""
    return render_template('index.html')

@app.template_filter("slug")
def jinja_slug(s):
    import unicodedata, re
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().strip()
    s = s.replace("/", "-")
    s = re.sub(r"[^a-z0-9\s\-]", "", s)
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"-{2,}", "-", s)
    return s 
@app.route('/upload', methods=['POST'])
def upload_file():
    """Maneja la subida y análisis del CV"""
    if 'cv_file' not in request.files:
        flash('No se seleccionó ningún archivo', 'error')
        return redirect(url_for('index'))

    file = request.files['cv_file']
    if file.filename == '':
        flash('No se seleccionó ningún archivo', 'error')
        return redirect(url_for('index'))

    if not (file and allowed_file(file.filename)):
        flash('Tipo de archivo no permitido. Usa PDF, DOCX o TXT.', 'error')
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        # 1) Extraer texto del CV
        cv_text = extract_cv_text(file_path)
        if not (cv_text or "").strip():
            flash('No se pudo extraer texto del CV. Verifica el formato.', 'error')
            return redirect(url_for('index'))

        # 2) Shortcircuit si no es perfil técnico
        res = shortcircuit_if_non_tech(cv_text, file.filename)
        if res is not None:
            return res

        # 3) Skills con LLM (JSON robusto)
        skills_prompt = create_skills_analysis_prompt(cv_text)
        skills_response = query_ollama(skills_prompt, expect_json=True)
        app.logger.debug("🔍 RESPUESTA OLLAMA SKILLS: %s", skills_response)

        try:
            cleaned = clean_json_response(skills_response)
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict) and 'skills' in parsed:
                analysis_data = build_analysis_from_llm(cv_text, parsed)
            else:
                raise ValueError("JSON sin 'skills'")
        except Exception:
            analysis_data = manual_skills_extraction(cv_text)

        # 4) Acciones prioritarias
        acciones_prioritarias = generar_prioridades(
            cv_text,
            skills_detectadas=[s.get("name", "") for s in analysis_data.get("skills", [])]
        )

        # 5) Feedback con LLM
        feedback_prompt = create_feedback_prompt(analysis_data, cv_text)
        feedback_response = query_ollama(feedback_prompt, expect_json=True)
        try:
            feedback_json = json.loads(clean_json_response(feedback_response))
        except Exception:
            feedback_json = {"fortalezas": [], "mejoras": [], "roles": [], "keywords_ats": []}

        # 6) UI viewmodel
        ui = build_ui_viewmodel(analysis_data)

        # 7) Keywords ATS + skills detectadas (para chips)
        keywords_ats = [
            (k or "").strip().lower()
            for k in (feedback_json.get("keywords_ats") or [])
            if k and isinstance(k, str)
        ][:20]

        skills_detectadas = sorted({
            (s.get("name") or "").strip()
            for s in analysis_data.get("skills", [])
            if (s.get("name") or "").strip()
        })

        # 8) Session
        session['skills_cv'] = [s.lower() for s in skills_detectadas]
        session['chat_ctx'] = {
            "especializacion": analysis_data.get("especializacion_principal"),
            "anios_total": analysis_data.get("años_experiencia_total"),
            "skills": [(s.get("name") or "").lower() for s in analysis_data.get("skills", [])],
            "fortalezas": (feedback_json or {}).get("fortalezas", []),
            "mejoras": (feedback_json or {}).get("mejoras", []),
            "keywords": (feedback_json or {}).get("keywords_ats", []),
            "cv_text": (cv_text or "")[:8000],
        }
        session['chat_history'] = []

        payload = {"cv_text": cv_text, "analysis": analysis_data}
        job_id = save_job_payload(payload)
        session['job_id'] = job_id

        # 9) ROLES (LLM → determinista si falla)  -------------------
        cv_text_for_roles = (analysis_data.get("cv_text_raw") or cv_text or "")
        roles, roles_pct = infer_roles_with_llm(cv_text_for_roles, analysis_data.get("skills", []), n=5)

        # si feedback ya traía roles, usarlos como candidatos
        if not roles:
            fb_roles = feedback_json.get("roles") or []
            if fb_roles:
                if isinstance(fb_roles[0], str):
                    roles = rank_roles_deterministic(cv_text_for_roles, analysis_data.get("skills", []), fb_roles)
                else:
                    roles = fb_roles
                if isinstance(roles, list):
                    for r in roles:
                        if isinstance(r, dict) and isinstance(r.get("fit_percent"), (int, float)):
                            roles_pct[(r.get("role") or r.get("name"))] = int(r["fit_percent"])

        # determinista puro si seguimos sin nada (sin defaults predefinidos)
        if not roles:
            spec = determine_specialization_enhanced(analysis_data.get("skills", []), cv_text_for_roles)
            focus = generate_job_focus(
                analysis_data.get("skills", []),
                spec,
                cv_text=cv_text_for_roles,
                total_years=analysis_data.get("años_experiencia_total"),
            )
            candidates = [c for c in (focus.get("buscar") or []) if isinstance(c, str) and c.strip()]
            ranked = rank_roles_deterministic(cv_text_for_roles, analysis_data.get("skills", []), candidates) if candidates else []
            roles = ranked or [{"role": r, "fit_percent": None} for r in candidates[:5]]
            if ranked:
                roles_pct = {r["role"]: int(r["fit_percent"]) for r in ranked if isinstance(r.get("fit_percent"), (int, float))}

        session['roles_recomendados'] = roles

        # 10) Exponer a la UI (barras + badges)
        ui.setdefault("headline", {})
        top_roles = [
            (r.get("role") or r.get("name") or "").strip()
            for r in roles if isinstance(r, dict) and (r.get("role") or r.get("name"))
        ]
        if top_roles:
            ui["headline"]["top_roles"] = top_roles[:5]
        if roles_pct:
            ui["headline"]["roles_pct"] = roles_pct

        # 11) Render
        return render_template(
            'results.html',
            roles_recomendados=roles,
            analysis=analysis_data,
            acciones_prioritarias=acciones_prioritarias,
            ui=ui,
            feedback=feedback_json,
            keywords_ats=keywords_ats,
            skills_detectadas=skills_detectadas,
            filename=file.filename
        )
    finally:
        # Limpieza: elimina archivo temporal siempre
        if os.path.exists(file_path):
            os.remove(file_path)




@app.route('/cv_html/<path:role>')
def cv_html(role):
    log_metrics(stage='cv_html:start')
    start_time = time.time()

    if 'job_id' not in session:
        # Sin sesión: usa perfil de fallback
        data, _used = _load_fallback_profile(app, role)
        example_used = True
    else:
        # Con sesión: carga payload
        payload = load_job_payload(session['job_id']) or {}
        data = _profile_from_payload_for_role(payload, role)
        example_used = False

        # Si el perfil está vacío, usar fallback
        if _is_empty_profile(data):
            data, _used = _load_fallback_profile(app, role)
            example_used = True

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"[METRICS] Tiempo total generación: {elapsed:.2f} s")

    log_metrics(stage='cv_html:before_render')
    return render_template("cv_template.html", data=data, role=role, example_used=example_used)


@app.route('/test-ollama')
def test_ollama():
    """Ruta para testear conexión con Ollama"""
    test_prompt = "Responde solo con: 'Ollama funcionando correctamente'"
    response = query_ollama(test_prompt)
    return jsonify({"status": "success", "response": response})


@app.route('/offer/compare', methods=['POST'])
def offer_compare():
    """Compara el CV en sesión con el texto de una oferta (jd_text) usando LLM."""
    start_time = time.time()

    # Sesión y payload
    if 'job_id' not in session:
        return jsonify({"error": "No hay CV activo en sesión. Sube un CV primero."}), 400

    payload = load_job_payload(session['job_id'])
    if not payload or not payload.get('analysis'):
        return jsonify({"error": "Sesión expirada o CV inválido. Vuelve a analizar tu CV."}), 400

    # Entrada
    data = request.get_json(silent=True) or {}
    jd_text = (data.get('jd_text') or "")
    jd_text = " ".join(jd_text.split())  # colapsar espacios
    if not jd_text:
        return jsonify({"error": "Pega el texto de la oferta (título + requisitos/funciones)."}), 400
    # límite defensivo (evita prompts gigantes al LLM)
    if len(jd_text) > 20000:
        jd_text = jd_text[:20000]

    # Texto plano del CV
    analysis = payload['analysis']
    cv_plain = _extract_cv_plaintext_from_analysis(analysis)
    if not cv_plain:
        return jsonify({"error": "No encuentro texto del CV en sesión. Re-analiza el CV."}), 400

    # Llamada LLM + robustez
    try:
        llm = jd_fit_with_llm(cv_plain, jd_text) or {}
    except Exception as e:
        app.logger.exception("jd_fit_with_llm falló")
        return jsonify({"error": f"Fallo al analizar la oferta con LLM: {e}"}), 500

    # Normalización segura del resultado
    fit_raw = llm.get("fit_percent", 0)
    try:
        fit_percent = max(0, min(100, int(fit_raw)))
    except (TypeError, ValueError):
        fit_percent = 0

    analisis = llm.get("analisis")
    if isinstance(analisis, list):
        analisis = [str(x) for x in analisis]
    elif isinstance(analisis, str):
        analisis = [analisis]
    else:
        analisis = []

    consejos = llm.get("consejos")
    if not isinstance(consejos, list):
        consejos = [str(consejos)] if consejos else []

    ats_pass = fit_percent >= 60

    # Métricas
    dt = (time.time() - start_time) * 1000
    app.logger.info("JD(LLM) fit=%d%% | consejos=%d | dt=%.1fms", fit_percent, len(consejos), dt)
    log_metrics(stage='offer_compare:done')

    # Respuesta
    return jsonify({
        "titulo": "Oferta",
        "ats_pass": ats_pass,
        "fit_percent": fit_percent,
        "skills": {  # por ahora vacías; el front lo sabe ocultar
            "core_match": [],
            "core_faltan": [],
            "perifericas_match": [],
            "perifericas_faltan": []
        },
        "analisis": analisis,
        "consejos": consejos
    }), 200

@app.route("/cv_pdf/<role>")
def cv_pdf(role):
    """Genera y descarga el PDF del CV para un rol dado."""
    start_time = time.time()

    if 'job_id' not in session:
        return "No hay CV en sesión", 400

    payload = load_job_payload(session['job_id']) or {}
    if not payload:
        return "Sesión expirada o payload inexistente", 400

    try:
        # Plantilla base PDF (tamaño y reader de la plantilla)
        tpl_path = _resolve_template_path()
        page_size_pts, tpl_reader = _get_template_pagesize(tpl_path)

        # Construir datos para el rol (con fallback si está vacío)
        data = _profile_from_payload_for_role(payload, role)
        if _is_empty_profile(data):
            data, _ = _load_fallback_profile(app, role)

        # Render de overlay y merge final sobre la plantilla
        overlay_pdf_bytes = _render_cv_on_overlay(data, page_size_pts=page_size_pts)
        merged_pdf_bytes  = _merge_overlay_on_template(overlay_pdf_bytes, tpl_reader)

        # Nombre de archivo seguro
        try:
            fname_name = slugify_role(data.get('nombre', 'SinNombre'))
            fname_role = slugify_role(role or 'IT')
        except Exception:
            # Fallback mínimo si no existe slugify_role
            import re
            def _slug(s): 
                s = (s or "").lower().strip().replace("/", "-")
                s = re.sub(r"[^a-z0-9\-\s]", "", s)
                s = re.sub(r"\s+", "-", s)
                return re.sub(r"-{2,}", "-", s)
            fname_name = _slug(data.get('nombre', 'SinNombre'))
            fname_role = _slug(role or 'IT')

        filename = f"CV_{fname_name}_{fname_role}.pdf"

        bio = io.BytesIO(merged_pdf_bytes)
        bio.seek(0)
        resp = send_file(
            bio,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=filename,
            max_age=0,
            last_modified=None,
            etag=False
        )
        # Cabeceras útiles
        resp.headers["Cache-Control"] = "no-store"
        resp.headers["X-Role"] = role
        resp.headers["X-Gen-Time-ms"] = f"{(time.time() - start_time)*1000:.1f}"
        return resp

    except Exception as e:
        app.logger.exception("Error generando PDF para role=%s", role)
        return (f"Error generando el PDF: {e}", 500)


@app.route("/cv_preview/<role>")
def cv_preview(role):
    """Preview HTML ligera del CV para un rol (sin PDF)."""
    import time, html
    start = time.time()
    log_metrics(stage="cv_preview:start")

    if 'job_id' not in session:
        return "No hay CV en sesión", 400

    payload = load_job_payload(session['job_id']) or {}
    if not payload:
        return "Sesión expirada o payload inexistente", 400

    # Construir datos (con fallback si el perfil está vacío)
    data = _profile_from_payload_for_role(payload, role)
    if _is_empty_profile(data):
        data, _ = _load_fallback_profile(app, role)

    # Escapes defensivos (la plantilla Jinja ya lo haría, pero aquí devolvemos HTML directo)
    def esc(x: object) -> str:
        return html.escape("" if x is None else str(x), quote=True)

    nombre = esc(data.get('nombre', ''))
    titulo = esc(data.get('titulo', ''))
    resumen_items = "".join(f"<li>{esc(b)}</li>" for b in (data.get('resumen_bullets') or []))

    # Experiencia
    exp_html = []
    for e in (data.get('experiencia') or []):
        empresa = esc(e.get('empresa', ''))
        pais    = esc(e.get('pais', ''))
        puesto  = esc(e.get('puesto', ''))
        fecha   = esc(e.get('fecha', ''))
        logros  = "".join(f"<li>{esc(l)}</li>" for l in (e.get('logros') or []))
        exp_html.append(
            f"<div class='mb-2'>"
            f"<div class='fw-semibold'>{empresa} — {pais} — {puesto}</div>"
            f"<div class='text-muted small'>{fecha}</div>"
            f"<ul>{logros}</ul>"
            f"</div>"
        )
    experiencia = "".join(exp_html)

    # Educación
    edu_html = "".join(
        f"<li>{esc(ed.get('titulo',''))} | {esc(ed.get('centro',''))} | "
        f"{esc(ed.get('fecha',''))} {esc(ed.get('estado',''))}</li>"
        for ed in (data.get('educacion') or [])
    )

    # Skills / Idiomas / Contacto
    skills_html = "".join(f"<span class='badge bg-secondary me-1 mb-1'>{esc(s)}</span>"
                          for s in (data.get('skills') or []))
    idiomas_html = esc(", ".join(data.get('idiomas') or []))
    contacto_html = " · ".join([esc(data.get('email','')), esc(data.get('web','')), esc(data.get('ciudad',''))])

    # Proyectos (opcional)
    proyectos_html = ""
    proys = data.get('proyectos') or []
    if proys:
        items = []
        for p in proys:
            if isinstance(p, dict):
                items.append(f"{esc(p.get('titulo',''))} — {esc(p.get('desc',''))}")
            else:
                items.append(esc(p))
        proyectos_html = "<h5 class='mt-3'>Proyectos</h5><ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"

    html_out = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Preview {esc(role)}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="p-3">
  <div class="container">
    <h3 class="mb-0">{nombre}</h3>
    <div class="text-muted mb-3">{titulo}</div>

    <h5>Resumen</h5>
    <ul>{resumen_items}</ul>

    <h5>Experiencia</h5>
    {experiencia}

    <h5>Educación</h5>
    <ul>{edu_html}</ul>

    <h5>Habilidades</h5>
    <div class="mb-2">{skills_html}</div>

    <h5>Idiomas</h5>
    <div>{idiomas_html}</div>

    <h5 class="mt-3">Contacto</h5>
    <div>{contacto_html}</div>

    {proyectos_html}
  </div>
</body>
</html>"""

    dt_ms = (time.time() - start) * 1000
    app.logger.info("cv_preview role=%s dt=%.1fms", role, dt_ms)
    log_metrics(stage='cv_preview:done')

    # Devuelve HTML explícitamente tipado
    return app.response_class(html_out, mimetype="text/html; charset=utf-8")


if __name__ == '__main__':
    print("🚀 Iniciando CVMatcher...")
    print("📊 Asegúrate de tener Ollama ejecutándose: ollama serve")
    print("🤖 Modelo requerido: ollama pull mistral")
    app.run(debug=True, port=5000)

