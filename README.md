# CVMatcher Local

Analiza tu CV frente a ofertas reales usando **IA local**, sin depender de OpenAI ni enviar tus datos fuera.  
Funciona 100% en local con **Ollama** y modelos ligeros como Mistral o LLaMA.  

---

## 🚀 Características

- Extracción de skills y experiencia desde tu CV (PDF/DOCX).
- Comparación contra descripciones de ofertas de empleo.
- Detección automática de dominio técnico (**tech** vs **non-tech**).
- Guardrails:
  - Si el CV no aporta skills → **rechazo directo** con mensaje sarcástico.
  - No se generan CVs falsos si no hay evidencia técnica.
- Generación de informes en **PDF/HTML** con el detalle de encaje.
- Ejecución **local** y privada (sin nube).

---

## 📂 Estructura del repo

cvmetrics.py → núcleo de la app (análisis y métricas)
skills_catalog.py → catálogo de skills y alias normalizados
requirements.txt → dependencias
cv_examples/ → CVs de ejemplo para probar
templates/ → plantillas para reportes


---

## ⚙️ Requisitos

- Python 3.10+
- [Ollama](https://ollama.ai) instalado y corriendo localmente
- Paquetes Python:

  pip install -r requirements.txt
▶️ Uso
Clonar el repo:


git clone https://github.com/oscaar90/cvmatcher-local.git
cd cvmatcher-local
Ejecutar el análisis sobre un CV:

Ver resultados en consola o exportar a PDF/HTML.

🧩 Roadmap
 V2: Guardrail para CVs vacíos (0 skills).

 HARSH_MODE: mensajes directos cuando el CV es genérico.

 Añadir soporte para más idiomas (EN/FR).

 Interfaz web simple en Flask.

📜 Licencia
MIT — libre para usar, modificar y compartir.
