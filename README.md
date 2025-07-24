# CVMatcher Local

Este proyecto analiza tu CV frente a ofertas de empleo usando IA local, sin enviar datos fuera ni depender de OpenAI. Ideal para validar el encaje antes de aplicar.

---

## 🚀 ¿Qué hace?

Este script:

* Extrae el texto de tu CV (.pdf) y de una oferta (.txt o URL).
* Evalúa si tu CV pasaría un filtro ATS.
* Calcula un porcentaje real de ajuste con la oferta.
* Si superas el 60 %, te da:

  * Análisis del encaje.
  * Puntos fuertes y débiles.
  * 5 consejos concretos para mejorar tu CV.

Todo con un tono claro, directo, como si te hablara un reclutador técnico sin bullshit.

---

## 💡 Ejemplo de uso

```bash
python3 analizar_cv.py --cv ./ejemplos/CV.pdf --oferta ./ejemplos/oferta_cloud.txt
```

Puedes pasarle:

* Tu CV en PDF (`--cv`)
* Una oferta de empleo en texto plano (`--oferta`)

---

## 🔹 Formato de ejemplo para `oferta.txt`

Evita pegar toda la web con banners, cookies o descripciones genéricas. Copia solo la parte relevante de la oferta:

```
Puesto: Arquitecto/a Cloud

Requisitos:
- Experiencia con AWS o Azure.
- Conocimientos en automatización: Terraform, Ansible.
- Familiaridad con ServiceNow, CI/CD, y GitOps.
- Valorable experiencia en seguridad y entornos SQL.

Funciones:
- Diseño y despliegue de infraestructuras cloud.
- Automatización de procesos.
- Mejora continua y observabilidad.
```

---

## 🧠 IA utilizada

* **Modelo:** [Mistral](https://ollama.com/library/mistral)
* **Motor:** [Ollama](https://ollama.com/) ejecutándose localmente
* Sin necesidad de tokens ni conexión a OpenAI
* Respuesta personalizada, sin plantillas genéricas

---

## 🔢 Ejemplo de salida positiva

```bash
python3 analizar_cv.py --cv ejemplos/CV.pdf --oferta ejemplos/oferta_cloud.txt
```

**Resultado**:

```
0. El CV pasó el filtro ATS.
1. Porcentaje de ajuste: 85%
2. Perfil encaja con la oferta.
3. Análisis: Experiencia en administración de sistemas, automatización, CI/CD, scripting, cloud (AWS, Azure), IaC (Terraform), y herramientas como Ansible o ServiceNow.
4. Puntos fuertes: Automatización, scripting, cloud, DevOps.
5. Puntos débiles: No se especifica seguridad en redes ni GCP.
6. Consejos:
   - Añadir experiencia en seguridad de redes.
   - Especificar proyectos con Ansible/ServiceNow.
   - Mencionar participación en comunidades técnicas.
   - Destacar experiencia cloud (aunque no sea GCP).
   - Incluir logros medibles.
```

---

## ❌ Ejemplo de salida cuando **no es una oferta técnica**

```bash
python3 analizar_cv.py --cv ejemplos/CV.pdf --oferta ejemplos/oferta_hosteleria.txt
```

**Resultado**:

```
ERROR: La oferta no parece estar relacionada con perfiles técnicos. No procede continuar con el análisis del CV técnico.
```

---

## 🔧 Requisitos técnicos

* Python 3.10+
* `pdfplumber`
* `requests`
* Ollama instalado localmente con el modelo `mistral`

---

🐛 Posibles errores en la primera ejecución

En algunos casos, la primera llamada al modelo Mistral vía Ollama puede devolver un resultado incorrecto o incompleto, especialmente si el sistema acaba de arrancar o está iniciando el modelo por primera vez.

¿Qué hacer si pasa?

Vuelve a ejecutar el mismo comando. A partir de la segunda ejecución, el resultado será correcto si todo está bien configurado.

---

## 📂 Estructura recomendada

```
.
├── analizar_cv.py
├── ejemplos/
│   ├── CV.pdf
│   ├── oferta_cloud.txt
│   └── oferta_hosteleria.txt
├── README.md
```

---

## 🚀 Roadmap

* [ ] Añadir soporte para URLs de ofertas
* [ ] Mejorar recomendaciones personalizadas
* [ ] Exportar informe a Markdown/PDF

---

🙌 Proyecto creado por [@oscar90](https://github.com/oscar90)

Si te resulta últil, ✨ dale una estrella, clónalo o mejóralo ✨
