import pdfplumber
import argparse
import subprocess
from pathlib import Path

PROMPT_PATH = Path("prompts/plantilla_prompt.txt")

def extraer_texto_cv(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

def cargar_oferta(txt_path):
    return Path(txt_path).read_text()

def cargar_plantilla_prompt():
    return PROMPT_PATH.read_text()

def construir_prompt(cv_texto, oferta_texto):
    plantilla = cargar_plantilla_prompt()
    return plantilla.replace("{{CV}}", cv_texto.strip()).replace("{{OFERTA}}", oferta_texto.strip())

def llamar_ollama(prompt):
    comando = ["ollama", "run", "mistral"]
    proceso = subprocess.Popen(
        comando, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    salida, error = proceso.communicate(prompt)
    return salida

def main():
    parser = argparse.ArgumentParser(description="Analiza un CV frente a una oferta usando IA local (Mistral via Ollama).")
    parser.add_argument("--cv", required=True, help="Ruta al archivo PDF del CV")
    parser.add_argument("--oferta", required=True, help="Ruta al archivo TXT con la oferta")
    args = parser.parse_args()

    print("[+] Extrayendo texto del CV...")
    texto_cv = extraer_texto_cv(args.cv)

    print("[+] Cargando oferta...")
    texto_oferta = cargar_oferta(args.oferta)

    print("[+] Construyendo prompt para IA...")
    prompt = construir_prompt(texto_cv, texto_oferta)

    print("[+] Llamando a Ollama (modelo: mistral)...")
    resultado = llamar_ollama(prompt)

    print("\n===== RESULTADO =====\n")
    print(resultado)

if __name__ == "__main__":
    main()
