from flask import Flask, render_template, request, jsonify
from deepface import DeepFace
import base64
import cv2
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────
# CONFIG BASE
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
FACES_DIR = BASE_DIR / "registered_faces"
FACES_DIR.mkdir(exist_ok=True)

app = Flask(
    __name__,
    template_folder=str(TEMPLATES_DIR)
)

MODEL_NAME = "SFace"
MATCH_THRESHOLD = 0.56  # ajustable si quieres más o menos sensibilidad

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# cache de embeddings en memoria para no recalcular todo cada vez
reference_embeddings: Dict[str, np.ndarray] = {}


# ─────────────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────────────
def safe_name(name: str) -> str:
    clean = "".join(c for c in name.strip() if c.isalnum() or c in (" ", "_", "-")).strip()
    return clean


def decode_image(data_url: str) -> np.ndarray:
    _, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("No se pudo decodificar la imagen.")
    return frame


def detect_largest_face(frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )
    if len(faces) == 0:
        return None
    return sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]


def crop_face(frame: np.ndarray) -> Optional[np.ndarray]:
    face = detect_largest_face(frame)
    if face is None:
        return None

    x, y, w, h = face
    margin = int(max(w, h) * 0.18)

    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(frame.shape[1], x + w + margin)
    y2 = min(frame.shape[0], y + h + margin)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def get_embedding(face_bgr: np.ndarray) -> np.ndarray:
    rep = DeepFace.represent(
        img_path=face_bgr,
        model_name=MODEL_NAME,
        detector_backend="skip",
        enforce_detection=False
    )
    return np.array(rep[0]["embedding"], dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def load_reference_embeddings() -> None:
    reference_embeddings.clear()

    for filename in os.listdir(FACES_DIR):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = FACES_DIR / filename
        image = cv2.imread(str(path))
        if image is None:
            print(f"[WARN] No se pudo leer: {filename}")
            continue

        face = crop_face(image)
        if face is None:
            print(f"[WARN] No se detectó rostro en: {filename}")
            continue

        try:
            embedding = get_embedding(face)
            name = os.path.splitext(filename)[0]
            reference_embeddings[name] = embedding
            print(f"[OK] Cargado: {name}")
        except Exception as e:
            print(f"[ERR] {filename}: {e}")


def list_registered_users() -> List[str]:
    return sorted(reference_embeddings.keys())


def delete_face_file(name: str) -> bool:
    removed = False
    for ext in (".jpg", ".jpeg", ".png"):
        path = FACES_DIR / f"{name}{ext}"
        if path.exists():
            path.unlink()
            removed = True
    return removed


def clear_face_folder() -> int:
    deleted = 0
    for filename in os.listdir(FACES_DIR):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            os.remove(FACES_DIR / filename)
            deleted += 1
    return deleted


# ─────────────────────────────────────────────────────────────
# RUTAS
# ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/register", methods=["POST"])
def register():
    try:
        payload = request.get_json(force=True)
        name = payload.get("name", "").strip()
        image = payload.get("image", "")

        if not name:
            return jsonify({"success": False, "message": "Ingresa un nombre."})

        if not image:
            return jsonify({"success": False, "message": "No se recibió ninguna imagen."})

        frame = decode_image(image)
        face = crop_face(frame)

        if face is None:
            return jsonify({
                "success": False,
                "message": "No se detectó ningún rostro. Intenta de nuevo."
            })

        clean = safe_name(name)
        if not clean:
            return jsonify({"success": False, "message": "El nombre no es válido."})

        filepath = FACES_DIR / f"{clean}.jpg"
        ok = cv2.imwrite(str(filepath), face)

        if not ok:
            return jsonify({"success": False, "message": "No se pudo guardar la imagen."})

        load_reference_embeddings()

        return jsonify({
            "success": True,
            "message": f"Rostro de '{name}' registrado correctamente."
        })

    except Exception as e:
        print(f"[ERROR] /register → {e}")
        return jsonify({
            "success": False,
            "message": "Error interno al registrar el rostro."
        })


@app.route("/login", methods=["POST"])
def login():
    try:
        payload = request.get_json(force=True)
        image = payload.get("image", "")

        if not image:
            return jsonify({
                "success": False,
                "recognized": False,
                "message": "No se recibió ninguna imagen."
            })

        if not reference_embeddings:
            return jsonify({
                "success": False,
                "recognized": False,
                "message": "No hay rostros registrados en la carpeta registered_faces."
            })

        frame = decode_image(image)
        face = crop_face(frame)

        if face is None:
            return jsonify({
                "success": False,
                "recognized": False,
                "message": "No se detectó ningún rostro."
            })

        probe_embedding = get_embedding(face)

        best_user = None
        best_score = -1.0

        for user, ref_embedding in reference_embeddings.items():
            score = cosine_similarity(probe_embedding, ref_embedding)
            if score > best_score:
                best_score = score
                best_user = user

        print("MEJOR SCORE:", best_score, "USUARIO:", best_user)

        if best_user is not None and best_score >= MATCH_THRESHOLD:
            return jsonify({
                "success": True,
                "recognized": True,
                "user": best_user,
                "score": round(best_score, 4),
                "message": (
                    f"Bienvenido {best_user}. "
                    "El sistema continúa en proceso de desarrollo."
                )
            })

        return jsonify({
            "success": False,
            "recognized": False,
            "message": (
                "Este rostro no está registrado en la base de datos. "
                "Por favor, regístralo."
            )
        })

    except Exception as e:
        print(f"[ERROR] /login → {e}")
        return jsonify({
            "success": False,
            "recognized": False,
            "message": "Error en reconocimiento facial."
        })


@app.route("/users")
def users():
    return jsonify({"users": list_registered_users()})


@app.route("/delete-user", methods=["POST"])
def delete_user():
    try:
        payload = request.get_json(force=True)
        name = safe_name(payload.get("name", ""))

        if not name:
            return jsonify({"success": False, "message": "Nombre inválido."})

        removed = delete_face_file(name)

        if not removed:
            return jsonify({"success": False, "message": "No se encontró ese rostro."})

        load_reference_embeddings()

        return jsonify({
            "success": True,
            "message": f"'{name}' fue eliminado correctamente."
        })

    except Exception as e:
        print(f"[ERROR] /delete-user → {e}")
        return jsonify({"success": False, "message": "No se pudo eliminar el rostro."})


@app.route("/clear-users", methods=["POST"])
def clear_users():
    try:
        deleted = clear_face_folder()
        load_reference_embeddings()

        return jsonify({
            "success": True,
            "message": f"Se eliminaron {deleted} rostros."
        })

    except Exception as e:
        print(f"[ERROR] /clear-users → {e}")
        return jsonify({"success": False, "message": "No se pudieron eliminar los rostros."})


if __name__ == "__main__":
    load_reference_embeddings()
    print("\nFaceID iniciado")
    print("Servidor: http://127.0.0.1:5000\n")
    app.run(debug=True)