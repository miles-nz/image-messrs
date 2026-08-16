from __future__ import annotations

import json
from io import BytesIO

from flask import Blueprint, abort, redirect, render_template, request, send_file, url_for

from imagemessrs.core.io_utils import load_image, resize_max_edge, save_image
from imagemessrs.core.registry import get_effect, list_effects
from imagemessrs.effects.base import coerce_params

from ..mask_utils import decode_mask
from ..store import IMAGE_STORE

images_bp = Blueprint("images", __name__, url_prefix="/images")

PREVIEW_MAX_EDGE = 800
THUMB_MAX_EDGE = 160


def _effects_json() -> str:
    effects = list_effects()
    data = [
        {
            "name": e.name,
            "label": e.label,
            "category": e.category,
            "multi_image": e.multi_image,
            "accepts_mask": e.accepts_mask,
            "description": e.description,
            "about": e.about,
            "params": [
                {
                    "name": p.name,
                    "kind": p.kind,
                    "default": p.default,
                    "label": p.label,
                    "description": p.description,
                    "min": p.min,
                    "max": p.max,
                    "step": p.step,
                    "choices": p.choices,
                }
                for p in e.params
            ],
        }
        for e in effects
    ]
    return json.dumps(data)


@images_bp.route("/", methods=["GET"])
def index():
    return render_template("image_upload.html")


@images_bp.route("/upload", methods=["POST"])
def upload():
    file_a = request.files.get("image_a")
    if not file_a or file_a.filename == "":
        abort(400, description="image_a is required")

    try:
        image_a = load_image(file_a.read())
    except Exception:
        abort(400, description="Could not read image_a - is it a valid image file?")

    session_id = IMAGE_STORE.create(image_a)

    file_b = request.files.get("image_b")
    if file_b and file_b.filename:
        try:
            image_b = load_image(file_b.read())
        except Exception:
            abort(400, description="Could not read image_b - is it a valid image file?")
        IMAGE_STORE.set_image_b(session_id, image_b)

    return redirect(url_for("images.edit", session_id=session_id))


@images_bp.route("/<session_id>/upload_second", methods=["POST"])
def upload_second(session_id: str):
    session = IMAGE_STORE.get(session_id)
    if session is None:
        abort(404)

    file_b = request.files.get("image_b")
    if not file_b or file_b.filename == "":
        abort(400, description="image_b is required")
    try:
        image_b = load_image(file_b.read())
    except Exception:
        abort(400, description="Could not read image_b - is it a valid image file?")

    IMAGE_STORE.set_image_b(session_id, image_b)
    return redirect(url_for("images.edit", session_id=session_id))


@images_bp.route("/<session_id>/edit", methods=["GET"])
def edit(session_id: str):
    session = IMAGE_STORE.get(session_id)
    if session is None:
        abort(404)
    return render_template(
        "image_editor.html",
        session_id=session_id,
        effects=list_effects(),
        has_image_b=session.original_b is not None,
        effects_json=_effects_json(),
    )


@images_bp.route("/<session_id>/original", methods=["GET"])
def original(session_id: str):
    session = IMAGE_STORE.get(session_id)
    if session is None:
        abort(404)
    preview_img = resize_max_edge(session.original, PREVIEW_MAX_EDGE)
    return send_file(BytesIO(save_image(preview_img, fmt="PNG")), mimetype="image/png")


@images_bp.route("/<session_id>/thumbnail/<slot>", methods=["GET"])
def thumbnail(session_id: str, slot: str):
    if slot not in ("a", "b"):
        abort(404)
    session = IMAGE_STORE.get(session_id)
    if session is None:
        abort(404)
    source = session.original if slot == "a" else session.original_b
    if source is None:
        abort(404)
    thumb = resize_max_edge(source, THUMB_MAX_EDGE)
    return send_file(BytesIO(save_image(thumb, fmt="PNG")), mimetype="image/png")


@images_bp.route("/<session_id>/replace/<slot>", methods=["POST"])
def replace_image(session_id: str, slot: str):
    if slot not in ("a", "b"):
        abort(404)
    session = IMAGE_STORE.get(session_id)
    if session is None:
        abort(404)

    file = request.files.get("image")
    if not file or file.filename == "":
        abort(400, description="image is required")
    try:
        image = load_image(file.read())
    except Exception:
        abort(400, description="Could not read image - is it a valid image file?")

    if slot == "a":
        IMAGE_STORE.set_image_a(session_id, image)
    else:
        IMAGE_STORE.set_image_b(session_id, image)

    return {"ok": True}


def _run_effect(session_id: str, full_res: bool):
    session = IMAGE_STORE.get(session_id)
    if session is None:
        abort(404)

    effect_name = request.form.get("effect")
    if not effect_name:
        abort(400, description="effect is required")
    try:
        effect = get_effect(effect_name)
    except KeyError:
        abort(404, description=f"unknown effect {effect_name!r}")

    raw_params = {k: v for k, v in request.form.items() if k != "effect"}
    params = coerce_params(effect.params, raw_params)

    swap_images = (
        effect.multi_image
        and session.original_b is not None
        and request.form.get("swap_images") in ("true", "on", "1")
    )
    source_a, source_b = session.original, session.original_b
    if swap_images:
        source_a, source_b = source_b, source_a

    image_a = source_a if full_res else resize_max_edge(source_a, PREVIEW_MAX_EDGE)

    mask_param = next((p.name for p in effect.params if p.kind == "mask"), None)
    if mask_param:
        params[mask_param] = decode_mask(request.form.get("mask"), image_a.shape[:2])

    if effect.multi_image:
        if source_b is None:
            abort(400, description="This effect needs a second image - upload one first")
        image_b = source_b if full_res else resize_max_edge(source_b, PREVIEW_MAX_EDGE)
        args = (image_a, image_b)
    else:
        args = (image_a,)

    try:
        result = effect.fn(*args, **params)
    except Exception as exc:  # user-controlled slider input reaching numerical code
        abort(400, description=f"Effect failed with current parameters: {exc}")

    return result


@images_bp.route("/<session_id>/preview", methods=["POST"])
def preview(session_id: str):
    result = _run_effect(session_id, full_res=False)
    return send_file(BytesIO(save_image(result, fmt="PNG")), mimetype="image/png")


@images_bp.route("/<session_id>/render", methods=["POST"])
def render(session_id: str):
    result = _run_effect(session_id, full_res=True)
    data = save_image(result, fmt="PNG")
    return send_file(
        BytesIO(data),
        mimetype="image/png",
        as_attachment=True,
        download_name="image-messrs-output.png",
    )
