from __future__ import annotations

import json
import uuid
from io import BytesIO

from flask import Blueprint, abort, jsonify, redirect, render_template, request, send_file, url_for

from imagemessrs.core.exif_utils import extract_camera_info
from imagemessrs.core.io_utils import load_image, resize_max_edge, save_image
from imagemessrs.core.registry import get_effect, list_effects
from imagemessrs.effects.base import coerce_params
from imagemessrs.effects.color.source_cameras import match_source_camera_from_exif
from imagemessrs.video.param_sweep import apply_param_sweep

from ..effect_serialization import serialize_effects
from ..jobs import JOB_TRACKER
from ..mask_utils import decode_mask
from ..store import IMAGE_STORE
from ..video_store import OUTPUTS_DIR

images_bp = Blueprint("images", __name__, url_prefix="/images")

PREVIEW_MAX_EDGE = 800
THUMB_MAX_EDGE = 160
MAX_ANIMATE_DURATION = 6.0
MAX_ANIMATE_FPS = 24.0


def _effects_json() -> str:
    return json.dumps(serialize_effects(list_effects()))


@images_bp.route("/", methods=["GET"])
def index():
    return render_template("image_upload.html")


@images_bp.route("/upload", methods=["POST"])
def upload():
    file_a = request.files.get("image_a")
    if not file_a or file_a.filename == "":
        abort(400, description="image_a is required")

    raw_a = file_a.read()
    try:
        image_a = load_image(raw_a)
    except Exception:
        abort(400, description="Could not read image_a - is it a valid image file?")

    make, model = extract_camera_info(raw_a)
    suggested_source_camera = match_source_camera_from_exif(make, model)
    session_id = IMAGE_STORE.create(image_a, suggested_source_camera=suggested_source_camera)

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
        return render_template(
            "session_expired.html",
            media_label="image",
            upload_url=url_for("images.index"),
        ), 404
    return render_template(
        "image_editor.html",
        session_id=session_id,
        has_image_b=session.original_b is not None,
        effects_json=_effects_json(),
        suggested_source_camera=session.suggested_source_camera or "",
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


@images_bp.route("/<session_id>/swap", methods=["POST"])
def swap_images(session_id: str):
    if not IMAGE_STORE.swap(session_id):
        abort(400, description="No second photo to swap in")
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

    source_a, source_b = session.original, session.original_b
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


@images_bp.route("/<session_id>/apply", methods=["POST"])
def apply_effect(session_id: str):
    """Bakes the current effect (at full resolution) into photo A, so a
    further effect can be cascaded on top of the result."""
    result = _run_effect(session_id, full_res=True)
    IMAGE_STORE.set_image_a(session_id, result)
    height, width = result.shape[:2]
    return jsonify({"ok": True, "width": int(width), "height": int(height)})


@images_bp.route("/<session_id>/animate", methods=["POST"])
def animate(session_id: str):
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

    sweep_param = request.form.get("sweep_param")
    param_spec = next((p for p in effect.params if p.name == sweep_param), None)
    if param_spec is None or param_spec.kind not in ("int", "float") or param_spec.min is None or param_spec.max is None:
        abort(400, description=f"{sweep_param!r} is not an animatable parameter of {effect_name!r}")

    seed_param = request.form.get("seed_param") or None
    if seed_param is not None:
        seed_spec = next((p for p in effect.params if p.name == seed_param), None)
        if seed_spec is None or seed_spec.kind != "int" or seed_param == sweep_param:
            abort(400, description=f"{seed_param!r} is not a valid seed parameter to vary")

    try:
        sweep_start = float(request.form.get("sweep_start", param_spec.min))
        sweep_end = float(request.form.get("sweep_end", param_spec.max))
        duration = float(request.form.get("duration", 3.0))
        fps = float(request.form.get("fps", 15.0))
    except (TypeError, ValueError):
        abort(400, description="sweep_start, sweep_end, duration, and fps must be numbers")

    duration = max(0.5, min(MAX_ANIMATE_DURATION, duration))
    fps = max(1.0, min(MAX_ANIMATE_FPS, fps))
    ping_pong = request.form.get("loop_style") == "ping_pong"
    full_res = request.form.get("full_res") in ("true", "on", "1")

    excluded = {"effect", "sweep_param", "sweep_start", "sweep_end", "duration", "fps", "loop_style", "seed_param", "full_res", "mask"}
    raw_params = {k: v for k, v in request.form.items() if k not in excluded}
    base_params = coerce_params(effect.params, raw_params)

    source_a, source_b = session.original, session.original_b
    image_a = source_a if full_res else resize_max_edge(source_a, PREVIEW_MAX_EDGE)

    mask_param = next((p.name for p in effect.params if p.kind == "mask"), None)
    if mask_param:
        base_params[mask_param] = decode_mask(request.form.get("mask"), image_a.shape[:2])

    image_b = None
    if effect.multi_image:
        if source_b is None:
            abort(400, description="This effect needs a second image - upload one first")
        image_b = source_b if full_res else resize_max_edge(source_b, PREVIEW_MAX_EDGE)

    output_path = OUTPUTS_DIR / f"{uuid.uuid4().hex}.mp4"

    def job_fn(on_progress):
        apply_param_sweep(
            image_a,
            str(output_path),
            effect_name,
            base_params,
            sweep_param,
            sweep_start,
            sweep_end,
            duration,
            fps,
            ping_pong=ping_pong,
            seed_param=seed_param,
            image_b=image_b,
            on_progress=on_progress,
        )
        return output_path

    job_id = JOB_TRACKER.start(job_fn)
    return jsonify({"job_id": job_id})


@images_bp.route("/<session_id>/animate/jobs/<job_id>/status", methods=["GET"])
def animate_job_status(session_id: str, job_id: str):
    job = JOB_TRACKER.get(job_id)
    if job is None:
        abort(404)
    return jsonify({"status": job.status, "progress": job.progress, "error": job.error})


@images_bp.route("/<session_id>/animate/jobs/<job_id>/result", methods=["GET"])
def animate_job_result(session_id: str, job_id: str):
    job = JOB_TRACKER.get(job_id)
    if job is None or job.status != "done" or job.result_path is None:
        abort(404)
    return send_file(str(job.result_path), mimetype="video/mp4")
