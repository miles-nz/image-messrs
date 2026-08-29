from __future__ import annotations

import shutil
import tempfile
import uuid
from io import BytesIO
from pathlib import Path

from flask import Blueprint, abort, jsonify, redirect, render_template, request, send_file, url_for

from imagemessrs.core.io_utils import save_image
from imagemessrs.core.registry import list_effects
from imagemessrs.effects.base import coerce_params
from imagemessrs.video import datamosh, frame_effects, slowmo
from imagemessrs.video import ffmpeg_wrapper as ffmpeg
from imagemessrs.video.frame_io import extract_frames

from ..effect_serialization import serialize_effects
from ..jobs import JOB_TRACKER
from ..video_store import OUTPUTS_DIR, UPLOADS_DIR, VIDEO_STORE

# Effects eligible to run as a per-frame video technique: single-image only
# (no second image to sync per-frame) and shape-preserving (seam carving
# changes output dimensions, which write_frames can't handle mid-stream).
FRAME_EFFECT_CATEGORIES_EXCLUDED = {"seam_carve"}


def _frame_effects():
    return [
        e for e in list_effects() if not e.multi_image and e.category not in FRAME_EFFECT_CATEGORIES_EXCLUDED
    ]

video_bp = Blueprint("video", __name__, url_prefix="/video")

TECHNIQUES = {
    "per_frame": {
        "label": "Per-Frame Effects",
        "description": "Runs any of this app's still-image effects independently on every frame of the video - no motion or temporal awareness, just the same image transform stamped onto each frame.",
        "about": {
            "what": "Applies one of this app's single-image effects (glitch, color, distortion, etc.) to every frame of the video on its own, frame by frame, then stitches the results back into a video at the original frame rate.",
            "how_to_use": "Pick an effect from the second dropdown below, then dial in its parameters exactly as you would in the image editor. Since every frame gets the exact same settings, a still, unmoving effect (like a fixed color grade) stays rock steady, while a randomized one (like Byte Corruption) reshuffles its noise pattern fresh on every single frame unless its Seed is held fixed relative to frame content.",
            "used_for": "Bringing any of this app's image effects to video without needing a dedicated video-specific implementation for each one - the fastest way to try glitch, color, and distortion looks on moving footage.",
            "examples": "This is a generic bridge rather than a named technique of its own - the interesting part is whichever image effect you choose. Effects with randomness (grain, dust, byte corruption) tend to read as lively, flickering texture across frames since each frame's noise is independent; deterministic effects (lens distortion, chromatic aberration, color grades) instead read as a steady, consistent look applied uniformly throughout the clip.",
        },
        "needs_motion_clip": False,
        "frame_effect_bridge": True,
        "params": [],
    },
    "frame_blend": {
        "label": "Frame Blend / Motion Trails",
        "description": "Blends each frame with a fading window of the frames before it, ffmpeg-filter style - no bitstream tricks, safest and most predictable of the three.",
        "about": {
            "what": "Blends each frame with a fading window of the frames before it, so movement leaves a trailing smear behind it.",
            "how_to_use": "Raise Trail Length to reach further back in time for longer trails, and adjust Decay to control how quickly older frames fade - low decay gives a tight trail, high decay gives a heavy, dragging smear.",
            "used_for": "Adding motion trails or a “long exposure” smeared look to moving subjects without any risky bitstream manipulation.",
            "examples": "This is a digital echo of long-exposure photography and analog video feedback - techniques pioneered by early video artists like Nam June Paik and Steina and Woody Vasulka, who fed a camera's own output back into itself to create trailing, smeared imagery.",
        },
        "needs_motion_clip": False,
        "params": [
            {
                "name": "trail_length",
                "label": "Trail Length (frames)",
                "kind": "int",
                "default": 4,
                "min": 1,
                "max": 20,
                "step": 1,
                "description": "How many previous frames are blended into each output frame. 1 = no trail (output ≈ original). Higher values reach further back in time, giving longer, smokier trails - but each extra frame also costs more processing time.",
            },
            {
                "name": "decay",
                "label": "Decay",
                "kind": "float",
                "default": 0.6,
                "min": 0.0,
                "max": 0.95,
                "step": 0.05,
                "description": "How much weight older frames keep in the blend, per step back. Near 0 = older frames vanish almost immediately (subtle, tight trail). Near 0.95 = older frames stay nearly as strong as the current one (heavy, dragging smear). Only matters if Trail Length is above 1.",
            },
        ],
    },
    "change_speed": {
        "label": "Change Speed",
        "description": "Slows footage down or speeds it up by synthesizing/combining frames from the motion between them, instead of just retiming or dropping frames - smoother than a naive fps change, no audio.",
        "about": {
            "what": "Changes the clip's playback speed by generating or combining frames based on the motion between them, rather than just stretching or dropping the existing ones.",
            "how_to_use": "Pick Slow Down or Speed Up below, then set that direction's Speed and Method.",
            "used_for": "Turning ordinary footage into slow motion or fast motion without the stutter/strobing of a naive frame-rate change.",
            "examples": "Motion-compensated frame interpolation (slowing down) and motion-compensated frame blending (speeding up) are the same family of technique behind TV/monitor \"motion smoothing\" and speed-ramp tools in video editing software.",
        },
        "needs_motion_clip": False,
        "sub_techniques": {
            "slow_down": {
                "label": "Slow Down",
                "about": {
                    "what": "Slows playback down by generating new frames between the real ones, based on the motion between them, rather than just duplicating or stretching the existing frames over a longer duration.",
                    "how_to_use": "Set Speed to how much slower the clip should play (0.5 = half speed). Method controls how the in-between frames are made: Optical Flow estimates the actual motion and warps toward it (smoother, slower to render); Blend just cross-dissolves adjacent frames (fast, but ghosts on anything moving); Duplicate just holds the previous frame (fastest, but stutters - the naive 'just lower the fps' baseline the other two methods improve on). Audio is dropped, since the clip's length changes.",
                    "used_for": "Turning ordinary footage into slow motion without the stutter you'd get from just re-timing the original frame count.",
                    "examples": "Motion-compensated frame interpolation is the same family of technique behind TV/monitor \"motion smoothing\" and slow-mo modes in video editing software - estimating where pixels moved between frames and synthesizing the frames in between, rather than just repeating or blending existing ones.",
                },
                "params": [
                    {
                        "name": "speed_factor",
                        "label": "Speed",
                        "kind": "float",
                        "default": 0.5,
                        "min": 0.1,
                        "max": 0.9,
                        "step": 0.05,
                        "description": "Playback speed multiplier. 0.5 = half speed (2x slower). Lower values need more synthesized frames between each original pair, so they cost more processing time.",
                    },
                    {
                        "name": "method",
                        "label": "Method",
                        "kind": "choice",
                        "default": "optical_flow",
                        "choices": ["optical_flow", "blend", "duplicate"],
                        "description": "Optical Flow estimates real motion between frames and warps toward it - smoother, no ghosting, but slower and can warp oddly around occlusion or scene cuts. Blend just cross-dissolves adjacent frames - fast, but visibly ghosts anything in motion. Duplicate just holds the previous frame - fastest, but stutters (the naive 'just lower the fps' baseline).",
                    },
                ],
            },
            "speed_up": {
                "label": "Speed Up",
                "about": {
                    "what": "Speeds playback up by combining each stretch of skipped source frames into the one output frame that represents it, rather than simply dropping them and keeping only every Nth frame.",
                    "how_to_use": "Set Speed to how much faster the clip should play (2 = twice as fast). Method controls how each group of skipped frames is combined: Optical Flow warps them onto the kept frame's motion position before combining (smoothest, closest to real motion blur, slower to render); Blend just averages the group (fast, but flattens fast motion into a soft ghost); Drop just keeps the last frame of each group and throws the rest away (fastest, but strobes/judders - the naive 'just lower the fps' baseline the other two methods improve on). Audio is dropped, since the clip's length changes.",
                    "used_for": "Turning ordinary footage into fast motion / time-lapse-style playback without the strobing you'd get from simply dropping frames.",
                    "examples": "This is the same idea in reverse as Slow Down - instead of estimating motion to invent new in-between frames, it estimates motion to combine several real frames into one, closer to how a camera's own motion blur looks at a slower shutter speed than to a jarring skipped-frame time-lapse.",
                },
                "params": [
                    {
                        "name": "speed_factor",
                        "label": "Speed",
                        "kind": "float",
                        "default": 2.0,
                        "min": 1.5,
                        "max": 8.0,
                        "step": 0.5,
                        "description": "Playback speed multiplier. 2 = twice as fast. Rounds to a whole number of source frames combined per output frame, so actual output speed is only approximate.",
                    },
                    {
                        "name": "method",
                        "label": "Method",
                        "kind": "choice",
                        "default": "optical_flow",
                        "choices": ["optical_flow", "blend", "drop"],
                        "description": "Optical Flow warps each skipped frame onto the kept frame's motion position before combining - smoothest, closest to real motion blur, but slower and can warp oddly around occlusion or scene cuts. Blend just averages the group - fast, but flattens fast motion into a soft ghost. Drop just keeps the last frame of the group - fastest, but strobes/judders (the naive 'just lower the fps' baseline).",
                    },
                ],
            },
        },
    },
    "feedback_loop": {
        "label": "Feedback Loop (generation loss)",
        "description": "Re-encodes the video several times in a row, like repeatedly re-saving a JPEG or dubbing a VHS tape - each pass compounds the compression damage from the last.",
        "about": {
            "what": "Re-encodes the video several times back to back, so each pass compounds the compression damage left by the last - like repeatedly re-saving a JPEG or dubbing a VHS tape from a copy of a copy.",
            "how_to_use": "Set Iterations for how many passes to stack, and Quality (CRF) for how much damage each individual pass adds - CRF is inverted, so higher numbers mean worse quality per pass, and that damage compounds fast when combined with more iterations.",
            "used_for": "Deliberately inducing “generation loss” - the deep-fried, over-compressed look of media that's been copied and re-copied many times.",
            "examples": "Generation loss is a well-documented phenomenon from analog tape culture (VHS dubbing) and, in the internet era, from repeated re-uploads and re-compressions of the same video - an aesthetic that gave rise to the “deep-fried meme” style of deliberately over-processed images and video.",
        },
        "needs_motion_clip": False,
        "params": [
            {
                "name": "iterations",
                "label": "Iterations",
                "kind": "int",
                "default": 3,
                "min": 1,
                "max": 10,
                "step": 1,
                "description": "How many re-encode passes to run back to back. Each pass re-compresses the previous pass's output, so damage compounds - 1 is a single light re-encode, 10 stacks that damage ten times over and can get extreme fast.",
            },
            {
                "name": "quality",
                "label": "Quality (CRF, higher = worse)",
                "kind": "int",
                "default": 28,
                "min": 18,
                "max": 45,
                "step": 1,
                "description": "ffmpeg's CRF quality setting used on every pass - the opposite of most quality sliders: lower numbers mean higher quality (less damage per pass), higher numbers mean more compression artifacts per pass. Combined with Iterations, high values here degrade very quickly since the damage stacks each pass.",
            },
        ],
    },
    "iframe_smear": {
        "label": "Datamosh (I-Frame Smear)",
        "description": "EXPERIMENTAL. Splices this motion clip's motion data directly onto your base video's last frame at the bitstream level, so the motion clip's movement drags and smears the base image instead of replacing it - the classic 'datamoshing' look. Results depend heavily on clip content and aren't guaranteed to play back identically everywhere; check the result before relying on it.",
        "about": {
            "what": "Splices a second “motion” clip's motion data directly onto your base video's last frame at the bitstream level, so the motion clip's movement drags and smears the base image instead of replacing it.",
            "how_to_use": "Upload a motion clip, then adjust GOP Size to control how long the smear can drag before an internal keyframe resets it, and Quality Scale to trade off crispness against extra compression artifacts in the intermediate re-encode. Results vary a lot with clip content, so check playback before relying on it.",
            "used_for": "Producing “datamoshing” - the classic glitch-video look of one scene's motion dragging and smearing pixels from a completely different scene.",
            "examples": "Datamoshing became widely known through 2000s/2010s music videos and viral videos that exploited P-frame/I-frame corruption in compressed video - Kanye West's “Welcome to Heartbreak” (2008) is one of the most frequently cited early examples that brought the technique into mainstream visibility.",
        },
        "needs_motion_clip": True,
        "params": [
            {
                "name": "gop_size",
                "label": "GOP Size",
                "kind": "int",
                "default": 300,
                "min": 30,
                "max": 1000,
                "step": 10,
                "description": "How far apart the internal keyframes are placed in the intermediate re-encode, in frames. Larger values mean fewer keyframes to anchor the image, so the borrowed motion has more room to drag the smear out before anything resets it - smaller values reset (and clean up) more often.",
            },
            {
                "name": "qscale",
                "label": "Quality Scale",
                "kind": "int",
                "default": 4,
                "min": 1,
                "max": 20,
                "step": 1,
                "description": "Compression quality of the intermediate re-encode used to prepare both clips for splicing - lower is higher quality (crisper smear, cleaner motion data), higher is blockier (adds its own extra compression artifacts on top of the smear).",
            },
        ],
    },
}


def _coerce(value, kind):
    if kind == "int":
        return int(float(value))
    if kind == "float":
        return float(value)
    return value


def _has_file(file_storage) -> bool:
    return bool(file_storage and file_storage.filename)


def _suffix_for(file_storage) -> str:
    return Path(file_storage.filename).suffix or ".mp4"


def _probe_meta(path: Path) -> dict:
    try:
        info = ffmpeg.probe(path)
    except ffmpeg.FFmpegError:
        return {}
    video_stream = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), {})
    return {
        "duration": info.get("format", {}).get("duration"),
        "width": video_stream.get("width"),
        "height": video_stream.get("height"),
        "codec": video_stream.get("codec_name"),
    }


def _maybe_trim(path: Path, trim: bool, tmp_dir: Path, name: str) -> Path:
    if not trim:
        return path
    trimmed_path = tmp_dir / name
    ffmpeg.run(["-i", str(path), "-t", "3", "-c", "copy", str(trimmed_path)])
    return trimmed_path


@video_bp.route("/", methods=["GET"])
def index():
    return render_template("video_upload.html")


@video_bp.route("/upload", methods=["POST"])
def upload():
    video_file = request.files.get("video")
    if not _has_file(video_file):
        abort(400, description="video is required")
    session_id = VIDEO_STORE.create(video_file.read(), suffix=_suffix_for(video_file))

    motion_file = request.files.get("motion_clip")
    if _has_file(motion_file):
        VIDEO_STORE.set_motion_clip(session_id, motion_file.read(), suffix=_suffix_for(motion_file))

    return redirect(url_for("video.edit", session_id=session_id))


@video_bp.route("/<session_id>/upload_motion", methods=["POST"])
def upload_motion(session_id: str):
    session = VIDEO_STORE.get(session_id)
    if session is None:
        abort(404)
    motion_file = request.files.get("motion_clip")
    if not _has_file(motion_file):
        abort(400, description="motion_clip is required")
    VIDEO_STORE.set_motion_clip(session_id, motion_file.read(), suffix=_suffix_for(motion_file))
    return redirect(url_for("video.edit", session_id=session_id))


@video_bp.route("/<session_id>/edit", methods=["GET"])
def edit(session_id: str):
    import json

    session = VIDEO_STORE.get(session_id)
    if session is None:
        return render_template(
            "session_expired.html",
            media_label="video",
            upload_url=url_for("video.index"),
        ), 404

    meta = _probe_meta(session.original_path)

    return render_template(
        "video_editor.html",
        session_id=session_id,
        techniques=TECHNIQUES,
        techniques_json=json.dumps(TECHNIQUES),
        frame_effects_json=json.dumps(serialize_effects(_frame_effects())),
        has_motion_clip=session.motion_path is not None,
        meta=meta,
    )


@video_bp.route("/<session_id>/heading_media", methods=["GET"])
def heading_media(session_id: str):
    """Silent, mp4-remuxed copy of the video for the masked-heading background.

    Safari's muted-autoplay exemption is unreliable for a video that carries
    an audio track even when `muted` is set client-side - stripping the
    audio track (and remuxing to a plain .mp4 container) server-side removes
    the ambiguity entirely.
    """
    session = VIDEO_STORE.get(session_id)
    if session is None:
        abort(404)
    heading_path = UPLOADS_DIR / f"{session_id}_heading.mp4"
    if not heading_path.exists():
        ffmpeg.run(["-i", str(session.original_path), "-an", "-c:v", "copy", "-movflags", "+faststart", "-f", "mp4", str(heading_path)])
    return send_file(heading_path, conditional=True)


@video_bp.route("/<session_id>/thumbnail", methods=["GET"])
def thumbnail(session_id: str):
    session = VIDEO_STORE.get(session_id)
    if session is None:
        abort(404)
    first = next(extract_frames(session.original_path), None)
    if first is None:
        abort(400, description="could not read a frame from the uploaded video")
    return send_file(BytesIO(save_image(first, fmt="PNG")), mimetype="image/png")


@video_bp.route("/<session_id>/frame_preview", methods=["POST"])
def frame_preview(session_id: str):
    """Runs a per-frame image effect on just the video's first frame, for
    instant live previewing without waiting on a full ffmpeg job."""
    session = VIDEO_STORE.get(session_id)
    if session is None:
        abort(404)

    effect_name = request.form.get("frame_effect")
    eligible = {e.name: e for e in _frame_effects()}
    if effect_name not in eligible:
        abort(400, description=f"unknown or ineligible frame_effect {effect_name!r}")
    effect = eligible[effect_name]

    raw_params = {k: v for k, v in request.form.items() if k != "frame_effect"}
    params = coerce_params(effect.params, raw_params)

    first = next(extract_frames(session.original_path), None)
    if first is None:
        abort(400, description="could not read a frame from the uploaded video")

    try:
        result = effect.fn(first, **params)
    except Exception as exc:  # user-controlled slider input reaching numerical code
        abort(400, description=f"Effect failed with current parameters: {exc}")

    return send_file(BytesIO(save_image(result, fmt="PNG")), mimetype="image/png")


@video_bp.route("/<session_id>/process", methods=["POST"])
def process(session_id: str):
    session = VIDEO_STORE.get(session_id)
    if session is None:
        abort(404)

    technique = request.form.get("technique")
    spec = TECHNIQUES.get(technique)
    if spec is None:
        abort(400, description=f"unknown technique {technique!r}")

    if spec["needs_motion_clip"] and session.motion_path is None:
        abort(400, description="This technique needs a second (motion) clip - upload one first")

    params: dict = {}
    frame_effect_name = None
    frame_effect_params: dict = {}
    sub_technique_name = None
    vary_param = None
    vary_start = None
    vary_end = None
    vary_ping_pong = False
    if spec.get("frame_effect_bridge"):
        frame_effect_name = request.form.get("frame_effect")
        eligible = {e.name: e for e in _frame_effects()}
        if frame_effect_name not in eligible:
            abort(400, description=f"unknown or ineligible frame_effect {frame_effect_name!r}")
        effect = eligible[frame_effect_name]

        vary_param = request.form.get("vary_param") or None
        if vary_param is not None:
            vary_spec = next((p for p in effect.params if p.name == vary_param), None)
            if vary_spec is None or vary_spec.kind not in ("int", "float") or vary_spec.min is None or vary_spec.max is None:
                abort(400, description=f"{vary_param!r} is not a varyable parameter of {frame_effect_name!r}")
            try:
                vary_start = float(request.form.get("vary_start", vary_spec.min))
                vary_end = float(request.form.get("vary_end", vary_spec.max))
            except (TypeError, ValueError):
                abort(400, description="vary_start and vary_end must be numbers")
            vary_ping_pong = request.form.get("vary_loop_style") == "ping_pong"

        excluded = {"technique", "frame_effect", "trim_preview", "vary_param", "vary_start", "vary_end", "vary_loop_style"}
        raw_params = {k: v for k, v in request.form.items() if k not in excluded}
        frame_effect_params = coerce_params(effect.params, raw_params)
    elif spec.get("sub_techniques"):
        sub_technique_name = request.form.get("sub_technique")
        sub_spec = spec["sub_techniques"].get(sub_technique_name)
        if sub_spec is None:
            abort(400, description=f"unknown sub_technique {sub_technique_name!r} for technique {technique!r}")
        params = {p["name"]: _coerce(request.form.get(p["name"], p["default"]), p["kind"]) for p in sub_spec["params"]}
    else:
        params = {p["name"]: _coerce(request.form.get(p["name"], p["default"]), p["kind"]) for p in spec["params"]}

    trim = request.form.get("trim_preview") in ("true", "on", "1")

    output_path = OUTPUTS_DIR / f"{uuid.uuid4().hex}.mp4"
    original_path = session.original_path
    motion_path = session.motion_path

    def job_fn(on_progress):
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            base_path = _maybe_trim(original_path, trim, tmp_dir, "base" + original_path.suffix)

            if technique == "per_frame":
                if vary_param is not None:
                    frame_effects.apply_frame_effect_with_sweep(
                        str(base_path),
                        str(output_path),
                        frame_effect_name,
                        frame_effect_params,
                        vary_param,
                        vary_start,
                        vary_end,
                        ping_pong=vary_ping_pong,
                        on_progress=on_progress,
                    )
                else:
                    frame_effects.apply_frame_effect(
                        str(base_path),
                        str(output_path),
                        frame_effect_name,
                        frame_effect_params,
                        on_progress=on_progress,
                    )
            elif technique == "frame_blend":
                datamosh.frame_blend(str(base_path), str(output_path), **params)
            elif technique == "change_speed":
                if sub_technique_name == "speed_up":
                    slowmo.speed_up(str(base_path), str(output_path), on_progress=on_progress, **params)
                else:
                    slowmo.slow_motion(str(base_path), str(output_path), on_progress=on_progress, **params)
            elif technique == "feedback_loop":
                datamosh.feedback_loop(str(base_path), str(output_path), **params)
            elif technique == "iframe_smear":
                trimmed_motion = _maybe_trim(motion_path, trim, tmp_dir, "motion" + motion_path.suffix)
                datamosh.iframe_smear(str(base_path), str(trimmed_motion), str(output_path), **params)
        return output_path

    job_id = JOB_TRACKER.start(job_fn)
    return jsonify({"job_id": job_id})


@video_bp.route("/<session_id>/jobs/<job_id>/status", methods=["GET"])
def job_status(session_id: str, job_id: str):
    job = JOB_TRACKER.get(job_id)
    if job is None:
        abort(404)
    return jsonify({"status": job.status, "progress": job.progress, "error": job.error})


@video_bp.route("/<session_id>/jobs/<job_id>/result", methods=["GET"])
def job_result(session_id: str, job_id: str):
    job = JOB_TRACKER.get(job_id)
    if job is None or job.status != "done" or job.result_path is None:
        abort(404)
    return send_file(str(job.result_path), mimetype="video/mp4")


@video_bp.route("/<session_id>/apply/<job_id>", methods=["POST"])
def apply_job_result(session_id: str, job_id: str):
    """Bakes a finished full-render job's output in as the session's new
    base clip, so a further technique can be cascaded on top of it."""
    session = VIDEO_STORE.get(session_id)
    if session is None:
        abort(404)
    job = JOB_TRACKER.get(job_id)
    if job is None or job.status != "done" or job.result_path is None:
        abort(404)

    new_path = UPLOADS_DIR / f"{session_id}_{uuid.uuid4().hex}.mp4"
    shutil.copy2(job.result_path, new_path)
    VIDEO_STORE.set_original(session_id, new_path)

    # The masked-heading video is cached by session id only - drop the stale
    # copy so it gets regenerated from the new base clip on next fetch.
    heading_path = UPLOADS_DIR / f"{session_id}_heading.mp4"
    heading_path.unlink(missing_ok=True)

    return jsonify({"ok": True, "meta": _probe_meta(new_path)})
