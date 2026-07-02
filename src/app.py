"""
Gradio web application for the Hand Gesture Classifier.

Uses MediaPipe hand cropping + MobileNetV2 model for live predictions.

Tabs:
  1. Upload Image — upload a photo file for classification
  2. Webcam       — live streaming prediction with smoothed output + crop preview

Usage:
  python src/app.py
"""

import os
import sys

import gradio as gr

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from config import MODEL_PATH  # noqa: E402
from predict import (  # noqa: E402
    crop_for_model,
    crop_preview,
    format_confidence,
    format_fallback_notice,
    load_model_and_classes,
    predict_probs,
    smooth_probs,
)

SMOOTH_WINDOW = 5
PREVIEW_HEIGHT = 550
CROP_PREVIEW_HEIGHT = 248
CONFIDENCE_MIN_HEIGHT = 290


def build_interface(model, idx_to_class):
    def predict_upload(image):
        if image is None:
            return {}, None, gr.update(value="", visible=False)

        cropped, hand_detected = crop_for_model(image, streaming=False)
        probs = predict_probs(model, cropped)
        confidence = format_confidence(probs, idx_to_class)
        fallback = format_fallback_notice(hand_detected)
        return (
            confidence,
            crop_preview(cropped),
            gr.update(value=fallback, visible=bool(fallback)),
        )

    def predict_webcam(image, prob_history):
        if image is None:
            return {}, None, prob_history, gr.update(value="", visible=False)

        cropped, hand_detected = crop_for_model(image, streaming=True)
        probs = predict_probs(model, cropped)

        prob_history = list(prob_history or [])
        prob_history.append(probs.tolist())
        prob_history = prob_history[-SMOOTH_WINDOW:]

        smoothed = smooth_probs(prob_history, SMOOTH_WINDOW)
        confidence = format_confidence(smoothed, idx_to_class)
        fallback = format_fallback_notice(hand_detected)
        return (
            confidence,
            crop_preview(cropped),
            prob_history,
            gr.update(value=fallback, visible=bool(fallback)),
        )

    css = f"""
    .gradio-container {{
        font-family: 'Segoe UI', Arial, sans-serif;
    }}
    #title {{
        text-align: center;
        margin-bottom: 0.5rem;
    }}
    #subtitle {{
        text-align: center;
        color: #666;
        margin-bottom: 1.5rem;
        font-size: 0.95rem;
    }}
    .gradio-container #fallback_notice,
    .gradio-container #fallback_notice > div,
    .gradio-container #fallback_notice .wrap,
    .gradio-container #fallback_notice .html-container,
    .gradio-container #fallback_notice .prose,
    .gradio-container .fallback-notice-html.block {{
        width: 100% !important;
        max-width: 100% !important;
        padding: 0 !important;
        margin: 0 !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
        background: transparent !important;
    }}
    .gradio-container #fallback_notice .prose p,
    .gradio-container .fallback-notice-html .prose p {{
        margin: 0 !important;
        padding: 0 !important;
    }}
    .gradio-container .fallback-notice-card {{
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        width: 100%;
        border-radius: 10px;
        border: 1px solid #fbbf24 !important;
        background: #fffbeb;
        box-sizing: border-box;
        padding: 1rem 1.15rem;
        margin: 0 0 0.75rem 0;
        color: #92400e;
        font-size: 0.95rem;
        line-height: 1.5;
    }}
    .gradio-container .fallback-notice-icon {{
        flex-shrink: 0;
        font-size: 1.1rem;
        line-height: 1.2;
    }}
    .gradio-container .fallback-notice-text {{
        flex: 1;
    }}
    .gradio-container .fallback-notice-text strong {{
        font-weight: 600;
        color: #78350f;
    }}
    #upload_input, #webcam_input {{
        height: {PREVIEW_HEIGHT}px !important;
    }}
    #upload_input .image-container, #webcam_input .image-container {{
        height: {PREVIEW_HEIGHT}px !important;
    }}
    #upload_input img, #webcam_input img,
    #upload_input video, #webcam_input video {{
        height: {PREVIEW_HEIGHT}px !important;
        object-fit: contain !important;
    }}
    #upload_crop_preview, #webcam_crop_preview {{
        height: {CROP_PREVIEW_HEIGHT}px !important;
    }}
    #upload_crop_preview .image-container, #webcam_crop_preview .image-container {{
        height: {CROP_PREVIEW_HEIGHT}px !important;
    }}
    #upload_crop_preview img, #webcam_crop_preview img {{
        height: {CROP_PREVIEW_HEIGHT}px !important;
        object-fit: contain !important;
    }}
    .results-panel {{
        gap: 0.75rem;
    }}
    #upload_confidence, #webcam_confidence {{
        min-height: {CONFIDENCE_MIN_HEIGHT}px !important;
    }}
    #upload_confidence > .wrap, #webcam_confidence > .wrap,
    #upload_confidence .label-wrap, #webcam_confidence .label-wrap {{
        min-height: {CONFIDENCE_MIN_HEIGHT}px !important;
    }}
    #class_reference,
    .class-reference-html,
    .class-reference-html > div,
    .class-reference-html .html-container,
    .class-reference-html .prose {{
        width: 100% !important;
        max-width: 100% !important;
        padding: 0 !important;
        margin: 0 !important;
        border: none !important;
        box-shadow: none !important;
        background: transparent !important;
    }}
    .class-reference-html .class-ref-card {{
        width: 100%;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #d7e3f4 !important;
        outline: none !important;
        box-shadow: none !important;
        background: #ffffff;
        box-sizing: border-box;
    }}
    .gradio-container .class-reference-html .prose table,
    .gradio-container .class-reference-html .prose thead,
    .gradio-container .class-reference-html .prose tbody,
    .gradio-container .class-reference-html .prose tr,
    .gradio-container .class-reference-html .prose th,
    .gradio-container .class-reference-html .prose td {{
        border: 0 !important;
        box-shadow: none !important;
        outline: none !important;
    }}
    #class_reference_accordion,
    #class_reference_accordion > div,
    #class_reference_accordion details,
    #class_reference_accordion summary,
    #class_reference_accordion .label-wrap,
    #class_reference_accordion .wrap,
    #class_reference_accordion .accordion-content,
    #class_reference_accordion .accordion-content > div {{
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }}
    #class_reference_accordion .accordion-content {{
        padding: 0 !important;
        margin: 0 !important;
    }}
    #class_reference_accordion button.label-wrap {{
        align-items: center !important;
    }}
    #class_reference_accordion button.label-wrap > span:first-of-type {{
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }}
    #class_reference_accordion button.label-wrap > span:first-of-type::before {{
        content: "";
        display: inline-block;
        flex-shrink: 0;
        width: 1rem;
        height: 1rem;
        background-color: #1a73e8;
        mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M4 19.5A2.5 2.5 0 0 1 6.5 17H20'/%3E%3Cpath d='M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z'/%3E%3C/svg%3E") center / contain no-repeat;
        -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M4 19.5A2.5 2.5 0 0 1 6.5 17H20'/%3E%3Cpath d='M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z'/%3E%3C/svg%3E") center / contain no-repeat;
    }}
    #class_reference_accordion button.label-wrap .icon {{
        display: inline-block;
        flex-shrink: 0;
        align-self: center;
        font-size: 0 !important;
        color: transparent !important;
        width: 0.45rem;
        height: 0.45rem;
        border: none;
        border-right: 2px solid #64748b;
        border-bottom: 2px solid #64748b;
        background: transparent !important;
        transition: transform 0.15s ease;
        transform: rotate(45deg) !important;
    }}
    #class_reference_accordion button.label-wrap.open .icon {{
        transform: rotate(225deg) !important;
    }}
    #class_reference_accordion::before,
    #class_reference_accordion::after,
    #class_reference_accordion details::before,
    #class_reference_accordion details::after,
    #class_reference_accordion summary::before,
    #class_reference_accordion summary::after {{
        content: none !important;
        border: 0 !important;
        box-shadow: none !important;
        display: none !important;
    }}
    .gradio-container .class-reference-html table.class-ref-table {{
        width: 100% !important;
        border-collapse: collapse !important;
        border-spacing: 0 !important;
        table-layout: fixed;
        margin: 0 !important;
        border: 0 !important;
        background: #ffffff;
    }}
    .gradio-container .class-reference-html table.class-ref-table th,
    .gradio-container .class-reference-html table.class-ref-table td {{
        border: 0 !important;
        border-style: none !important;
        border-width: 0 !important;
        border-color: transparent !important;
        box-shadow: none !important;
        outline: none !important;
        background-clip: padding-box;
    }}
    .gradio-container .class-reference-html table.class-ref-table thead tr {{
        background: linear-gradient(135deg, #1a73e8 0%, #4f8df7 100%) !important;
    }}
    .gradio-container .class-reference-html table.class-ref-table thead th {{
        background: transparent !important;
        color: #ffffff !important;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        padding: 0.9rem 1rem;
        text-align: center;
    }}
    .gradio-container .class-reference-html table.class-ref-table tbody tr {{
        transition: background 0.15s ease;
    }}
    .gradio-container .class-reference-html table.class-ref-table tbody tr:nth-child(even) {{
        background: #f8fafc !important;
    }}
    .gradio-container .class-reference-html table.class-ref-table tbody tr:hover {{
        background: #eef4ff !important;
    }}
    .gradio-container .class-reference-html table.class-ref-table tbody tr + tr td {{
        border-top: 1px solid #e8eef5 !important;
        border-left: 0 !important;
        border-right: 0 !important;
        border-bottom: 0 !important;
    }}
    .gradio-container .class-reference-html table.class-ref-table tbody td {{
        padding: 0.9rem 1rem;
        vertical-align: middle;
        font-size: 0.92rem;
        color: #334155;
        text-align: center;
        background: transparent !important;
    }}
    .gradio-container .class-reference-html table.class-ref-table tbody td:first-child {{
        font-size: 1.6rem;
        width: 80px;
    }}
    .gradio-container .class-reference-html table.class-ref-table tbody td:nth-child(2) {{
        font-weight: 600;
        color: #1e293b;
        width: 140px;
    }}
    .gradio-container .class-reference-html table.class-ref-table tbody td:last-child {{
        text-align: left;
        color: #64748b;
        line-height: 1.5;
    }}
    """

    with gr.Blocks(title="Hand Gesture Classifier") as demo:

        gr.Markdown(
            "# ✌️ 👌 👊 👍 🤚  Hand Gesture Classifier",
            elem_id="title",
        )
        gr.Markdown(
            "Upload a photo or use your webcam. MediaPipe crops your hand, then the CNN classifies the gesture.",
            elem_id="subtitle",
        )

        with gr.Tabs():
            with gr.TabItem("📁 Upload Image"):
                gr.Markdown("_Select or drop an image — prediction appears instantly._")
                with gr.Row():
                    with gr.Column(scale=3):
                        upload_input = gr.Image(
                            type="pil",
                            label="Upload a hand gesture image",
                            sources=["upload"],
                            elem_id="upload_input",
                            height=PREVIEW_HEIGHT,
                        )

                    with gr.Column(scale=2, elem_classes=["results-panel"]):
                        upload_crop_preview = gr.Image(
                            type="pil",
                            label="Model input (hand crop)",
                            interactive=False,
                            elem_id="upload_crop_preview",
                            height=CROP_PREVIEW_HEIGHT,
                        )
                        upload_confidence = gr.Label(
                            label="Confidence Scores",
                            num_top_classes=5,
                            elem_id="upload_confidence",
                        )

            with gr.TabItem("📷 Webcam"):
                gr.Markdown(
                    "_Point your hand at the camera — prediction is smoothed over the last "
                    f"{SMOOTH_WINDOW} frames to reduce flicker._",
                )
                webcam_prob_history = gr.State([])

                with gr.Row():
                    with gr.Column(scale=3):
                        webcam_input = gr.Image(
                            type="pil",
                            label="Live Webcam",
                            sources=["webcam"],
                            streaming=True,
                            elem_id="webcam_input",
                            height=PREVIEW_HEIGHT,
                        )

                    with gr.Column(scale=2, elem_classes=["results-panel"]):
                        webcam_crop_preview = gr.Image(
                            type="pil",
                            label="Model input (hand crop)",
                            interactive=False,
                            elem_id="webcam_crop_preview",
                            height=CROP_PREVIEW_HEIGHT,
                        )
                        webcam_confidence = gr.Label(
                            label="Confidence Scores (smoothed)",
                            num_top_classes=5,
                            elem_id="webcam_confidence",
                        )

        fallback_notice = gr.HTML(
            visible=False,
            elem_id="fallback_notice",
            elem_classes=["fallback-notice-html"],
        )

        with gr.Accordion("Class Reference Guide", open=False, elem_id="class_reference_accordion"):
            gr.HTML("""
<div class="class-ref-card">
<table class="class-ref-table">
<thead>
<tr>
<th>Key</th>
<th>Class</th>
<th>Gesture Description</th>
</tr>
</thead>
<tbody>
<tr><td>✌️</td><td>Peace</td><td>Index and middle fingers extended, others closed</td></tr>
<tr><td>👌</td><td>Okay</td><td>Thumb and index finger forming a circle</td></tr>
<tr><td>👊</td><td>Fist</td><td>All fingers closed into a fist</td></tr>
<tr><td>👍</td><td>Thumbs Up</td><td>Thumb pointing up, fingers closed</td></tr>
<tr><td>🤚</td><td>High Five</td><td>Open palm facing forward, all fingers extended</td></tr>
</tbody>
</table>
</div>
""", elem_id="class_reference", elem_classes=["class-reference-html"])

        upload_input.change(
            fn=predict_upload,
            inputs=upload_input,
            outputs=[upload_confidence, upload_crop_preview, fallback_notice],
        )
        webcam_input.stream(
            fn=predict_webcam,
            inputs=[webcam_input, webcam_prob_history],
            outputs=[
                webcam_confidence,
                webcam_crop_preview,
                webcam_prob_history,
                fallback_notice,
            ],
            time_limit=30,
            stream_every=0.15,
        )

        gr.Markdown(
            "CSC583 Group Project — Hand Gesture CNN Classifier (MediaPipe + MobileNetV2)",
            elem_id="subtitle",
        )

    return demo, css


def main():
    print("=" * 60)
    print("Hand Gesture Classifier — Web App")
    print("=" * 60)
    print(f"Loading model from: {MODEL_PATH}")

    try:
        model, idx_to_class = load_model_and_classes()
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)

    print(f"Model loaded. Classes: {list(idx_to_class.values())}")

    demo, css = build_interface(model, idx_to_class)

    print("\nStarting Gradio server...")
    print("Open your browser at: http://localhost:7860\n")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(primary_hue="blue"),
        css=css,
    )


if __name__ == "__main__":
    main()
