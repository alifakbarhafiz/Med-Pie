"""
Med-Pie - Professional Medical Imaging Demonstration Platform
Main Streamlit Application
"""
import streamlit as st
import torch
import numpy as np
from PIL import Image
import io
import warnings
import os

# Suppress torch.classes warning from Streamlit file watcher
# This is a known issue with Streamlit and PyTorch
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'poll'
# Suppress the specific RuntimeError about torch.classes
warnings.filterwarnings('ignore', category=RuntimeWarning)
# Also suppress the error in stderr by catching it
import sys
from io import StringIO
# This is a workaround for the torch.classes issue - it's harmless
if hasattr(sys, '_getframe'):
    try:
        # Try to suppress the error by setting the environment variable earlier
        pass
    except:
        pass

# Import custom modules
from config import MODEL_WEIGHTS_PATH, TB_TYPE_VALUES
from models.lumina_inference import LuminaInference
from utils.preprocessing import denormalize, apply_clahe
from utils.visualization import draw_bounding_boxes_on_image
from utils.gradcam import generate_gradcam_heatmap
from components.sidebar import render_sidebar
from components.cards import render_classification_card, render_severity_gauge
from reports.pdf_generator import generate_clinical_pdf

# Page Configuration
st.set_page_config(
    page_title="Med-Pie - Pro",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom CSS
with open("styles/custom.css", "r") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_resource
def load_model(weights_path):
    """Load and cache the model."""
    try:
        model = LuminaInference(weights_path)
        return model, True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, False


def main():
    """Main application function."""
    # Initialize session state
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False
    if "inference_result" not in st.session_state:
        st.session_state.inference_result = None
    if "uploaded_image" not in st.session_state:
        st.session_state.uploaded_image = None
    
    # Sidebar
    nms_threshold, enhanced_viz = render_sidebar(st.session_state.model_loaded)
    
    # Load model
    if not st.session_state.model_loaded:
        with st.spinner("Loading Med-Pie model..."):
            model, loaded = load_model(MODEL_WEIGHTS_PATH)
            if loaded:
                st.session_state.model = model
                st.session_state.model_loaded = True
                st.rerun()
    
    # Main Content Area
    st.markdown("""
    <div style='text-align: center; padding: 40px 0;'>
        <h1 class='lumina-title'>Med-Pie</h1>
        <p class='lumina-subtitle'>Professional Medical Imaging Analysis Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Image Upload Section
    if st.session_state.inference_result is None:
        st.markdown("---")
        
        uploaded_file = st.file_uploader(
            "Upload Chest X-ray Image",
            type=["png", "jpg", "jpeg"],
            help="Upload a chest X-ray image for TB detection and analysis"
        )
        
        if uploaded_file is not None:
            # Load and display image
            image = Image.open(uploaded_file).convert("RGB")
            st.session_state.uploaded_image = image
            
            # Apply CLAHE if enabled
            if enhanced_viz:
                image_np = np.array(image)
                image_np = apply_clahe(image_np)
                image = Image.fromarray((image_np * 255).astype(np.uint8))
            
            # Run inference
            if st.session_state.model_loaded:
                with st.spinner("Med-Pie is analyzing clinical features..."):
                    try:
                        result = st.session_state.model.predict(
                            image,
                            nms_threshold=nms_threshold,
                            score_threshold=nms_threshold
                        )
                        st.session_state.inference_result = result
                        st.rerun()
                    except Exception as e:
                        st.error(f"Inference error: {e}")
            else:
                st.warning("Model not loaded. Please check the model weights path.")
    
    # Display Results
    if st.session_state.inference_result is not None and st.session_state.uploaded_image is not None:
        result = st.session_state.inference_result
        image = st.session_state.uploaded_image
        
        # Stage 1: At-a-Glance Row (Unified Single Box - Two Columns)
        st.markdown("""
        <h2 style='font-size: 24px; font-weight: 700; color: #1D1D1F; margin: 32px 0 16px 0;'>
            At-a-Glance Analysis
        </h2>
        """, unsafe_allow_html=True)
        
        # Unified single box container with two columns - no margin-top to eliminate gap
        st.markdown("""
        <div style='background: #FFFFFF; border-radius: 16px; padding: 24px; 
                    box-shadow: 0 4px 12px rgba(0,0,0,0.08); margin-bottom: 24px; 
                    display: flex; align-items: stretch; margin-top: 0;'>
        """, unsafe_allow_html=True)
        
        # Two columns: Image on left, Predictions (Diagnosis + Severity) on right
        col_image, col_predictions = st.columns([1.5, 1], gap="large")
        
        with col_image:
            # Wrap in container to ensure proper height matching
            st.markdown("""
            <div style='display: flex; flex-direction: column; height: 100%; width: 100%;'>
            """, unsafe_allow_html=True)
            
            # X-ray image with detected lesions (no label box above)
            # Get classification for color coding
            cls_probs = result["cls_probs"]
            if hasattr(cls_probs, 'argmax'):
                pred_idx = cls_probs.argmax().item()
            else:
                pred_idx = max(range(len(cls_probs)), key=lambda i: cls_probs[i])
            
            # Draw bounding boxes with labels
            image_np = np.array(image) / 255.0
            if len(result["boxes"]) > 0:
                viz_image = draw_bounding_boxes_on_image(
                    image_np,
                    result["boxes"],
                    result["scores"],
                    result["labels"],
                    score_threshold=nms_threshold
                )
            else:
                viz_image = image_np
            
            st.image(viz_image, use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_predictions:
            # Get all data first
            cls_name, confidence, pred_idx = st.session_state.model.get_classification(result)
            
            # Format classification name
            display_name = cls_name.replace('_', ' ').title()
            if cls_name == 'none':
                display_name = "No TB Detected"
            elif cls_name == 'latent_tb':
                display_name = "Latent TB"
            elif cls_name == 'active_tb':
                display_name = "Active TB"
            
            # Get color
            from config import TB_COLORS, COLORS
            color = TB_COLORS.get(pred_idx, COLORS['accent_blue'])
            
            # Get severity
            severity = result.get("severity", 0.0)
            if isinstance(severity, (int, float)):
                severity_value = float(severity)
            else:
                severity_value = float(severity)
            
            # Determine severity color and status
            if severity_value < 0.33:
                severity_color = COLORS['success_green']
                severity_status = "Low"
            elif severity_value < 0.67:
                severity_color = COLORS['warning_orange']
                severity_status = "Moderate"
            else:
                severity_color = COLORS['critical_red']
                severity_status = "High"
            
            # Prepare summary and recommendation
            if cls_name == "active_tb":
                summary_text = "Active tuberculosis detected with radiographic evidence of disease activity."
                if severity_value >= 0.7:
                    recommendation = "Immediate medical evaluation and treatment required. High severity indicates extensive disease."
                elif severity_value >= 0.4:
                    recommendation = "Urgent medical consultation recommended. Moderate severity requires prompt treatment."
                else:
                    recommendation = "Medical evaluation advised. Early treatment can prevent disease progression."
                summary_color = "#FF3B30"
            elif cls_name == "latent_tb":
                summary_text = "Latent tuberculosis infection detected. Bacteria present but inactive."
                recommendation = "Consult healthcare provider for preventive treatment options to reduce risk of progression."
                summary_color = "#FF9500"
            else:
                summary_text = "No tuberculosis detected in the chest X-ray analysis."
                recommendation = "Continue routine health monitoring. No immediate action required."
                summary_color = "#34C759"
            
            # Confidence values
            confidence_pct = confidence * 100
            confidence_width = f"{confidence_pct}%"
            confidence_display = f"{confidence_pct:.1f}%"
            
            # Single unified bento box - matches image height
            unified_box_html = (
                "<div class='lumina-card' style='text-align: center; padding: 28px; height: 100%; "
                "display: flex; flex-direction: column; justify-content: space-between;'>"
                
                # Diagnosis Section
                "<div style='flex-shrink: 0;'>"
                "<div class='lumina-label' style='margin-bottom: 12px;'>Diagnosis</div>"
                f"<div class='lumina-value' style='color: {color}; font-size: 28px; margin: 12px 0;'>"
                f"{display_name}"
                "</div>"
                "<div style='display: flex; flex-direction: column; align-items: center; gap: 8px; margin: 16px 0;'>"
                "<div style='font-size: 13px; color: #86868B;'>Confidence</div>"
                "<div style='width: 100%; max-width: 200px; height: 8px; background: rgba(0,0,0,0.1); border-radius: 4px; overflow: hidden; margin: 0 auto;'>"
                f"<div style='width: {confidence_width}; height: 100%; background: {color}; border-radius: 4px; transition: width 0.5s ease;'></div>"
                "</div>"
                f"<div style='font-size: 16px; font-weight: 600; color: {color}; margin-top: 4px;'>"
                f"{confidence_display}"
                "</div>"
                "</div>"
                "</div>"
                
                # Divider
                "<div style='height: 1px; background: rgba(0,0,0,0.1); margin: 20px 0;'></div>"
                
                # Severity Section
                "<div style='flex-shrink: 0;'>"
                "<div class='lumina-label' style='margin-bottom: 12px;'>Severity Index</div>"
                f"<div class='lumina-value' style='color: {severity_color}; font-size: 28px; margin: 12px 0;'>"
                f"{severity_value:.2f} <span style='font-size: 18px; color: #86868B;'>/ 1.0</span>"
                "</div>"
                "<div class='severity-gauge' style='margin: 16px 0;'>"
                f"<div class='severity-indicator' style='width: {severity_value * 100}%;'></div>"
                "</div>"
                f"<div style='font-size: 14px; color: {severity_color}; font-weight: 600; margin-top: 8px;'>"
                f"{severity_status} Severity"
                "</div>"
                "</div>"
                
                # Divider
                "<div style='height: 1px; background: rgba(0,0,0,0.1); margin: 20px 0;'></div>"
                
                # Quick Summary Section - expands to fill remaining space
                "<div style='flex: 1; display: flex; flex-direction: column; justify-content: flex-start; min-height: 0;'>"
                "<div style='font-size: 13px; font-weight: 600; color: #86868B; text-transform: uppercase; "
                "letter-spacing: 0.5px; margin-bottom: 16px; text-align: center;'>Quick Summary</div>"
                f"<div style='font-size: 14px; color: #1D1D1F; line-height: 1.7; margin-bottom: 20px; text-align: center;'>"
                f"{summary_text}"
                "</div>"
                "<div style='font-size: 13px; font-weight: 600; color: #86868B; text-transform: uppercase; "
                "letter-spacing: 0.5px; margin-bottom: 16px; text-align: center;'>Recommendation</div>"
                f"<div style='font-size: 14px; color: {summary_color}; line-height: 1.7; font-weight: 500; text-align: center;'>"
                f"{recommendation}"
                "</div>"
                "</div>"
                
                "</div>"
            )
            st.markdown(unified_box_html, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Stage 2: Deep Analysis
        st.markdown("---")
        st.markdown("""
        <h2 style='font-size: 24px; font-weight: 700; color: #1D1D1F; margin: 32px 0 24px 0;'>
            Spatial Intelligence & Saliency
        </h2>
        """, unsafe_allow_html=True)
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("""
            <div class='lumina-label' style='margin-bottom: 12px;'>Structural Localization</div>
            """, unsafe_allow_html=True)
            
            # Detection boxes visualization with labels
            image_np = np.array(image) / 255.0
            if len(result["boxes"]) > 0:
                det_image = draw_bounding_boxes_on_image(
                    image_np,
                    result["boxes"],
                    result["scores"],
                    result["labels"],
                    score_threshold=nms_threshold
                )
            else:
                det_image = image_np
            
            st.image(det_image, use_container_width=True)
        
        with col_right:
            st.markdown("""
            <div class='lumina-label' style='margin-bottom: 12px;'>Feature Activation (Grad-CAM)</div>
            """, unsafe_allow_html=True)
            
            # Grad-CAM visualization
            try:
                # Preprocess image for model
                img_tensor = st.session_state.model.preprocess_image(image)
                img_tensor = img_tensor.to(st.session_state.model.device)
                
                # Get target layer (FPN inner_blocks[3] - exact match with tb_inference.py)
                # From tb_inference.py line 897: target_layer = model.detector.backbone.fpn.inner_blocks[3]
                try:
                    target_layer = st.session_state.model.model.detector.backbone.fpn.inner_blocks[3]
                except (AttributeError, IndexError):
                    # Fallback: try to get the last FPN block
                    try:
                        fpn = st.session_state.model.model.detector.backbone.fpn
                        if hasattr(fpn, 'inner_blocks') and len(fpn.inner_blocks) > 0:
                            target_layer = fpn.inner_blocks[-1]
                        else:
                            # Last resort: use ConvNeXt backbone last stage
                            backbone = st.session_state.model.model.detector.backbone.backbone
                            if hasattr(backbone, 'stages'):
                                target_layer = backbone.stages[-1]
                            else:
                                raise AttributeError("Could not find suitable target layer")
                    except Exception:
                        raise AttributeError("Could not find suitable target layer")
                
                # Generate Grad-CAM with better error reporting
                import traceback
                cam_overlay, cam_raw, _ = generate_gradcam_heatmap(
                    st.session_state.model.model,
                    img_tensor,
                    target_layer,
                    class_idx=pred_idx
                )
                
                if cam_overlay is not None:
                    st.image(cam_overlay, use_container_width=True)
                else:
                    st.warning("Grad-CAM returned None. Showing original image.")
                    st.image(image, use_container_width=True)
                    
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                st.error(f"Grad-CAM generation error: {e}")
                st.code(error_trace, language='python')
                st.info("Grad-CAM visualization is currently unavailable. Showing original image.")
                st.image(image, use_container_width=True)
        
        # Footer caption
        st.markdown("""
        <div style='text-align: center; margin-top: 16px; padding: 16px; 
                    background: rgba(0, 0, 0, 0.02); border-radius: 8px;'>
            <p style='font-size: 12px; color: #86868B; font-style: italic; margin: 0;'>
                The heatmap visualization (Grad-CAM) highlights regions of the image that contribute 
                most to the classification decision, providing spatial correlation with detected bounding boxes.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Stage 3: Detailed Clinical Report & Analysis
        st.markdown("---")
        st.markdown("""
        <h2 style='font-size: 24px; font-weight: 700; color: #1D1D1F; margin: 32px 0 24px 0;'>
            💡 Detailed Interpretation & Analysis
        </h2>
        """, unsafe_allow_html=True)
        
        # Extract all prediction data
        num_boxes = len(result.get("boxes", []))
        cls_name, confidence, pred_idx = st.session_state.model.get_classification(result)
        severity = result.get("severity", 0.0)
        if isinstance(severity, (int, float)):
            severity_value = float(severity)
        else:
            severity_value = float(severity)
        
        # Format classification name
        display_name = cls_name.replace('_', ' ').title()
        if cls_name == 'none':
            display_name = "No TB Detected"
        elif cls_name == 'latent_tb':
            display_name = "Latent TB"
        elif cls_name == 'active_tb':
            display_name = "Active TB"
        
        # Get probabilities
        cls_probs = result.get("cls_probs", [])
        if isinstance(cls_probs, torch.Tensor):
            cls_probs = cls_probs.cpu().numpy()
        
        # Main prediction alert with color coding
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if cls_name == "active_tb":
                st.warning("""
                ### 🔴 Active Tuberculosis Detected
                
                **Clinical Significance**: The model has identified radiographic signs consistent with active tuberculosis. 
                Active TB is a contagious condition where the bacteria are actively multiplying and can be transmitted to others.
                
                **Key Indicators**:
                - Presence of TB lesions visible on chest X-ray
                - Radiographic patterns consistent with active disease
                - Potential for disease progression if untreated
                
                **⚠️ Important**: This requires **immediate medical evaluation** and treatment. Active TB is a serious 
                condition that can be life-threatening if left untreated.
                """)
            elif cls_name == "latent_tb":
                st.info("""
                ### 🟡 Latent Tuberculosis Detected
                
                **Clinical Significance**: The model suggests the presence of latent tuberculosis infection. 
                In latent TB, the bacteria are present but inactive, meaning the person is not contagious.
                
                **Key Characteristics**:
                - TB bacteria are present but dormant
                - No symptoms and not contagious
                - Can progress to active TB if immune system weakens
                - Treatment can prevent progression to active disease
                
                **Recommendation**: Consultation with a healthcare provider is recommended to discuss 
                preventive treatment options and monitoring.
                """)
            else:
                st.success("""
                ### 🟢 No Tuberculosis Detected
                
                **Clinical Significance**: The model did not detect radiographic signs of tuberculosis in this image.
                
                **Interpretation**:
                - No visible TB lesions or abnormalities detected
                - Chest X-ray appears normal for TB screening purposes
                - This is a negative screening result
                
                **Note**: A negative result does not completely rule out TB, especially in early stages. 
                Clinical correlation and follow-up may be recommended based on symptoms and risk factors.
                """)
        
        with col2:
            # Confidence indicator
            if confidence >= 0.9:
                conf_level = "Very High"
                conf_color = "🟢"
            elif confidence >= 0.7:
                conf_level = "High"
                conf_color = "🟡"
            else:
                conf_level = "Moderate"
                conf_color = "🟠"
            
            st.metric("Prediction Confidence", f"{confidence:.1%}", f"{conf_color} {conf_level}")
            
            # Severity level
            if severity_value < 0.33:
                severity_level = "Low"
            elif severity_value < 0.67:
                severity_level = "Moderate"
            else:
                severity_level = "High"
            
            st.metric("Severity Score", f"{severity_value:.3f}", 
                     "High" if severity_value > 0.5 else "Moderate" if severity_value > 0.2 else "Low")
            st.metric("Lesions Detected", num_boxes, 
                     "Visible" if num_boxes > 0 else "None")
        
        st.divider()
        
        # Detailed analysis tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Prediction Analysis", "🔍 Detection Details", "📈 Probability Breakdown", "ℹ️ Understanding Results"])
        
        with tab1:
            st.markdown("### Prediction Analysis")
            
            st.markdown(f"""
            **Overall Classification**: {display_name}
            
            The model analyzed the entire chest X-ray image and classified it as **{display_name}** 
            with a confidence of **{confidence:.1%}**. This global classification considers the overall pattern 
            and distribution of findings across the entire image.
            """)
            
            # Confidence interpretation
            st.markdown("#### Confidence Level Interpretation")
            if confidence >= 0.9:
                st.success(f"""
                **Very High Confidence ({confidence:.1%})**: The model is highly certain about this prediction. 
                The radiographic features strongly support the {display_name} classification.
                """)
            elif confidence >= 0.7:
                st.info(f"""
                **High Confidence ({confidence:.1%})**: The model is confident in this prediction, though some 
                uncertainty exists. The features are consistent with {display_name}.
                """)
            else:
                st.warning(f"""
                **Moderate Confidence ({confidence:.1%})**: The model has moderate certainty. The features suggest 
                {display_name} but may benefit from additional clinical correlation.
                """)
            
            # Severity interpretation
            st.markdown("#### Severity Score Interpretation")
            st.markdown(f"""
            **Severity Score: {severity_value:.3f}**
            
            The severity score (ranging from 0.0 to 1.0) indicates the estimated extent and severity of disease:
            """)
            
            if severity_value > 0.7:
                st.error(f"""
                - **High Severity ({severity_value:.3f})**: Indicates extensive disease involvement
                - Large lesion areas or multiple affected regions
                - May suggest advanced disease requiring immediate attention
                """)
            elif severity_value > 0.4:
                st.warning(f"""
                - **Moderate Severity ({severity_value:.3f})**: Moderate disease involvement
                - Some areas affected but not extensive
                - Requires medical evaluation and monitoring
                """)
            elif severity_value > 0.1:
                st.info(f"""
                - **Low-Moderate Severity ({severity_value:.3f})**: Mild disease involvement
                - Limited areas affected
                - Early detection may allow for effective treatment
                """)
            else:
                st.success(f"""
                - **Minimal Severity ({severity_value:.3f})**: Very limited or no disease involvement
                - Minimal radiographic changes detected
                - May represent early stage or resolved disease
                """)
        
        with tab2:
            st.markdown("### Detection Details")
            
            if num_boxes > 0:
                boxes = result.get("boxes", [])
                scores = result.get("scores", [])
                labels = result.get("labels", [])
                
                # Convert to numpy if needed
                if isinstance(boxes, torch.Tensor):
                    boxes = boxes.cpu().numpy()
                if isinstance(scores, torch.Tensor):
                    scores = scores.cpu().numpy()
                if isinstance(labels, torch.Tensor):
                    labels = labels.cpu().numpy()
                
                st.markdown(f"""
                **Total Lesions Detected**: {num_boxes}
                
                The Faster R-CNN detection model identified **{num_boxes} distinct lesion(s)** in the chest X-ray. 
                Each detection represents a localized area where the model identified potential TB-related abnormalities.
                """)
                
                # Count by type
                type_counts = {}
                for label in labels:
                    label_idx = int(label)
                    if 0 <= label_idx < len(TB_TYPE_VALUES):
                        tb_type = TB_TYPE_VALUES[label_idx]
                        type_counts[tb_type] = type_counts.get(tb_type, 0) + 1
                
                st.markdown("#### Lesion Distribution by Type")
                for tb_type, count in type_counts.items():
                    color_emoji = "🔴" if tb_type == "active_tb" else "🟡" if tb_type == "latent_tb" else "🟢"
                    st.markdown(f"- {color_emoji} **{tb_type.replace('_', ' ').title()}**: {count} lesion(s)")
                
                # Average confidence
                if len(scores) > 0:
                    avg_confidence = float(np.mean(scores))
                    st.metric("Average Detection Confidence", f"{avg_confidence:.2%}")
                
                # Detailed lesion table
                st.markdown("#### Individual Lesion Details")
                lesion_data = []
                for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                    label_idx = int(label)
                    if 0 <= label_idx < len(TB_TYPE_VALUES):
                        tb_type = TB_TYPE_VALUES[label_idx]
                    else:
                        tb_type = "unknown"
                    
                    x1, y1, x2, y2 = box
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    
                    lesion_data.append({
                        "Lesion #": i + 1,
                        "Type": tb_type.replace('_', ' ').title(),
                        "Confidence": f"{float(score):.2%}",
                        "Location": f"({int(x1)}, {int(y1)})",
                        "Size": f"{int(width)}×{int(height)} px",
                        "Area": f"{int(area)} px²"
                    })
                
                st.dataframe(lesion_data, use_container_width=True, hide_index=True)
                
                st.markdown("""
                **Understanding Bounding Boxes**:
                - Each colored rectangle represents a detected lesion
                - Box color indicates the predicted lesion type (red=active, blue=latent, green=none)
                - The confidence score shows how certain the model is about each detection
                - Larger boxes may indicate more extensive lesions
                """)
            else:
                st.info(f"""
                **No Lesions Detected Above Threshold**
                
                The detection model did not identify any distinct lesions above the current confidence threshold 
                ({nms_threshold:.0%}). This could mean:
                
                - **No visible lesions**: The image may not contain detectable TB lesions
                - **Threshold too high**: Try lowering the detection threshold in the sidebar
                - **Early stage disease**: Lesions may be too subtle for detection
                - **Diffuse patterns**: Some TB patterns may not present as discrete lesions
                
                **Note**: The global classification (shown above) may still indicate TB presence even without 
                discrete lesion detections, as it analyzes the entire image pattern.
                """)
        
        with tab3:
            st.markdown("### Probability Breakdown Analysis")
            
            if len(cls_probs) == 3:
                st.markdown("""
                The model provides probability scores for each TB classification category. These probabilities 
                represent the model's confidence that the image belongs to each category.
                """)
                
                # Probability bars with interpretation
                prob_names = ["No TB", "Latent TB", "Active TB"]
                for i, (name, prob) in enumerate(zip(prob_names, cls_probs)):
                    # Convert to Python float (handle numpy/torch types)
                    prob_float = float(prob) if not isinstance(prob, float) else prob
                    prob_val = prob_float * 100
                    st.markdown(f"#### {name}: {prob_val:.2%}")
                    st.progress(prob_float, text=f"{name}: {prob_val:.2%}")
                    
                    # Interpretation for each
                    if i == 2:  # active_tb
                        if prob > 0.7:
                            st.markdown("**High probability** - Strong evidence for active TB")
                        elif prob > 0.4:
                            st.markdown("**Moderate probability** - Some evidence for active TB")
                        else:
                            st.markdown("**Low probability** - Limited evidence for active TB")
                    elif i == 1:  # latent_tb
                        if prob > 0.7:
                            st.markdown("**High probability** - Strong evidence for latent TB")
                        elif prob > 0.4:
                            st.markdown("**Moderate probability** - Some evidence for latent TB")
                        else:
                            st.markdown("**Low probability** - Limited evidence for latent TB")
                    else:  # none
                        if prob > 0.7:
                            st.markdown("**High probability** - Strong evidence for no TB")
                        elif prob > 0.4:
                            st.markdown("**Moderate probability** - Some evidence for no TB")
                        else:
                            st.markdown("**Low probability** - Uncertain classification")
                    
                    st.markdown("---")
                
                # Probability distribution analysis
                max_prob = max(cls_probs)
                max_class_idx = np.argmax(cls_probs)
                max_class_name = prob_names[max_class_idx]
                
                st.markdown(f"""
                **Analysis Summary**:
                - **Highest Probability**: {max_class_name} ({max_prob:.2%})
                - **Prediction Certainty**: {"High" if max_prob > 0.8 else "Moderate" if max_prob > 0.6 else "Low"}
                - **Class Separation**: {"Clear" if max_prob > 0.6 and (max_prob - min(cls_probs)) > 0.3 else "Uncertain"}
                """)
        
        with tab4:
            st.markdown("### Understanding the Results")
            
            st.markdown("""
            #### How to Interpret These Results
            
            **1. Global Classification vs. Local Detections**
            - **Global Classification**: Analyzes the entire image to determine overall TB status
            - **Local Detections**: Identifies specific lesion locations with bounding boxes
            - Both provide complementary information for comprehensive assessment
            
            **2. Confidence Scores**
            - **Prediction Confidence**: How certain the model is about the overall classification
            - **Detection Confidence**: How certain the model is about each individual lesion
            - Higher confidence (>70%) generally indicates more reliable predictions
            
            **3. Severity Score**
            - Estimates disease extent based on lesion area and distribution
            - Higher scores indicate more extensive involvement
            - Useful for treatment planning and monitoring disease progression
            
            **4. Probability Distribution**
            - Shows the model's confidence across all possible classifications
            - Large differences between probabilities indicate clear classification
            - Similar probabilities suggest uncertainty or ambiguous features
            """)
            
            st.markdown("""
            #### Model Capabilities & Limitations
            
            **What This Model Does Well**:
            - ✅ Detects visible TB lesions in chest X-rays
            - ✅ Classifies TB type (none, latent, active)
            - ✅ Estimates disease severity
            - ✅ Provides localization of lesions
            
            **Important Limitations**:
            - ⚠️ **Not a replacement for clinical diagnosis**: Always requires medical professional interpretation
            - ⚠️ **May miss early-stage disease**: Very early TB may not show visible radiographic changes
            - ⚠️ **False positives/negatives possible**: Model accuracy depends on image quality and disease stage
            - ⚠️ **No clinical context**: Model doesn't consider patient history, symptoms, or lab results
            - ⚠️ **Training data limitations**: Performance may vary with different patient populations
            """)
            
            st.markdown("""
            #### Next Steps & Recommendations
            
            **If Active TB is Detected**:
            1. **Immediate medical consultation** is essential
            2. Confirmatory testing (sputum culture, PCR) should be performed
            3. Contact tracing may be necessary
            4. Treatment should begin promptly under medical supervision
            
            **If Latent TB is Detected**:
            1. Consult with healthcare provider for evaluation
            2. Consider preventive treatment to prevent progression
            3. Regular monitoring may be recommended
            4. Assess risk factors and immune status
            
            **If No TB is Detected**:
            1. Results should be interpreted in clinical context
            2. Consider patient symptoms and risk factors
            3. Follow-up may be recommended if clinical suspicion remains
            4. Regular screening may be appropriate for high-risk individuals
            """)
            
            st.markdown("""
            #### Educational Note
            
            This tool is designed for **educational and research purposes only**. It demonstrates the 
            capabilities of deep learning in medical imaging but should **never be used as the sole basis 
            for clinical decisions**. All results must be interpreted by qualified healthcare professionals 
            in conjunction with:
            - Patient history and symptoms
            - Physical examination
            - Laboratory tests
            - Other diagnostic modalities
            - Clinical judgment
            """)
        
        # Final disclaimer
        st.divider()
        st.markdown("""
        <div style="background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 1rem; border-radius: 4px; margin-top: 1rem;">
        <strong>⚠️ Medical Disclaimer</strong><br>
        This application is for educational and demonstration purposes only. It is not intended for clinical use, 
        diagnosis, or treatment decisions. All results must be interpreted by qualified medical professionals. 
        The developers assume no responsibility for any medical decisions made based on this tool.
        </div>
        """, unsafe_allow_html=True)
        
        # PDF Download Button - Centered
        st.markdown("---")
        # Center the button using columns with empty space on sides
        col_left, col_center, col_right = st.columns([1, 2, 1])
        with col_center:
            if st.button("📄 Download Clinical PDF", use_container_width=True, type="primary"):
                try:
                    pdf_bytes = generate_clinical_pdf(result)
                    st.download_button(
                        label="⬇️ Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"medpie_report_{st.session_state.model.get_classification(result)[0]}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"PDF generation error: {e}")
        
        # Reset Button
        if st.button("🔄 Analyze New Image", use_container_width=True):
            st.session_state.inference_result = None
            st.session_state.uploaded_image = None
            st.rerun()


if __name__ == "__main__":
    main()

