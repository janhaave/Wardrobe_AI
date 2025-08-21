import streamlit as st
import cv2, os
from PIL import Image
import numpy as np
import mediapipe as mp
import time
import base64
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.storage import LocalFileStore

# Set Streamlit page configuration
st.set_page_config(page_title="Wardrobe.AI", layout="wide")

# Initialize session state for navigation
if "page" not in st.session_state:
    st.session_state.page = 1

# Function to change pages
def change_page(page_number):
    st.session_state.page = page_number

def encode_to_base64(image_path):
    """
    Encode the image to a Base64 string.
    """
    with open(image_path, "rb") as img_file:
        base64_string = base64.b64encode(img_file.read()).decode("utf-8")
    return base64_string

def image_summarize(img_base64, prompt, model_name):
    chat = ChatOpenAI(model=model_name, api_key=os.getenv("OPENAI_API_KEY"))
    msg = chat.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpg/jpeg;base64,{img_base64}"},
                    },
                ]
            )
        ]
    )
    print(msg.content)
    return msg.content

def tshirt_suggestion(summary, prompt2, model_name):
    chat = ChatOpenAI(model=model_name, api_key=os.getenv("OPENAI_API_KEY"))
    
    # Combining prompt and summary text since both are text messages
    combined_message = f"{prompt2}\n{summary}"
    
    msg = chat.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": combined_message}
                ]
            )
        ]
    )
    print(msg.content)
    return msg.content

def get_retriever(embeddings_model, filestore):
    embeddings = OpenAIEmbeddings(model=embeddings_model)
    vectorstore = Chroma(
        collection_name="tshirt",
        embedding_function=embeddings,
        persist_directory="../chroma_langchain_db",  # Updated path
    )
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore, docstore=filestore, id_key="doc_id", return_doc_ids=True
    )
    return retriever

def get_retriever_pant(embeddings_model, filestore):
    embeddings = OpenAIEmbeddings(model=embeddings_model)
    vectorstore = Chroma(
        collection_name="pant",
        embedding_function=embeddings,
        persist_directory="../chroma_langchain_db",  # Updated path
    )
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore, docstore=filestore, id_key="doc_id", return_doc_ids=True
    )
    return retriever

def get_image_from_docstore(docstore, doc_id):
    image_bytes = docstore.mget([doc_id])
    image_base64 = image_bytes[0].decode("utf-8")
    return image_base64

def base64_to_image(base64_string, output_path):
    """
    Converts a base64 string to an image file and saves it.
    
    Args:
        base64_string (str): The base64-encoded string of the image.
        output_path (str): The path to save the decoded image, without extension.
        
    Returns:
        str: The full path of the saved image.
    """
    # Split the base64 string to separate metadata and actual data
    if ',' in base64_string:
        header, base64_data = base64_string.split(',', 1)
    else:
        # Default to PNG if no header is provided
        header = "data:image/png;base64"
        base64_data = base64_string

    # Decode the base64 string
    image_data = base64.b64decode(base64_data)

    # Detect the image format from the header
    if 'jpeg' in header.lower():
        extension = 'jpg'
    elif 'png' in header.lower():
        extension = 'png'
    elif 'gif' in header.lower():
        extension = 'gif'
    elif 'bmp' in header.lower():
        extension = 'bmp'
    elif 'webp' in header.lower():
        extension = 'webp'
    else:
        raise ValueError("Unsupported or unknown image format in base64 string")

    # Full output path with the detected extension
    full_output_path = f"{output_path}.{extension}"

    # Save the decoded image data to the file
    with open(full_output_path, 'wb') as image_file:
        image_file.write(image_data)
    
    return full_output_path

def outfit_overlay_application(webcam_placeholder, shirt_image_path, pant_image_path):
    """
    Function to perform real-time full outfit overlay using MediaPipe Pose and OpenCV,
    with Streamlit webcam placeholder display.

    Parameters:
        webcam_placeholder: Streamlit placeholder to display webcam feed.
        shirt_image_path (str): Path to the shirt image (with transparency).
        pant_image_path (str): Path to the pant image (with transparency).
    """
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose

    # Load shirt and pant images
    shirt_img = None
    if shirt_image_path and os.path.exists(shirt_image_path):
        shirt_img = cv2.imread(shirt_image_path, cv2.IMREAD_UNCHANGED)
        if shirt_img is not None and shirt_img.shape[2] == 3:
            shirt_img = cv2.cvtColor(shirt_img, cv2.COLOR_BGR2BGRA)
    
    pant_img = None
    if pant_image_path and os.path.exists(pant_image_path):
        pant_img = cv2.imread(pant_image_path, cv2.IMREAD_UNCHANGED)
        if pant_img is not None and pant_img.shape[2] == 3:
            pant_img = cv2.cvtColor(pant_img, cv2.COLOR_BGR2BGRA)

    if shirt_img is None and pant_img is None:
        webcam_placeholder.warning("Could not load shirt or pant images. Overlay will not work.")
        return
    elif shirt_img is None:
        webcam_placeholder.warning("Could not load shirt image. Only pant will be overlaid.")
    elif pant_img is None:
        webcam_placeholder.warning("Could not load pant image. Only shirt will be overlaid.")

    # Overlay function
    def overlay_image(background, overlay, x, y, w, h):
        if overlay is None:
            return

        # Ensure width and height are positive
        if w <= 0 or h <= 0:
            return

        try:
            overlay_resized = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_AREA)
        except cv2.error as e:
            st.warning(f"Error resizing overlay image: {e}")
            return

        # Calculate bounding box coordinates for overlay
        y1, y2 = max(0, y), min(background.shape[0], y + overlay_resized.shape[0])
        x1, x2 = max(0, x), min(background.shape[1], x + overlay_resized.shape[1])

        # Calculate the corresponding region in the overlay image
        overlay_y1 = max(0, -y)
        overlay_y2 = overlay_y1 + (y2 - y1)
        overlay_x1 = max(0, -x)
        overlay_x2 = overlay_x1 + (x2 - x1)

        # Ensure dimensions are valid for slicing
        if y2 - y1 <= 0 or x2 - x1 <= 0 or \
           overlay_y2 - overlay_y1 <= 0 or overlay_x2 - overlay_x1 <= 0:
            return

        roi = background[y1:y2, x1:x2]
        overlay_cropped = overlay_resized[overlay_y1:overlay_y2, overlay_x1:overlay_x2]

        # Ensure roi and overlay_cropped have the same dimensions before blending
        if roi.shape[:2] != overlay_cropped.shape[:2]:
            st.warning(f"ROI and overlay_cropped dimensions mismatch: {roi.shape[:2]} vs {overlay_cropped.shape[:2]}")
            return

        alpha_channel = overlay_cropped[:, :, 3] / 255.0
        alpha_rgb = np.stack([alpha_channel, alpha_channel, alpha_channel], axis=-1)

        background[y1:y2, x1:x2] = (roi * (1 - alpha_rgb) + overlay_cropped[:, :, :3] * alpha_rgb).astype(np.uint8)

    # Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        webcam_placeholder.error("Could not open webcam. Please ensure it's connected and not in use.")
        st.session_state.webcam_active = False
        return

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while st.session_state.webcam_active:
            ret, frame = cap.read()
            if not ret:
                webcam_placeholder.warning("Failed to grab frame from webcam. Retrying...")
                time.sleep(0.1)
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Shirt coordinates
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                left_x = int(left_shoulder.x * frame.shape[1])
                left_y = int(left_shoulder.y * frame.shape[0])
                right_x = int(right_shoulder.x * frame.shape[1])
                right_y = int(right_shoulder.y * frame.shape[0])

                shoulder_width = int(np.linalg.norm([left_x - right_x, left_y - right_y]))
                
                if shoulder_width > 0:
                    shirt_width = int(shoulder_width * 2.0)
                    shirt_height = int(shirt_width * 1.5)

                    shirt_x = int((left_x + right_x) / 2 - shirt_width / 2)
                    shirt_y = int(min(left_y, right_y)) - 60

                    # Overlay shirt
                    if shirt_img is not None:
                        overlay_image(frame, shirt_img, shirt_x, shirt_y, shirt_width, shirt_height)

                # Pant coordinates
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
                hip_x = int((left_hip.x + right_hip.x) / 2 * frame.shape[1])
                hip_y = int((left_hip.y + right_hip.y) / 2 * frame.shape[0])

                if shoulder_width > 0:
                    pant_width = int(shoulder_width * 2.5)
                    pant_height = int(pant_width * 1.8)

                    pant_x = int(hip_x - pant_width / 2)
                    pant_y = hip_y - 10

                    # Overlay pant
                    if pant_img is not None:
                        overlay_image(frame, pant_img, pant_x, pant_y, pant_width, pant_height)

            # Show frame in Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            webcam_placeholder.image(frame_rgb, caption="Live Outfit Overlay", use_container_width=True)

            time.sleep(0.05)

    cap.release()

# Page 1: Upload Image
if st.session_state.page == 1:
    def get_image_as_base64(image_file):
        with open(image_file, "rb") as image:
            return base64.b64encode(image.read()).decode()

    # Get the image in base64 format - updated path
    background_image_path = "assets/bg.jpg"
    if os.path.exists(background_image_path):
        background_image = get_image_as_base64(background_image_path)

        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url('data:image/jpeg;base64,{background_image}');
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                height: 100vh;
                color: white;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <link href="https://fonts.googleapis.com/css2?family=Didot&display=swap" rel="stylesheet">
        <div style='text-align: center; font-size: 50px; font-weight: bold; font-family: "Didot", serif;'>
            Wardrobe.AI
        </div>
        """,
        unsafe_allow_html=True)
    
    st.markdown(
        """
        <link href="https://fonts.googleapis.com/css2?family=Didot&display=swap" rel="stylesheet">
        <div style='text-align: center; margin-top: 20px; font-size: 20px; font-family: "Didot", serif;'>
            Upload Image of your bottomwear
        </div>
        """,
        unsafe_allow_html=True)

    col1, col2, col3 = st.columns([0.2, 0.6, 0.2])
    with col2:
        uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], key="image_uploader")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Submit", key="submit_button"):
        if uploaded_file is not None:
            # Save the uploaded file with a fixed name - updated path
            SAVE_FOLDER = "../data/pant_samples"
            
            # Ensure the folder exists
            if not os.path.exists(SAVE_FOLDER):
                os.makedirs(SAVE_FOLDER)

            # Define the full path with the fixed name
            file_path = os.path.join(SAVE_FOLDER, "current_pant.png")

            # Save the file
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.success(f"File saved successfully at: {file_path}")

            # Proceed to the next page
            st.session_state.uploaded_image = uploaded_file
            change_page(2)
        else:
            st.warning("Please upload an image of pant before submitting.")

# Page 2: Webcam Feed and AI Processing
elif st.session_state.page == 2:
    def get_image_as_base64(image_file):
        with open(image_file, "rb") as image:
            return base64.b64encode(image.read()).decode()
    
    # Background image - updated path
    background_image_path = "assets/bg2.jpg"
    if os.path.exists(background_image_path):
        background_image = get_image_as_base64(background_image_path)

        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url('data:image/jpeg;base64,{background_image}');
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                height: 100vh;
                color: white;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

    # Initialize session state variables
    if "suggestion" not in st.session_state:
        st.session_state.suggestion = None
    if "summary_pant" not in st.session_state:
        st.session_state.summary_pant = None

    # Updated file paths
    file_path = "../data/pant_samples/current_pant.png"
    file_store = LocalFileStore("../TSHIRT_DOCSTORE")
    embeddings_model = "text-embedding-3-small"
    
    retriever2 = get_retriever_pant(embeddings_model, file_store)
    retriever = get_retriever(embeddings_model, file_store)
    
    prompt = """You are an assistant tasked with summarizing images for retrieval. 
    These summaries will be embedded and used to retrieve the raw image. 
    Give a concise summary of the image that is well optimized for retrieval in a single line.
    Do not talk about anything else just the pants you see in the image"""
    
    prompt2 = """You are a fashion stylist suggesting diverse and creative T-shirt designs for men. 
    Ensure each suggestion is unique, avoiding repetition of common choices like plain white tees. 
    Explore a wide range of styles, band or cartoon graphics, color-blocking, retro patterns, and athleisure designs. 
    Include distinctive necklines (crew, V-neck, turtleneck, boat neck), sleeve types (short, long, sleeveless), 
    and features like button accents, embroidery, or pockets. Suggest a single suggestion of T-shirt design in one concise sentence (8-10 words), 
    highlighting key style, fit, and design features. Focus on creative, fresh ideas that stand out"""
    
    # Process the uploaded image
    if os.path.exists(file_path):
        encoded_image = encode_to_base64(file_path)
        st.session_state.summary_pant = image_summarize(encoded_image, prompt, "gpt-4o-mini")
        
        st.markdown(
            f"""
            <div style='text-align: center; font-size: 24px; font-weight: bold; color: #3E2723;'>
                Summary of the Uploaded Image
            </div>
            <div style='text-align: center; font-size: 18px; margin-top: 10px; color: #3E2723;'>
                I see that you have uploaded: <b>{st.session_state.summary_pant}</b>
            </div>
            """,
            unsafe_allow_html=True)
        
        st.session_state.suggestion = tshirt_suggestion(st.session_state.summary_pant, prompt2, "gpt-4o-mini")
        
        st.markdown(
            f"""
            <div style='text-align: center; font-size: 24px; font-weight: bold; color: #3E2723;'>
                T-shirt Suggestion
            </div>
            <div style='text-align: center; font-size: 18px; margin-top: 10px; color: #3E2723;'>
                I personally suggest that you should wear: <b>{st.session_state.suggestion}</b><br>
                Let me find it in my Vector Store for you...
            </div>
            """,
            unsafe_allow_html=True)
        
        # Retrieve matching items from vector store
        result_pant = retriever2.vectorstore.similarity_search(st.session_state.summary_pant)[0]
        result = retriever.vectorstore.similarity_search(st.session_state.suggestion)[0]
        
        doc_id_result_pant = result_pant.metadata["doc_id"]
        doc_id_result = result.metadata["doc_id"]
        
        image_base64_pant = get_image_from_docstore(file_store, doc_id_result_pant)
        image_base64 = get_image_from_docstore(file_store, doc_id_result)
        
        # Updated output paths
        pant_path = base64_to_image(image_base64_pant, "../results/result_pant")
        tshirt_path = base64_to_image(image_base64, "../results/result_tshirt")
        
        if "webcam_active" not in st.session_state:
            st.session_state.webcam_active = True
            
        webcam_placeholder = st.empty()
        outfit_overlay_application(webcam_placeholder, tshirt_path, pant_path)
