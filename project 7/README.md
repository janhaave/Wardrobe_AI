# Wardrobe.AI 👔

An AI-powered virtual wardrobe application that suggests T-shirt combinations based on your bottomwear using computer vision and machine learning. The application uses a sophisticated two-page workflow that combines computer vision, AI analysis, and augmented reality to provide personalized fashion recommendations.

## 🌟 Features

- **AI-Powered Recommendations**: Upload a photo of your pants/bottomwear and get intelligent T-shirt suggestions
- **Augmented Reality Try-On**: Real-time AR overlay using your webcam to virtually try on suggested outfits
- **Smart Image Analysis**: Uses OpenAI's vision models to analyze clothing styles, colors, and patterns
- **Vector Database Search**: Fast similarity search through a curated clothing database
- **Interactive Web Interface**: User-friendly Streamlit application with elegant design

## 📱 Application Workflow (app.py)

The main application (`src/app.py`) implements a sophisticated two-page workflow:

### **Page 1: Image Upload & Analysis** 🖼️

**User Interface:**
- Elegant background with custom Didot font styling
- Centered file upload component accepting JPG, PNG, JPEG formats
- Real-time image preview upon upload
- Professional styling with custom CSS for backgrounds

**Backend Processing:**
- **Image Validation**: Automatic format checking and file validation
- **Secure Storage**: Uploaded images saved to `../data/pant_samples/current_pant.png`
- **Session Management**: Streamlit session state tracks user progress
- **Navigation Control**: Seamless transition to analysis page

**Technical Implementation:**
```python
# File upload with validation
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

# Secure file saving
file_path = os.path.join(SAVE_FOLDER, "current_pant.png")
with open(file_path, "wb") as f:
    f.write(uploaded_file.getbuffer())
```

### **Page 2: AI Analysis & AR Try-On** 🤖

**AI-Powered Analysis Pipeline:**

1. **Image Encoding & Processing**
   - Converts uploaded image to base64 for AI processing
   - Uses OpenAI's GPT-4o-mini vision model for image analysis
   - Specialized prompts for clothing description and style analysis

2. **Intelligent Clothing Description**
   ```python
   def image_summarize(img_base64, prompt, model_name):
       # Sends image to OpenAI Vision API
       # Returns detailed clothing description optimized for retrieval
   ```
   - **Purpose**: Analyzes uploaded pants for style, color, fit, material
   - **Output**: Concise, retrieval-optimized description
   - **Example**: "Dark blue slim-fit denim jeans with modern cut"

3. **Smart T-shirt Recommendations**
   ```python
   def tshirt_suggestion(summary, prompt2, model_name):
       # Combines pant analysis with fashion expertise
       # Generates creative, diverse t-shirt suggestions
   ```
   - **AI Fashion Stylist**: Acts as a professional fashion consultant
   - **Diversity Focus**: Avoids repetitive suggestions (no plain white tees)
   - **Style Range**: Covers casual, formal, vintage, athletic, graphic designs
   - **Features Considered**: Necklines, sleeves, patterns, colors, textures

**Vector Database Integration:**

4. **Similarity Search Engine**
   ```python
   # Dual retriever system for pants and t-shirts
   retriever2 = get_retriever_pant(embeddings_model, file_store)  # Pant matching
   retriever = get_retriever(embeddings_model, file_store)        # T-shirt matching
   ```
   - **ChromaDB Collections**: Separate collections for pants and t-shirts
   - **OpenAI Embeddings**: Uses `text-embedding-3-small` for semantic search
   - **MultiVector Architecture**: Combines text descriptions with image storage

5. **Image Retrieval & Processing**
   ```python
   # Find similar items in vector database
   result_pant = retriever2.vectorstore.similarity_search(summary_pant)[0]
   result = retriever.vectorstore.similarity_search(suggestion)[0]
   
   # Retrieve actual images from document store
   image_base64_pant = get_image_from_docstore(file_store, doc_id_result_pant)
   image_base64 = get_image_from_docstore(file_store, doc_id_result)
   ```

**Augmented Reality Implementation:**

6. **Real-Time AR Overlay System**
   ```python
   def outfit_overlay_application(webcam_placeholder, shirt_image_path, pant_image_path):
       # MediaPipe pose detection + OpenCV image processing
   ```

   **Pose Detection Technology:**
   - **MediaPipe Integration**: Real-time human pose estimation
   - **Body Landmark Detection**: Identifies shoulders, hips, body proportions
   - **Automatic Scaling**: Dynamically adjusts clothing size based on body measurements

   **Advanced Overlay Algorithm:**
   - **Precise Positioning**: 
     - T-shirts: Positioned using shoulder landmarks with 2.0x width multiplier
     - Pants: Positioned using hip landmarks with 2.5x width multiplier
   - **Intelligent Sizing**: 
     - Shirt height = width × 1.5 ratio
     - Pant height = width × 1.8 ratio
   - **Error Handling**: Graceful degradation if pose detection fails

   **Image Processing Pipeline:**
   ```python
   def overlay_image(background, overlay, x, y, w, h):
       # Advanced alpha blending for realistic overlay
       # Handles transparency, resizing, boundary checking
       # Prevents crashes with dimension validation
   ```

**Real-Time User Interface:**

7. **Dynamic Status Updates**
   - **Progress Indicators**: Shows AI analysis progress
   - **Real-time Feedback**: 
     - "I see that you have uploaded: [AI description]"
     - "I personally suggest that you should wear: [AI recommendation]"
     - "Let me find it in my Vector Store for you..."

8. **Webcam Integration**
   - **Automatic Activation**: Webcam starts immediately after processing
   - **Live Overlay**: Real-time clothing overlay on video feed
   - **Performance Optimization**: 50ms delay for smooth frame rate
   - **Error Recovery**: Handles webcam access issues gracefully

## 🏗️ Project Structure

```
wardrobe-ai/
│
├── src/
│   └── app.py                    # Main Streamlit application
│
├── data/
│   ├── pant_samples/             # Sample pant images
│   ├── tshirt_samples/           # Sample t-shirt images
│   └── README.md                 # Data management guide
│
├── notebooks/
│   └── prepare_vector_db.ipynb   # Vector database setup
│
├── results/                      # Generated outfit combinations
│
├── requirements.txt              # Python dependencies
├── .gitignore                   # Git ignore rules
├── .env.example                 # Environment variables template
└── README.md                    # This file
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8+
- OpenAI API key
- Webcam (for AR features)

### 2. Installation

```bash
# Clone the repository
git clone <repository-url>
cd project-7

# Create virtual environment
python -m venv myenv
myenv\Scripts\activate  # On Windows
# source myenv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Setup

```bash
# Copy environment template
copy .env.example .env  # On Windows
# cp .env.example .env  # On macOS/Linux

# Edit .env file and add your OpenAI API key
# OPENAI_API_KEY=your_api_key_here
```

### 4. Prepare Vector Database

```bash
# Run the setup notebook to prepare the vector database
jupyter notebook notebooks/prepare_vector_db.ipynb
```

Or run all cells in the notebook to:
- Process clothing images
- Generate AI descriptions
- Create vector embeddings
- Populate ChromaDB

### 5. Run the Application

```bash
# Navigate to src directory
cd src

# Run Streamlit app
streamlit run app.py
```

### **AI Prompt Engineering**

The application uses sophisticated prompt engineering for optimal AI performance:

#### **Clothing Analysis Prompt (for Pants)**
```
"You are an assistant tasked with summarizing pant images for retrieval.
These summaries will be embedded and used to retrieve the raw image.
Give a concise summary of the pants that is well optimized for retrieval in a single line.
Focus on style, color, fit, material, and any distinctive features.
Do not talk about anything else just the pants you see in the image."
```

#### **T-shirt Recommendation Prompt**
```
"You are a fashion stylist suggesting diverse and creative T-shirt designs for men.
Ensure each suggestion is unique, avoiding repetition of common choices like plain white tees.
Explore a wide range of styles, band or cartoon graphics, color-blocking, retro patterns, and athleisure designs.
Include distinctive necklines (crew, V-neck, turtleneck, boat neck), sleeve types (short, long, sleeveless),
and features like button accents, embroidery, or pockets. Suggest a single suggestion of T-shirt design in one concise sentence (8-10 words),
highlighting key style, fit, and design features. Focus on creative, fresh ideas that stand out"
```

### **Application Configuration**

**Model Selection:**
- **Vision Model**: `gpt-4o-mini` - Optimized for image analysis with cost efficiency
- **Embedding Model**: `text-embedding-3-small` - Fast retrieval with 1536 dimensions
- **Pose Model**: MediaPipe Pose with 0.5 confidence thresholds

**File Handling:**
- **Supported Formats**: JPG, PNG, JPEG with automatic validation
- **Storage Strategy**: Temporary storage in `data/pant_samples/current_pant.png`
- **Output Management**: Results saved to `../results/` directory

**Vector Database Configuration:**
- **ChromaDB Collections**: `"tshirt"` and `"pant"` with separate embeddings
- **Persistence**: Local storage in `../chroma_langchain_db/`
- **Document Store**: LocalFileStore in `../TSHIRT_DOCSTORE/`

## 📱 How to Use the Application

### **Step-by-Step User Journey**

1. **Launch Application**
   ```bash
   cd src
   streamlit run app.py
   ```
   - Application opens in browser with elegant landing page
   - Custom background and professional typography loaded

2. **Upload Bottomwear Image**
   - Click on upload area or drag-and-drop image file
   - Supported formats: JPG, PNG, JPEG
   - Real-time preview appears immediately
   - File validation happens automatically

3. **AI Analysis Process** (Automatic)
   - Image converted to base64 for AI processing
   - OpenAI Vision API analyzes clothing style, color, fit
   - AI generates optimized description for vector search
   - Progress shown with descriptive messages

4. **T-shirt Recommendation** (Automatic)
   - AI fashion stylist analyzes your bottomwear
   - Generates creative, unique t-shirt suggestions
   - Avoids common/boring recommendations
   - Considers style harmony and color coordination

5. **Vector Database Search** (Automatic)
   - Embedding generated from AI descriptions
   - Similarity search through curated clothing database
   - Best matching t-shirt retrieved from collection
   - Original images reconstructed from base64 storage

6. **Augmented Reality Try-On**
   - Webcam automatically activates
   - MediaPipe detects your body pose in real-time
   - T-shirt and pant overlays positioned accurately
   - Live preview of suggested outfit combination
   - Responsive scaling based on your body measurements

### **User Interface Features**

- **Responsive Design**: Adapts to desktop, tablet, mobile screens
- **Professional Styling**: Custom fonts, backgrounds, color schemes
- **Real-time Feedback**: Progress indicators and status messages
- **Error Handling**: Graceful degradation with helpful error messages
- **Performance Monitoring**: Smooth 20 FPS webcam feed

## 🔧 Technical Implementation Details

### **Core Application Architecture (app.py)**

The application is built on a modular architecture with clear separation of concerns:

#### **1. AI Vision Processing Module**

**`image_summarize(img_base64, prompt, model_name)`**
- **Purpose**: Converts uploaded clothing images into detailed, searchable descriptions
- **Technology**: OpenAI GPT-4o-mini vision model
- **Input**: Base64 encoded image + specialized clothing analysis prompt
- **Output**: Optimized description for vector similarity search
- **Prompt Engineering**: Uses carefully crafted prompts for maximum retrieval accuracy

**`tshirt_suggestion(summary, prompt2, model_name)`**
- **Purpose**: Acts as an AI fashion stylist to recommend complementary t-shirts
- **Intelligence**: Considers style harmony, color coordination, occasion appropriateness
- **Creativity Engine**: Generates diverse suggestions avoiding common defaults
- **Output**: Specific t-shirt recommendations (8-10 words, highly descriptive)

#### **2. Vector Database Management**

**`get_retriever(embeddings_model, filestore)` & `get_retriever_pant()`**
- **Dual Collection System**: Separate ChromaDB collections for t-shirts and pants
- **Embedding Model**: OpenAI text-embedding-3-small for semantic similarity
- **MultiVector Architecture**: Combines text embeddings with image storage
- **Persistent Storage**: ChromaDB with local file persistence

**`get_image_from_docstore(docstore, doc_id)`**
- **Purpose**: Retrieves original images from document store using document IDs
- **Efficiency**: Direct lookup by vector search results
- **Format**: Returns base64 encoded images for processing

#### **3. Image Processing Pipeline**

**`base64_to_image(base64_string, output_path)`**
- **Format Detection**: Automatically detects image format from base64 header
- **Multi-format Support**: PNG, JPG, GIF, BMP, WebP
- **File Management**: Saves processed images to results directory
- **Error Handling**: Comprehensive format validation and error recovery

#### **4. Augmented Reality Engine**

**`outfit_overlay_application(webcam_placeholder, shirt_image_path, pant_image_path)`**

**MediaPipe Pose Integration:**
```python
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # Real-time pose landmark detection
    # 33 body landmarks for precise positioning
```

**Advanced Overlay Mathematics:**
- **Shoulder Width Calculation**: `np.linalg.norm([left_x - right_x, left_y - right_y])`
- **T-shirt Positioning**: 
  - Width: `shoulder_width × 2.0`
  - Height: `shirt_width × 1.5`
  - X-offset: `(left_x + right_x) / 2 - shirt_width / 2`
  - Y-offset: `min(left_y, right_y) - 60px`

- **Pant Positioning**:
  - Width: `shoulder_width × 2.5`
  - Height: `pant_width × 1.8`
  - Hip-centered positioning with automatic scaling

**Alpha Blending Algorithm:**
```python
alpha_channel = overlay_cropped[:, :, 3] / 255.0
alpha_rgb = np.stack([alpha_channel, alpha_channel, alpha_channel], axis=-1)
background[y1:y2, x1:x2] = (roi * (1 - alpha_rgb) + overlay_cropped[:, :, :3] * alpha_rgb)
```

#### **5. Session Management & Navigation**

**Streamlit State Management:**
- **Page Navigation**: `st.session_state.page` for multi-page workflow
- **Data Persistence**: Maintains uploaded images and AI analysis across pages
- **Process Tracking**: Tracks analysis progress and user interactions

**Error Handling & Recovery:**
- **Webcam Failures**: Graceful degradation with user feedback
- **AI API Errors**: Retry mechanisms and error messages
- **File Processing**: Comprehensive validation and error recovery
- **Memory Management**: Efficient handling of large image data

#### **6. User Interface Design**

**Custom Styling System:**
```python
st.markdown(f"""
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
""", unsafe_allow_html=True)
```

**Typography & Branding:**
- **Google Fonts Integration**: Didot serif for elegant branding
- **Responsive Design**: Adapts to different screen sizes
- **Professional Color Scheme**: Strategic use of white text on themed backgrounds

### **Data Flow Architecture**

```
[User Upload] → [Base64 Encoding] → [OpenAI Vision API] → [Description Generation]
       ↓                                                            ↓
[File Storage] ← [Image Processing] ← [Vector Search] ← [Embedding Creation]
       ↓                                     ↓
[AR Processing] ← [Image Retrieval] ← [Similarity Matching]
       ↓
[Real-time Overlay] → [Webcam Feed] → [User Experience]
```

### **Performance Optimizations**

1. **Efficient Image Processing**: Base64 encoding for API compatibility
2. **Vector Search Speed**: ChromaDB with optimized embedding dimensions
3. **Real-time AR**: 50ms frame delays for smooth video processing
4. **Memory Management**: Lazy loading of images and models
5. **Error Recovery**: Non-blocking error handling for robust user experience

## 📊 Adding Your Own Data

### Adding Clothing Items

1. Add images to `data/tshirt_samples/` or `data/pant_samples/`
2. Run the preparation notebook to update the vector database
3. Supported formats: PNG, JPG, JPEG
4. Recommended: 512x512 resolution, transparent backgrounds for best AR results

### Database Management

The vector database stores:
- Base64 encoded images
- AI-generated descriptions
- Vector embeddings for similarity search
- Metadata (clothing type, style attributes)

## 🛠️ Development

### Project Architecture

```
User Upload → AI Analysis → Vector Search → Recommendation → AR Overlay
     ↓              ↓             ↓              ↓            ↓
  Image Input → Description → Embedding → Match Finding → Real-time Try-on
```

### **Core Application Functions (app.py)**

#### **AI & Machine Learning Functions**

**`image_summarize(img_base64, prompt, model_name)`**
- **Input**: Base64 image, analysis prompt, AI model name
- **Process**: Sends image to OpenAI Vision API with specialized clothing prompts
- **Output**: Detailed clothing description optimized for vector search
- **Usage**: Analyzes uploaded pants for style, color, fit, material characteristics

**`tshirt_suggestion(summary, prompt2, model_name)`** 
- **Input**: Pant description summary, recommendation prompt, AI model
- **Process**: Combines fashion expertise with pant analysis to suggest complementary t-shirts
- **Intelligence**: Considers color theory, style harmony, occasion appropriateness
- **Output**: Creative t-shirt recommendation (8-10 words, highly descriptive)

#### **Vector Database Functions**

**`get_retriever(embeddings_model, filestore)`**
- **Purpose**: Initializes t-shirt vector database retriever
- **Components**: ChromaDB vectorstore + OpenAI embeddings + LocalFileStore
- **Configuration**: Uses "tshirt" collection with text-embedding-3-small model
- **Return**: MultiVectorRetriever for t-shirt similarity search

**`get_retriever_pant(embeddings_model, filestore)`**
- **Purpose**: Initializes pant vector database retriever  
- **Components**: Separate ChromaDB collection for pant matching
- **Configuration**: Uses "pant" collection with identical embedding setup
- **Return**: MultiVectorRetriever for pant similarity search

**`get_image_from_docstore(docstore, doc_id)`**
- **Purpose**: Retrieves original clothing images from document storage
- **Process**: Uses document ID from vector search to fetch base64 image
- **Efficiency**: Direct lookup without re-embedding or re-search
- **Return**: Base64 encoded image ready for processing

#### **Image Processing Functions**

**`encode_to_base64(image_path)`**
- **Purpose**: Converts local image files to base64 strings
- **Usage**: Prepares images for AI API calls and storage
- **Error Handling**: File validation and encoding error management
- **Return**: UTF-8 decoded base64 string

**`base64_to_image(base64_string, output_path)`**
- **Purpose**: Converts base64 strings back to image files
- **Intelligence**: Auto-detects image format (PNG, JPG, GIF, BMP, WebP)
- **Process**: Parses header, decodes data, saves with correct extension
- **Usage**: Saves retrieved clothing images for AR overlay

#### **Augmented Reality Functions**

**`outfit_overlay_application(webcam_placeholder, shirt_image_path, pant_image_path)`**
- **Purpose**: Real-time AR clothing overlay on webcam feed
- **Technology**: MediaPipe pose detection + OpenCV image processing
- **Features**: Dynamic scaling, alpha blending, error recovery
- **Performance**: Optimized for smooth real-time processing

**`overlay_image(background, overlay, x, y, w, h)` (Internal)**
- **Purpose**: Advanced image blending for realistic clothing overlay
- **Algorithm**: Alpha channel compositing with transparency support
- **Validation**: Comprehensive dimension and boundary checking
- **Quality**: Anti-aliased resizing with INTER_AREA interpolation

#### **Utility Functions**

**`change_page(page_number)`**
- **Purpose**: Manages multi-page application navigation
- **State Management**: Updates Streamlit session state
- **Usage**: Transitions between upload page and AR try-on page

**`get_image_as_base64(image_file)` (Local)**
- **Purpose**: Helper function for loading background images
- **Usage**: Loads custom backgrounds for application styling
- **Return**: Base64 string for CSS background-image property

## 🔒 Environment Variables

```bash
OPENAI_API_KEY=your_openai_api_key_here     # Required
OPENAI_EMBEDDINGS_MODEL=text-embedding-3-small  # Optional
OPENAI_VISION_MODEL=gpt-4o-mini            # Optional
```

## 🐛 Troubleshooting

### Common Issues

1. **OpenAI API Key Error**
   - Ensure your API key is correctly set in `.env`
   - Check your OpenAI account has sufficient credits

2. **Webcam Not Working**
   - Check camera permissions in your browser
   - Ensure no other applications are using the camera

3. **Vector Database Issues**
   - Re-run the preparation notebook
   - Check if ChromaDB files are corrupted

4. **Image Processing Errors**
   - Ensure images are in supported formats
   - Check image file sizes (very large images may cause issues)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for vision and embedding models
- MediaPipe for pose detection
- ChromaDB for vector database
- Streamlit for the web framework

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review existing GitHub issues
3. Create a new issue with detailed information

---

**Note**: This application requires an internet connection for AI features and sufficient system resources for real-time video processing.
