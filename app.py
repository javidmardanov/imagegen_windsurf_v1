import streamlit as st
import requests
import io
from PIL import Image
from dotenv import load_dotenv
import os
import concurrent.futures
import time
import uuid
import base64

# Load environment variables
load_dotenv()

# Translations
translations = {
    'en': {
        'title': '🎨 Flux Image Generator',
        'subtitle': 'Generate amazing images using the Flux AI model!',
        'prompt_label': 'Enter your image description (in English):',
        'prompt_placeholder': 'Example: Professional young Azerbaijani news reporter in a modern TV studio, Canon C300, cinematic lighting',
        'prompt_help': 'Prompts must be in English! Be descriptive and include technical details for better results.',
        'example_prompts': '📝 Example Prompts',
        'example_list': """
        - Professional female Azerbaijani news anchor, modern TV studio, perfect lighting, 8K, photorealistic
        - Young male news reporter in formal attire, broadcasting studio background, cinematic composition, Blackmagic Studio 4K
        - News presenter at modern desk, blue and white studio design, professional lighting setup, Sony Venice camera
        - TV host in traditional Azerbaijani formal wear, contemporary news studio, dramatic lighting, 35mm lens
        - Professional journalist conducting interview, depth of field, studio lighting, Canon C500, 4K resolution
        """,
        'advanced_settings': '⚙️ Advanced Settings',
        'image_settings': 'Image Settings',
        'height': 'Image Height',
        'width': 'Image Width',
        'num_images': 'Number of Images',
        'guidance_scale': 'Guidance Scale',
        'guidance_help': 'Higher values make the image more closely match the prompt, but may reduce quality',
        'num_steps': 'Number of Steps',
        'steps_help': 'More steps generally means better quality but slower generation',
        'random_seed': 'Use random seed',
        'seed': 'Seed',
        'seed_help': 'Set a specific seed for reproducible results',
        'choose_model': '🚀 Model Selection',
        'high_quality': '🎨 High Quality',
        'high_quality_desc': 'FLUX.1-dev: Higher quality, better prompt following, ideal for detailed and artistic images. Takes longer to generate.',
        'fast_gen': '⚡ Standard',
        'fast_gen_desc': 'FLUX.1-schnell: Faster generation times, good for rapid prototyping and quick iterations.',
        'generate': '🎨 Generate Images',
        'select_model': 'Select a model to continue',
        'about_models': 'ℹ️ About the Models',
        'dev_title': 'FLUX.1-dev',
        'dev_features': """
        - 12 billion parameter model
        - Higher quality output
        - Better prompt following
        - Ideal for final images and detailed artwork
        - Longer generation time
        """,
        'schnell_title': 'FLUX.1-schnell',
        'schnell_features': """
        - Optimized for speed
        - Good for quick iterations and prototyping
        - Faster generation time
        - Suitable for testing prompts before using the high-quality model
        """,
        'enter_prompt': 'Please enter a prompt first!'
    },
    'az': {
        'title': '🎨 Şəkil Generatoru',
        'subtitle': 'Süni intellekt vasitəsilə şəkillər yaradın!',
        'prompt_label': 'Şəklin təsvirini daxil edin (İngilis dilində):',
        'prompt_placeholder': 'Example: Professional young Azerbaijani news reporter in a modern TV studio, Canon C300, cinematic lighting',
        'prompt_help': 'Təsvir mütləq İNGİLİS DİLİNDƏ olmalıdır! Daha keyfiyyətli nəticə üçün texniki detalları əlavə edin.',
        'example_prompts': '📝 Nümunə Təsvirlər',
        'example_list': """
        - Professional female Azerbaijani news anchor, modern TV studio, perfect lighting, 8K, photorealistic
        - Young male news reporter in formal attire, broadcasting studio background, cinematic composition, Arri Alexa
        - News presenter at modern desk, blue and white studio design, professional lighting setup, Sony Venice camera
        - TV host in traditional Azerbaijani formal wear, contemporary news studio, dramatic lighting, 35mm lens
        - Professional journalist conducting interview, depth of field, studio lighting, Canon C500, 4K resolution
        """,
        'advanced_settings': '⚙️ Əlavə Parametrlər',
        'image_settings': 'Şəkil Parametrləri',
        'height': 'Hündürlük',
        'width': 'En',
        'num_images': 'Şəkil sayı',
        'guidance_scale': 'Dəqiqlik əmsalı',
        'guidance_help': 'Yüksək əmsal şəkili təsvirə daha uyğun edir, lakin keyfiyyətə təsir edə bilər',
        'num_steps': 'Addım sayı',
        'steps_help': 'Addım sayı çox olduqca keyfiyyət artır, lakin proses daha çox vaxt aparır',
        'random_seed': 'Təsadüfi başlanğıc nöqtəsi',
        'seed': 'Başlanğıc nöqtəsi',
        'seed_help': 'Eyni nəticəni təkrar almaq üçün xüsusi başlanğıc nöqtəsi təyin edin',
        'choose_model': '🚀 Model seçimi',
        'high_quality': '🎨 Keyfiyyətli',
        'high_quality_desc': 'FLUX.1-dev: Yüksək keyfiyyətli və dəqiq nəticələr üçün. Detallı şəkillər yaratmaq üçün ideal seçimdir.',
        'fast_gen': '⚡ Sürətli',
        'fast_gen_desc': 'FLUX.1-schnell: Sürətli nəticələr üçün. İlkin yoxlama və test məqsədləri üçün əlverişlidir.',
        'generate': '🎨 Şəkil yarat',
        'select_model': 'Davam etmək üçün model seçin',
        'about_models': 'ℹ️ Modellər haqqında',
        'dev_title': 'FLUX.1-dev',
        'dev_features': """
        - 12 milyard parametrli model
        - Yüksək keyfiyyətli nəticə
        - Dəqiq təsvir uyğunluğu
        - Professional şəkillər üçün ideal
        - Nisbətən yavaş işləyir
        """,
        'schnell_title': 'FLUX.1-schnell',
        'schnell_features': """
        - Sürətli iş rejimi
        - Test və yoxlama üçün əlverişli
        - Qısa gözləmə müddəti
        - İlkin nəticələri görmək üçün ideal
        """,
        'enter_prompt': 'Zəhmət olmasa, təsvir daxil edin!'
    }
}

# Configure the page
st.set_page_config(
    page_title="ASAN AI Hub - Image Generator",
    page_icon="🎨",
    layout="centered"
)

# Add ASAN logo
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("asan_removebg.png", width=75, use_column_width=True)

# Language selector
lang = st.radio(
    "",
    ["English", "Azərbaycanca"],
    horizontal=True,
    key="language",
    label_visibility="collapsed"
)
lang = 'az' if lang == "Azərbaycanca" else 'en'

# Get translations for current language
t = translations[lang]

st.title(t['title'])
st.markdown(t['subtitle'])

# User input
prompt = st.text_area(t['prompt_label'], 
                     placeholder=t['prompt_placeholder'],
                     help=t['prompt_help'])

# Add an expander with example prompts
with st.expander(t['example_prompts']):
    st.markdown(t['example_list'])

# Parameters in expander
with st.expander(t['advanced_settings'], expanded=False):
    st.markdown(f"### {t['image_settings']}")
    col1, col2 = st.columns(2)

    with col1:
        # Image dimensions
        height = st.select_slider(
            t['height'],
            options=[512, 768, 1024],
            value=1024
        )
        
        width = st.select_slider(
            t['width'],
            options=[512, 768, 1024],
            value=1024
        )
        
        # Add batch size selector
        num_images = st.number_input(
            t['num_images'],
            min_value=1,
            max_value=4,
            value=1
        )

    with col2:
        # Model parameters
        guidance_scale = st.slider(
            t['guidance_scale'],
            min_value=1.0,
            max_value=20.0,
            value=3.5,
            step=0.5,
            help=t['guidance_help']
        )
        
        num_steps = st.slider(
            t['num_steps'],
            min_value=20,
            max_value=100,
            value=50,
            step=5,
            help=t['steps_help']
        )

        # Add seed control
        use_random_seed = st.checkbox(t['random_seed'], value=True)
        if not use_random_seed:
            seed = st.number_input(t['seed'], value=0, help=t['seed_help'])

# Function to load and encode the image
def get_image_base64(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

# Get API token from environment variable or let user input it
api_token = os.getenv("HUGGINGFACE_TOKEN")
if not api_token:
    api_token = st.text_input("Enter your HuggingFace API token:", type="password")
    if not api_token:
        st.warning("Please enter your HuggingFace API token to continue.")
        st.stop()

# API configuration
API_URLS = {
    "dev": "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev",
    "schnell": "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
}

def generate_single_image(args):
    """Generate a single image with progress tracking"""
    prompt, seed_value, image_num, total_images, model_type = args
    
    try:
        payload = {
            "inputs": prompt,
            "parameters": {
                "height": height,
                "width": width,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_steps,
            }
        }
        
        if seed_value is not None:
            payload["parameters"]["seed"] = seed_value
        
        # Make the API request
        headers = {"Authorization": f"Bearer {api_token}"}
        response = requests.post(API_URLS[model_type], headers=headers, json=payload)
        
        if response.status_code != 200:
            return None, f"Error: {response.status_code} - {response.text}"
        
        image = Image.open(io.BytesIO(response.content))
        return image, None
        
    except Exception as e:
        return None, f"Error: {str(e)}"

def generate_images(prompt, num_images, model_type):
    """Generate multiple images in parallel"""
    images = []
    errors = []
    
    # Create a progress bar
    progress_bar = st.progress(0, "Starting generation...")
    status_text = st.empty()
    
    # Create a list of seeds if using fixed seed
    seeds = None if use_random_seed else [seed + i for i in range(num_images)]
    
    # Prepare arguments for parallel processing
    args_list = [
        (prompt, seeds[i] if seeds else None, i, num_images, model_type)
        for i in range(num_images)
    ]
    
    # Use ThreadPoolExecutor for parallel processing
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(generate_single_image, args) for args in args_list]
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            image, error = future.result()
            if error:
                errors.append(error)
            if image:
                images.append(image)
            
            # Update progress safely from main thread
            completed += 1
            progress_bar.progress(completed / num_images)
            status_text.text(f"Generated image {completed} of {num_images}")
    
    # Clear the status text after completion
    status_text.empty()
    return images, errors

# Model selection section
st.markdown("---")
st.markdown(f"### {t['choose_model']}")

col1, col2 = st.columns(2)

with col1:
    dev_button = st.button(
        t['high_quality'],
        use_container_width=True,
        type="primary" if st.session_state.get('selected_model') == 'dev' else "secondary"
    )
    st.markdown(f"<small>{t['high_quality_desc']}</small>", unsafe_allow_html=True)

with col2:
    schnell_button = st.button(
        t['fast_gen'],
        use_container_width=True,
        type="primary" if st.session_state.get('selected_model') == 'schnell' else "secondary"
    )
    st.markdown(f"<small>{t['fast_gen_desc']}</small>", unsafe_allow_html=True)

# Handle model selection
if dev_button:
    st.session_state.selected_model = 'dev'
if schnell_button:
    st.session_state.selected_model = 'schnell'

# Show generate button if model is selected
if 'selected_model' in st.session_state:
    st.markdown("---")
    model_name = "FLUX.1-dev" if st.session_state.selected_model == 'dev' else "FLUX.1-schnell"
    st.markdown(f"### {t['choose_model']}: {model_name}")
    
    generate_button = st.button(t['generate'], type="primary", use_container_width=True)
    
    if generate_button:
        if not prompt:
            st.warning(t['enter_prompt'])
        else:
            images, errors = generate_images(prompt, num_images, st.session_state.selected_model)
            
            # Display any errors
            for error in errors:
                st.error(error)
            
            # Display successful generations
            if images:
                st.success(f"Successfully generated {len(images)} image(s) using {model_name}!")
                
                # Create columns for displaying images
                cols = st.columns(min(len(images), 2))
                for idx, image in enumerate(images):
                    col_idx = idx % 2
                    with cols[col_idx]:
                        st.image(image, caption=f"Image {idx + 1}", use_column_width=True)
                        
                        # Create a unique filename for each image
                        unique_id = uuid.uuid4().hex[:8]
                        img_buffer = io.BytesIO()
                        image.save(img_buffer, format="PNG")
                        st.download_button(
                            label=f"Download Image {idx + 1}",
                            data=img_buffer.getvalue(),
                            file_name=f"flux_{st.session_state.selected_model}_{unique_id}.png",
                            mime="image/png"
                        )
else:
    st.markdown(t['select_model'])

# Add footer with additional information
st.markdown("---")
with st.expander(t['about_models']):
    st.markdown(f"""
    ### {t['dev_title']}
    {t['dev_features']}
    
    ### {t['schnell_title']}
    {t['schnell_features']}
    """)
st.markdown("Javid Mardanov | javid.mardanov14@gmail.com")
