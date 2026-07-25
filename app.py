import os
import io
import cv2
import numpy as np
import psycopg2
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# Configuração da Página
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Segmentação e Análise Visual",
    page_icon="🔍",
    layout="wide"
)

# -----------------------------------------------------------------------------
# Inicialização de Modelos e Banco de Dados
# -----------------------------------------------------------------------------
@st.cache_resource
def load_segmentation_model():
    # Modelo leve para processamento em CPU (Render-friendly)
    return YOLO("yolov8n-seg.pt")

model = load_segmentation_model()

def get_db_connection():
    # Tenta ler do environment (Render/Local) ou Streamlit Secrets
    db_url = os.getenv("DATABASE_URL")
    if not db_url and "DATABASE_URL" in st.secrets:
        db_url = st.secrets["DATABASE_URL"]
        
    if not db_url:
        st.error("DATABASE_URL não configurada. Defina a variável de ambiente ou streamlit secrets.")
        return None
    
    try:
        conn = psycopg2.connect(db_url)
        return conn
    except Exception as e:
        st.error(f"Erro ao conectar ao Neon DB: {e}")
        return None

def save_to_neon(filename, dimensions, main_objects, object_count, segmented_area_pct):
    conn = get_db_connection()
    if conn is None:
        return False
    
    try:
        with conn.cursor() as cursor:
            query = """
                INSERT INTO image_analytics 
                (filename, dimensions, main_objects, object_count, segmented_area_pct)
                VALUES (%s, %s, %s, %s, %s)
            """
            cursor.execute(query, (filename, dimensions, main_objects, object_count, segmented_area_pct))
            conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Erro ao salvar no banco de dados: {e}")
        if conn:
            conn.close()
        return False

# -----------------------------------------------------------------------------
# Interface do Usuário (Streamlit)
# -----------------------------------------------------------------------------
st.title("🔍 Sistema de Segmentação & Análise Visual")
st.markdown("Upload de imagem, segmentação via IA e salvamento automático no **Neon.tech**.")

sidebar = st.sidebar
sidebar.header("Configurações")
conf_threshold = sidebar.slider("Confiança do Modelo", 0.1, 1.0, 0.35, 0.05)

uploaded_file = st.file_uploader("Selecione uma imagem...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Carregar imagem original
    image_bytes = uploaded_file.read()
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(pil_image)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Imagem Original")
        st.image(pil_image, use_container_width=True)
    
    # Botão de Processamento
    if st.button("🚀 Processar Imagem", type="primary"):
        with st.spinner("Processando segmentação e extraindo características..."):
            
            # Inferência do Modelo
            results = model.predict(source=img_np, conf=conf_threshold, save=False)
            result = results[0]
            
            # Renderizar a imagem segmentada
            res_plotted = result.plot()
            segmented_pil = Image.fromarray(res_plotted)
            
            with col2:
                st.subheader("Resultado da Segmentação")
                st.image(segmented_pil, use_container_width=True)
            
            # Extração de Métricas e Características
            height, width, channels = img_np.shape
            dim_str = f"{width}x{height} ({channels} ch)"
            
            detected_classes = []
            object_count = 0
            total_mask_area = 0
            
            if result.boxes is not None and len(result.boxes) > 0:
                object_count = len(result.boxes)
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                class_names = [model.names[cid] for cid in class_ids]
                detected_classes = list(set(class_names))
                
                # Cálculo da área segmentada (se houver máscaras)
                if result.masks is not None:
                    masks = result.masks.data.cpu().numpy()
                    combined_mask = np.any(masks, axis=0)
                    total_mask_area = (np.sum(combined_mask) / (height * width)) * 100
            
            main_objects_str = ", ".join(detected_classes) if detected_classes else "Nenhum objeto relevante"
            
            # Exibição dos Resultados
            st.markdown("---")
            st.subheader("📊 Diagnóstico e Características Extraídas")
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Dimensões", f"{width}x{height}px")
            m2.metric("Objetos Encontrados", object_count)
            m3.metric("Área Segmentada", f"{total_mask_area:.1f}%")
            m4.metric("Canais", channels)
            
            st.info(f"**Conteúdo Detectado na Imagem:** {main_objects_str}")
            
            # Persistência no PostgreSQL (Neon)
            saved = save_to_neon(
                filename=uploaded_file.name,
                dimensions=dim_str,
                main_objects=main_objects_str,
                object_count=object_count,
                segmented_area_pct=round(float(total_mask_area), 2)
            )
            
            if saved:
                st.success("✅ Dados salvos com sucesso no banco de dados Neon.tech!")
            else:
                st.warning("⚠️ Não foi possível registrar as métricas no banco de dados.")