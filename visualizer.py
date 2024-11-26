import streamlit as st
import numpy as np
from PIL import Image
import io
import json
from datasets.features import Image as ImageFeature
from datasets.features import Video as VideoFeature

class TextVisualizer:
    def can_visualize(self, data):
        return isinstance(data, (str, int, float, list, dict))
    
    def visualize(self, data):
        if isinstance(data, (list, dict)):
            # Handle conversation format
            try:
                if isinstance(data, str):
                    # Try to parse string as JSON
                    data = json.loads(data)
                
                # Format conversations
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and 'from' in item and 'value' in item:
                            st.markdown(f"**{item['from']}**: {item['value']}")
                        else:
                            st.write(item)
                else:
                    st.write(data)
            except Exception as e:
                st.error(f"Error parsing conversation: {str(e)}")
                st.write(data)
        else:
            st.write(data)

class ImageVisualizer:
    def can_visualize(self, data, feature):
        return (
            isinstance(feature, ImageFeature) or 
            isinstance(data, np.ndarray) and len(data.shape) in [2, 3] or
            isinstance(data, Image.Image) or
            (isinstance(data, str) and data.endswith(('.jpg', '.png', '.jpeg')))
        )
    
    def visualize(self, data):
        try:
            if isinstance(data, str) and data.endswith(('.jpg', '.png', '.jpeg')):
                # Handle image paths/filenames
                st.write(f"Image file: {data}")
                return
                
            if isinstance(data, dict) and 'bytes' in data:
                # Handle image bytes format
                image = Image.open(io.BytesIO(data['bytes']))
                st.image(image, width=400)
            elif isinstance(data, Image.Image):
                # Handle PIL Image directly
                st.image(data, width=400)
            elif isinstance(data, np.ndarray):
                # Handle numpy array format
                if data.dtype == np.uint8:
                    st.image(data, width=400)
                else:
                    st.image(data.astype(np.uint8), width=400)
            else:
                st.write("Unsupported image format")
                st.write(f"Data type: {type(data)}")
                if isinstance(data, dict):
                    st.write(f"Dictionary keys: {data.keys()}")
        except Exception as e:
            st.error(f"Error visualizing image: {str(e)}")
            st.write(f"Data type: {type(data)}")
            if isinstance(data, np.ndarray):
                st.write(f"Data shape: {data.shape}")
                st.write(f"Data dtype: {data.dtype}")

class VideoVisualizer:
    def can_visualize(self, data, feature):
        return isinstance(feature, VideoFeature)
    
    def visualize(self, data):
        if isinstance(data, dict) and 'bytes' in data:
            with st.spinner("Loading video..."):
                st.video(data['bytes'])
        else:
            st.write("Unsupported video format") 