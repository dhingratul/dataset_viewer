import streamlit as st
from datasets import load_dataset
from visualizer import TextVisualizer, ImageVisualizer, VideoVisualizer
import traceback
from PIL import Image
import io
import numpy as np
import pandas as pd
import random
import json

def is_valid_dataset(dataset_name):
    """Check if dataset exists in Hugging Face hub"""
    try:
        # Try loading just the dataset info first
        dataset_info = load_dataset(dataset_name, download_mode='force_redownload')
        return True
    except Exception as e:
        st.error(f"Error loading dataset '{dataset_name}': {str(e)}")
        return False

def format_conversation(conversations):
    """Format conversation data for display"""
    try:
        if isinstance(conversations, str):
            conversations = json.loads(conversations)
        
        if isinstance(conversations, list):
            for conv in conversations:
                if isinstance(conv, dict) and 'from' in conv and 'value' in conv:
                    st.markdown(f"**{conv['from']}**: {conv['value']}")
                else:
                    st.write(conv)
        else:
            st.write(conversations)
    except Exception as e:
        st.error(f"Error formatting conversation: {str(e)}")
        st.write(conversations)

def format_llava_sample(sample):
    """Special handling for LLaVA dataset format"""
    try:
        # Handle conversations
        if 'conversations' in sample:
            st.markdown("**Conversation:**")
            for conv in sample['conversations']:
                role = conv.get('from', 'unknown')
                content = conv.get('value', '')
                st.markdown(f"**{role}**: {content}")
        
        # Handle image
        if 'image' in sample:
            st.markdown("**Image URL:**")
            st.write(sample['image'])
            try:
                # You might need to add image loading from URL here
                st.image(sample['image'], width=400)
            except:
                st.write("Could not load image directly. URL provided above.")
        
        # Handle other fields
        for key, value in sample.items():
            if key not in ['conversations', 'image']:
                st.markdown(f"**{key}**: {value}")
    
    except Exception as e:
        st.error(f"Error formatting LLaVA sample: {str(e)}")
        st.write(sample)

def main():
    st.title("Hugging Face Dataset Visualizer")

    # Dataset input with examples
    st.markdown("""
    Try these example datasets:
    - `mnist`: Handwritten digits
    - `fashion_mnist`: Fashion items
    - `cifar10`: Natural images
    - `beans`: Plant disease images
    """)

    dataset_name = st.text_input("Enter Hugging Face dataset name")

    if dataset_name:
        try:
            # Special handling for LLaVA dataset
            if 'llava' in dataset_name.lower():
                with st.spinner(f"Loading LLaVA dataset '{dataset_name}'..."):
                    try:
                        dataset = load_dataset(dataset_name, trust_remote_code=True)
                    except Exception as e:
                        st.error(f"Failed to load LLaVA dataset: {str(e)}")
                        return
            else:
                # Regular dataset loading
                if not is_valid_dataset(dataset_name):
                    return
                dataset = load_dataset(dataset_name)

            # Display dataset info
            st.subheader("Dataset Information")
            st.write(f"Available splits: {list(dataset.keys())}")

            # Select split
            split = st.selectbox("Select split", list(dataset.keys()))

            if split:
                selected_data = dataset[split]

                # Display features
                st.subheader("Features")
                features = selected_data.features
                st.write(features)

                # Sample size selector
                total_samples = len(selected_data)
                st.write(f"Total samples in split: {total_samples}")
                sample_size = st.slider("Select number of samples to visualize",
                                      min_value=1,
                                      max_value=min(10, total_samples),
                                      value=min(3, total_samples))

                # Sampling method selection
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("Get First K Samples"):
                        indices = range(sample_size)
                with col2:
                    if st.button("Get Random K Samples"):
                        indices = random.sample(range(total_samples), sample_size)

                # Default to first K samples if no button is clicked
                if 'indices' not in locals():
                    indices = range(sample_size)

                # Visualize samples
                st.subheader("Data Samples")

                # Create column headers once
                header_col1, header_col2 = st.columns([3, 1])
                with header_col1:
                    st.markdown("**Data**")
                with header_col2:
                    st.markdown("**Label**")
                st.markdown("---")

                try:
                    samples = selected_data.select(indices)

                    # Create visualizers
                    text_viz = TextVisualizer()
                    image_viz = ImageVisualizer()
                    video_viz = VideoVisualizer()

                    # Create a table-like display
                    for idx, sample in enumerate(samples):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            if 'llava' in dataset_name.lower():
                                format_llava_sample(sample)
                            else:
                                # Regular visualization code...
                                if 'conversations' in sample:
                                    st.markdown("**Conversation:**")
                                    format_conversation(sample['conversations'])
                                
                                if 'image' in sample:
                                    st.markdown("**Image:**")
                                    image_viz.visualize(sample['image'])
                                elif 'img' in sample:
                                    image_viz.visualize(sample['img'])
                                elif 'pixel_values' in sample:
                                    image_viz.visualize(sample['pixel_values'])
                        
                        with col2:
                            if 'label' in sample:
                                label_value = sample['label']
                                if hasattr(features['label'], 'names'):
                                    label_name = features['label'].names[label_value]
                                    st.markdown(f"**{label_name}**")
                                else:
                                    st.markdown(f"**{label_value}**")
                            
                            # Display other metadata
                            for key, value in sample.items():
                                if key not in ['image', 'img', 'pixel_values', 'label', 'conversations']:
                                    st.markdown(f"**{key}:** {value}")
                        
                        st.markdown("---")

                except Exception as e:
                    st.error(f"Error visualizing samples: {str(e)}")
                    st.code(traceback.format_exc())

        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
