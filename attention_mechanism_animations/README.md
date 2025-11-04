# 🎓 BERT Attention Mechanism - Educational Demo

An animated visualization showing how the word "park" gets different embeddings based on context through BERT's attention mechanism.

## 📁 File Structure

- **`attention_animation.py`** — Main script for generating animated GIFs
- **`bert_attention_demo/`** — Output folder for generated visualizations
- **`pixi.toml`** — Dependencies and environment setup

## ✨ Key Features

### 1. **Context-Specific Animations** (3 GIFs)
Three different meanings of "park" are visualized:
- **Location**: "I went to the **park** to play soccer"
- **Verb**: "Please **park** the car in the garage"
- **Proper Noun**: "Central **Park** is in New York City"

Each animation has **3 frames**:
- **Frame 1**: Shows initial (non-contextualized) vs. final (contextualized) position in 2D embedding space
- **Frame 2**: Displays movement vector with distance (Δ) showing how far "park" moved
- **Frame 3**: Reveals attention connections (blue lines) showing which words influenced the movement

### 2. **Dual-Panel Visualization**
- **Left Panel**: Attention weights bar chart (which words "park" pays attention to)
- **Right Panel**: 2D embedding space (PCA projection showing semantic positions)

### 3. **Comparison Plot** (1 PNG)
Static visualization showing all three "park" embeddings in the same 2D space, demonstrating how context changes word meaning.

## 🎨 Visual Design

- **Orange**: Location context
- **Blue**: Verb context  
- **Pink**: Proper noun context
- **Blue lines**: Attention connections (thickness = attention weight)
- Special tokens ([CLS], [SEP]) are hidden for clarity

## 🚀 Quick Start

```bash
# Run the script
pixi run python attention_animation.py
```

This generates 4 files in `bert_attention_demo/`:
1. `park_location_attention.gif`
2. `park_verb_attention.gif`
3. `park_proper_noun_attention.gif`
4. `park_comparison.png`

## 🎮 How to Use in Class

### Teaching Sequence:

1. **Show Frame 1**: "Here's where 'park' starts before any context (gray dot) and where it ends up after attention (colored star)"

2. **Show Frame 2**: "Look at this arrow - it shows HOW FAR the word moved in semantic space because of context"

3. **Show Frame 3**: "These blue lines show WHICH words influenced 'park' - thicker lines mean stronger attention"

4. **Compare Contexts**: Show the comparison plot to demonstrate that the same word occupies different semantic positions based on surrounding words

### Key Teaching Points:

- **Contextualization**: Same word, different positions = different meanings
- **Attention Mechanism**: Words "look at" other words to understand context
- **Vector Representation**: Words exist in high-dimensional space (768D for BERT), projected to 2D for visualization
- **Semantic Distance**: Words closer in space have more similar meanings

## 🔬 Technical Details

- **Model**: BERT-base-uncased
- **Layer**: 8 (middle layer for interesting attention patterns)
- **Dimensionality**: 768D embeddings → 2D via PCA
- **Attention**: Averaged across all attention heads in layer 8
- **Frame Duration**: 3 seconds per frame
