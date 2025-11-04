#!/usr/bin/env python3
"""
BERT Attention Animation - Educational Demo
Shows how the word "park" gets different embeddings based on context
"""

import os
import io
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import PCA
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Configuration
MODEL_NAME = "bert-base-cased"
LAYER = 8  # Middle layer for interesting attention patterns
OUTPUT_DIR = "bert_attention_demo"
FRAME_DURATION_MS = 10000  # Duration of each frame in milliseconds (10 seconds)

# Three sentences with "park" in different contexts
SENTENCES = {
    "location": "I went to the park to play soccer.",
    "verb": "Please park the car in the garage.",
    "proper_noun": "Central Park is in New York City."
}

COLORS = {
    "location": "#FF9500",      # Orange
    "verb": "#007AFF",          # Blue
    "proper_noun": "#FF2D55"    # Pink
}


def load_bert():
    """Load BERT model and tokenizer"""
    print("Loading BERT model...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertModel.from_pretrained(MODEL_NAME, output_attentions=True)
    model.eval()
    return tokenizer, model


def get_embeddings_and_attention(tokenizer, model, sentence):
    """
    Get the final contextualized embeddings and attention weights for a sentence
    Returns: tokens, embeddings, attention_weights, initial_embeddings
    """
    # Tokenize
    inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
    
    # Get initial embeddings (layer 0 - just word embeddings before any context)
    initial_embeddings = outputs.hidden_states[0][0].numpy()  # [seq_len, hidden_dim]
    
    # Get embeddings from the specified layer (after attention)
    embeddings = outputs.hidden_states[LAYER][0].numpy()  # [seq_len, hidden_dim]
    
    # Get attention weights (average across all heads)
    attention = outputs.attentions[LAYER][0].mean(dim=0).numpy()  # [seq_len, seq_len]
    
    return tokens, embeddings, attention, initial_embeddings


def find_park_index(tokens):
    """Find the index of 'park' in the token list"""
    for i, token in enumerate(tokens):
        if 'park' in token.lower():
            return i
    return None


def create_attention_animation(tokenizer, model):
    """
    Create an animation showing how attention affects the embedding of 'park'
    """
    # Collect all data
    all_data = {}
    all_embeddings = []
    
    for context_name, sentence in SENTENCES.items():
        tokens, embeddings, attention, initial_embeddings = get_embeddings_and_attention(tokenizer, model, sentence)
        park_idx = find_park_index(tokens)
        
        if park_idx is not None:
            all_data[context_name] = {
                'tokens': tokens,
                'embeddings': embeddings,
                'initial_embeddings': initial_embeddings,
                'attention': attention,
                'park_idx': park_idx,
                'sentence': sentence
            }
            all_embeddings.append(embeddings)
            all_embeddings.append(initial_embeddings)
    
    # Fit PCA on all embeddings
    all_emb_concat = np.vstack(all_embeddings)
    pca = PCA(n_components=2, random_state=42)
    pca.fit(all_emb_concat)
    
    # Project embeddings to 2D for each sentence
    for context_name in all_data:
        embeddings_2d = pca.transform(all_data[context_name]['embeddings'])
        initial_embeddings_2d = pca.transform(all_data[context_name]['initial_embeddings'])
        all_data[context_name]['embeddings_2d'] = embeddings_2d
        all_data[context_name]['initial_embeddings_2d'] = initial_embeddings_2d
    
    return all_data, pca


def animate_attention_flow(all_data):
    """
    Create animation showing attention flow for each context
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for context_name, data in all_data.items():
        print(f"Creating animation for: {context_name}")
        
        tokens = data['tokens']
        embeddings_2d = data['embeddings_2d']
        initial_embeddings_2d = data['initial_embeddings_2d']
        attention = data['attention']
        park_idx = data['park_idx']
        sentence = data['sentence']
        color = COLORS[context_name]
        
        # Get attention weights TO park FROM other tokens
        attention_to_park = attention[park_idx, :]
        
        # Get initial and final positions of park
        park_initial = initial_embeddings_2d[park_idx]
        park_final = embeddings_2d[park_idx]
        
        # Debug: print positions
        print(f"  Initial: ({park_initial[0]:.2f}, {park_initial[1]:.2f})")
        print(f"  Final: ({park_final[0]:.2f}, {park_final[1]:.2f})")
        
        # Calculate fixed axis limits for consistent scaling across all frames
        all_x = embeddings_2d[:, 0].tolist() + initial_embeddings_2d[:, 0].tolist()
        all_y = embeddings_2d[:, 1].tolist() + initial_embeddings_2d[:, 1].tolist()
        x_margin = (max(all_x) - min(all_x)) * 0.1
        y_margin = (max(all_y) - min(all_y)) * 0.1
        xlim = (min(all_x) - x_margin, max(all_x) + x_margin)
        ylim = (min(all_y) - y_margin, max(all_y) + y_margin)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        def animate(frame):
            ax1.clear()
            ax2.clear()
            
            # Filter out special tokens for cleaner visualization
            filtered_indices = [i for i, tok in enumerate(tokens) if tok not in ['[CLS]', '[SEP]']]
            filtered_tokens = [tokens[i] for i in filtered_indices]
            filtered_attention = [attention_to_park[i] for i in filtered_indices]
            
            # Find park index in filtered list
            park_filtered_idx = filtered_indices.index(park_idx)
            
            # Left panel: Attention weights (without special tokens)
            ax1.barh(range(len(filtered_tokens)), filtered_attention, color=color, alpha=0.7)
            ax1.set_yticks(range(len(filtered_tokens)))
            ax1.set_yticklabels(filtered_tokens)
            ax1.set_xlabel('Attention Weight', fontsize=12)
            ax1.set_title(f'Attention to "park" (frame {frame+1}/3)', fontsize=14, fontweight='bold')
            ax1.invert_yaxis()
            
            # Highlight the park token
            ax1.barh(park_filtered_idx, filtered_attention[park_filtered_idx], color='gold', alpha=0.9, edgecolor='black', linewidth=2)
            
            # Right panel: Embedding space
            if frame == 0:
                # Show initial position (before context) - excluding special tokens
                for i, token in enumerate(tokens):
                    if token not in ['[CLS]', '[SEP]']:
                        ax2.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], 
                                   s=100, c='gray', alpha=0.3)
                
                # Show initial park position
                ax2.scatter(park_initial[0], park_initial[1], 
                           s=300, c='lightgray', marker='o', edgecolor='black', linewidth=2, 
                           label='"park" (initial, no context)')
                ax2.annotate('park\n(initial)', park_initial, fontsize=12, fontweight='bold',
                           xytext=(5, 5), textcoords='offset points')
                
                # Show final park position
                ax2.scatter(park_final[0], park_final[1], 
                           s=300, c=color, marker='*', edgecolor='black', linewidth=2, 
                           label=f'"park" (after attention)')
                ax2.annotate('park\n(contextualized)', park_final, fontsize=12, fontweight='bold',
                           xytext=(5, 5), textcoords='offset points', color=color)
                
            elif frame == 1:
                # Show the movement vector - excluding special tokens
                for i, token in enumerate(tokens):
                    if token not in ['[CLS]', '[SEP]']:
                        ax2.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], 
                                   s=100, c='gray', alpha=0.3)
                
                # Initial position
                ax2.scatter(park_initial[0], park_initial[1], 
                           s=200, c='lightgray', marker='o', edgecolor='black', linewidth=2,
                           label='"park" (initial)')
                
                # Final position
                ax2.scatter(park_final[0], park_final[1], 
                           s=300, c=color, marker='*', edgecolor='black', linewidth=2,
                           label='"park" (final)')
                
                # Draw the movement vector as an arrow
                ax2.annotate('', xy=park_final, xytext=park_initial,
                           arrowprops=dict(arrowstyle='->', lw=3, color=color, 
                                         connectionstyle='arc3,rad=0'))
                
                # Calculate and show distance
                distance = np.linalg.norm(park_final - park_initial)
                mid_point = (park_initial + park_final) / 2
                ax2.text(mid_point[0], mid_point[1], f'Δ = {distance:.2f}',
                        fontsize=12, fontweight='bold', color=color,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=color))
                
            else:
                # Show attention connections - excluding special tokens
                for i, token in enumerate(tokens):
                    if token not in ['[CLS]', '[SEP]']:
                        ax2.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], 
                                   s=100, c='gray', alpha=0.5)
                
                # Draw attention lines - excluding special tokens (thicker and more visible)
                for i in range(len(tokens)):
                    if i != park_idx and tokens[i] not in ['[CLS]', '[SEP]']:
                        weight = attention_to_park[i]
                        # Scale up the linewidth: minimum 1, maximum 6
                        linewidth = 1 + weight * 10
                        alpha = 0.3 + weight * 1.4  # More visible alpha
                        ax2.plot([embeddings_2d[i, 0], park_final[0]],
                               [embeddings_2d[i, 1], park_final[1]],
                               'b-', alpha=min(alpha, 0.9), linewidth=linewidth)
                
                # Add token labels
                for i, token in enumerate(tokens):
                    if token not in ['[CLS]', '[SEP]']:
                        ax2.annotate(token, embeddings_2d[i], 
                                   fontsize=10, fontweight='bold' if i == park_idx else 'normal',
                                   xytext=(5, 5), textcoords='offset points',
                                   color=color if i == park_idx else 'black')
                
                # Show movement vector
                ax2.annotate('', xy=park_final, xytext=park_initial,
                           arrowprops=dict(arrowstyle='->', lw=2, color=color, alpha=0.5))
                
                ax2.scatter(park_initial[0], park_initial[1], 
                           s=150, c='lightgray', marker='o', edgecolor='black', linewidth=1)
                ax2.scatter(park_final[0], park_final[1], 
                           s=300, c=color, marker='*', edgecolor='black', linewidth=2,
                           label=f'"park" moved by attention')
            
            ax2.set_xlabel('PCA Dimension 1', fontsize=12)
            ax2.set_ylabel('PCA Dimension 2', fontsize=12)
            ax2.set_title('Embedding Space (2D projection)', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
            
            # Set fixed axis limits for consistent scaling across frames
            ax2.set_xlim(xlim)
            ax2.set_ylim(ylim)
            
            # Main title
            fig.suptitle(f'Context: {context_name.upper()} - "{sentence}"', 
                        fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            return []
        
        # Create animation with 3 frames - save each frame manually for precise timing
        output_file = os.path.join(OUTPUT_DIR, f"park_{context_name}_attention.gif")
        frames = []
        
        for i in range(3):
            animate(i)
            # Save current figure to a buffer
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            frames.append(Image.open(buf).copy())
            buf.close()
        
        # Save as GIF with proper duration (in milliseconds)
        frames[0].save(
            output_file,
            save_all=True,
            append_images=frames[1:],
            duration=FRAME_DURATION_MS,
            loop=0
        )
        plt.close()
        print(f"  Saved: {output_file}")


def create_comparison_plot(all_data):
    """
    Create a comparison showing all three 'park' embeddings in the same space
    """
    print("Creating comparison plot...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot each context
    for context_name, data in all_data.items():
        embeddings_2d = data['embeddings_2d']
        park_idx = data['park_idx']
        tokens = data['tokens']
        color = COLORS[context_name]
        
        # Debug: print the park position
        park_pos = embeddings_2d[park_idx]
        print(f"  {context_name}: park at ({park_pos[0]:.2f}, {park_pos[1]:.2f})")
        
        # Plot all tokens lightly
        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                  s=50, c=color, alpha=0.2)
        
        # Highlight park token
        ax.scatter(embeddings_2d[park_idx, 0], embeddings_2d[park_idx, 1], 
                  s=400, c=color, marker='*', edgecolor='black', linewidth=2,
                  label=f'park ({context_name})')
        
        ax.annotate(f'park\n({context_name})', embeddings_2d[park_idx],
                   fontsize=12, fontweight='bold',
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    ax.set_xlabel('PCA Dimension 1', fontsize=14)
    ax.set_ylabel('PCA Dimension 2', fontsize=14)
    ax.set_title('How "park" moves in embedding space based on context\n(BERT Layer 8)', 
                fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_file = os.path.join(OUTPUT_DIR, "park_comparison.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")


def main():
    """Main execution"""
    print("=" * 60)
    print("BERT Attention Mechanism - Educational Demo")
    print("Showing how 'park' changes based on context")
    print("=" * 60)
    
    # Load model
    tokenizer, model = load_bert()
    
    # Process sentences and get embeddings
    print("\nProcessing sentences...")
    all_data, pca = create_attention_animation(tokenizer, model)
    
    # Create animations
    print("\nCreating animations...")
    animate_attention_flow(all_data)
    
    # Create comparison
    print("\nCreating comparison...")
    create_comparison_plot(all_data)
    
    print("\n" + "=" * 60)
    print(f"✓ Done! Check the '{OUTPUT_DIR}' folder for results")
    print("=" * 60)
    print("\nFiles created:")
    print("  1. park_location_attention.gif")
    print("  2. park_verb_attention.gif")
    print("  3. park_proper_noun_attention.gif")
    print("  4. park_comparison.png")


if __name__ == "__main__":
    main()
