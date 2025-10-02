"""
Complete Text Entropy Analysis and Visualization

A self-contained module for analyzing character-level entropy in texts
and creating visualizations. Perfect for Jupyter notebooks.
"""

import matplotlib.pyplot as plt
import numpy as np
import nltk
from collections import Counter, defaultdict
import math
import requests
import string

# Download required NLTK data quietly
try:
    nltk.download('gutenberg')
except:
    pass

# Set matplotlib style and define colors
plt.style.use('ggplot')
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8B1E3F', '#3D5A80']

class TextEntropyAnalyzer:
    """A class for analyzing text entropy and Markov chains"""
    
    def __init__(self, max_gram_order=3):
        """
        Initialize the analyzer with configurable maximum gram order
        
        Args:
            max_gram_order (int): Maximum n-gram order to analyze (default 3)
                                 Markov chains will go up to max_gram_order - 1
        """
        self.max_gram_order = max_gram_order
        self.max_markov_order = max_gram_order - 1
        self.texts = {}
        self.results = {}
    
    def download_text(self, url, filename):
        """Download text from Project Gutenberg"""
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.text
        except Exception as e:
            return None

    def preprocess_text(self, text):
        """Clean and preprocess text - keep only letters and convert to lowercase"""
        lines = text.split('\n')
        start_idx = 0
        end_idx = len(lines)
        
        # Find actual text start (after header)
        for i, line in enumerate(lines):
            if '*** START' in line.upper() or 'CHAPTER' in line.upper():
                start_idx = i + 1
                break
        
        # Find actual text end (before footer)
        for i in range(len(lines) - 1, -1, -1):
            if '*** END' in lines[i].upper() or 'THE END' in lines[i].upper():
                end_idx = i
                break
        
        text = '\n'.join(lines[start_idx:end_idx])
        cleaned = ''.join(c.lower() if c.isalpha() else ' ' for c in text)
        cleaned = ' '.join(cleaned.split())
        return cleaned

    def get_ngrams(self, text, n):
        """Generate n-grams from text"""
        return [text[i:i+n] for i in range(len(text) - n + 1)]

    def calculate_entropy(self, frequencies):
        """Calculate entropy given frequency counts"""
        total = sum(frequencies.values())
        if total == 0:
            return 0
        
        entropy = 0
        for count in frequencies.values():
            if count > 0:
                prob = count / total
                entropy -= prob * math.log2(prob)
        return entropy

    def build_markov_chain(self, text, order):
        """Build a Markov chain of given order"""
        chain = defaultdict(Counter)
        for i in range(len(text) - order):
            context = text[i:i+order]
            next_char = text[i+order]
            chain[context][next_char] += 1
        return dict(chain)

    def calculate_markov_entropy(self, chain):
        """Calculate entropy of a Markov chain"""
        total_entropy = 0
        total_contexts = 0
        
        for context, next_chars in chain.items():
            context_entropy = self.calculate_entropy(next_chars)
            total_chars = sum(next_chars.values())
            total_entropy += context_entropy * total_chars
            total_contexts += total_chars
        
        return total_entropy / total_contexts if total_contexts > 0 else 0

    def analyze_ngrams(self, text, max_n=None):
        """Analyze n-gram entropy for n=1 to max_n"""
        if max_n is None:
            max_n = self.max_gram_order
            
        ngram_results = {}
        for n in range(1, max_n + 1):
            ngrams = self.get_ngrams(text, n)
            frequencies = Counter(ngrams)
            entropy = self.calculate_entropy(frequencies)
            unique_ngrams = len(frequencies)
            
            ngram_results[f'{n}gram'] = {
                'entropy': entropy,
                'unique_count': unique_ngrams,
                'total_count': len(ngrams)
            }
        return ngram_results

    def analyze_markov_chains(self, text, max_order=None):
        """Analyze Markov chain entropy for orders 0 to max_order"""
        if max_order is None:
            max_order = self.max_markov_order
            
        markov_results = {}
        
        # 0th order (unigram baseline)
        unigram_freq = Counter(text)
        markov_results[0] = {
            'entropy': self.calculate_entropy(unigram_freq),
            'contexts': len(unigram_freq)
        }
        
        # Higher order Markov chains
        for order in range(1, max_order + 1):
            chain = self.build_markov_chain(text, order)
            entropy = self.calculate_markov_entropy(chain)
            markov_results[order] = {
                'entropy': entropy,
                'contexts': len(chain)
            }
        return markov_results

    def calculate_entropy_rate_by_order(self, text, max_n=None):
        """
        Calculate entropy rate H_n/n for different n-gram orders
        
        The entropy rate H(X) is defined as lim(n→∞) (1/n) H_n
        This function computes H_n/n for each n to see the convergence
        
        Args:
            text (str): Input text
            max_n (int): Maximum n-gram order to analyze
            
        Returns:
            dict: Contains entropy rates for different n-gram orders
        """
        if max_n is None:
            max_n = self.max_gram_order
            
        entropy_rate_results = {
            'orders': [],
            'entropy_rates': [],  # H_n/n for each n
            'entropies': [],      # H_n for each n
            'unique_counts': []
        }
        
        # Calculate entropy rate for each n-gram order
        for n in range(1, max_n + 1):
            ngrams = self.get_ngrams(text, n)
            
            if len(ngrams) > 0:
                frequencies = Counter(ngrams)
                entropy = self.calculate_entropy(frequencies)
                entropy_rate = entropy / n  # H_n / n
                unique_count = len(frequencies)
                
                entropy_rate_results['orders'].append(n)
                entropy_rate_results['entropy_rates'].append(entropy_rate)
                entropy_rate_results['entropies'].append(entropy)
                entropy_rate_results['unique_counts'].append(unique_count)
        
        return entropy_rate_results

    def calculate_conditional_entropy_by_order(self, text, max_order=None):
        """
        Calculate conditional entropy H(X_n | X_1...X_{n-1}) for different Markov orders
        This represents the entropy rate for Markov chains
        
        Args:
            text (str): Input text
            max_order (int): Maximum Markov chain order to analyze
            
        Returns:
            dict: Contains conditional entropies for different orders
        """
        if max_order is None:
            max_order = self.max_markov_order
            
        conditional_entropy_results = {
            'orders': [],
            'conditional_entropies': [],
            'context_counts': []
        }
        
        # Calculate conditional entropy for each order
        # Note: Order 0 gives H(X), order 1 gives H(X|previous), etc.
        for order in range(0, max_order + 1):
            if order == 0:
                # 0th order: unconditional entropy H(X)
                unigram_freq = Counter(text)
                entropy = self.calculate_entropy(unigram_freq)
                context_count = len(unigram_freq)
            else:
                # Higher order: conditional entropy H(X | context)
                if len(text) > order:
                    chain = self.build_markov_chain(text, order)
                    entropy = self.calculate_markov_entropy(chain)
                    context_count = len(chain)
                else:
                    entropy = 0
                    context_count = 0
            
            conditional_entropy_results['orders'].append(order)
            conditional_entropy_results['conditional_entropies'].append(entropy)
            conditional_entropy_results['context_counts'].append(context_count)
        
        return conditional_entropy_results

    def analyze_text(self, text, title):
        """Perform complete entropy analysis on text"""
        ngram_results = self.analyze_ngrams(text)
        markov_results = self.analyze_markov_chains(text)
        
        # Calculate entropy rates by order
        entropy_rate_results = self.calculate_entropy_rate_by_order(text)
        conditional_entropy_results = self.calculate_conditional_entropy_by_order(text)
        
        results = {
            'title': title,
            'length': len(text),
            'ngrams': ngram_results,
            'markov': markov_results,
            'entropy_rate_by_order': entropy_rate_results,
            'conditional_entropy_by_order': conditional_entropy_results
        }
        return results

    def load_and_analyze_text(self, url, title):
        """Download, preprocess, and analyze a text"""
        raw_text = self.download_text(url, f"{title}.txt")
        if raw_text:
            cleaned_text = self.preprocess_text(raw_text)
            self.texts[title] = cleaned_text
            results = self.analyze_text(cleaned_text, title)
            self.results[title] = results
            return results
        return None

def get_classic_texts():
    """Return dictionary of classic text URLs from Project Gutenberg"""
    return {
        "Moby Dick": "https://www.gutenberg.org/files/2701/2701-0.txt",
        "War and Peace": "https://www.gutenberg.org/files/2600/2600-0.txt",
        "Pride and Prejudice": "https://www.gutenberg.org/files/1342/1342-0.txt",
        "Alice in Wonderland": "https://www.gutenberg.org/files/11/11-0.txt",
        "The Great Gatsby": "https://www.gutenberg.org/files/64317/64317-0.txt"
    }

def plot_ngram_entropy(analyzer, title, figsize=(10, 6)):
    """Plot n-gram entropy progression"""
    
    if title not in analyzer.results:
        print(f"No results found for '{title}'")
        return None
    
    results = analyzer.results[title]
    
    # Extract n-gram data up to analyzer's max_gram_order
    ngram_orders = []
    ngram_entropies = []
    
    for n in range(1, analyzer.max_gram_order + 1):
        gram_key = f'{n}gram'
        if gram_key in results['ngrams']:
            ngram_orders.append(n)
            ngram_entropies.append(results['ngrams'][gram_key]['entropy'])
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(ngram_orders, ngram_entropies, 'o-', linewidth=3, markersize=8, color=colors[0])
    ax.set_xlabel('N-gram Order (n)', fontsize=12)
    ax.set_ylabel('Entropy (bits)', fontsize=12)
    ax.set_title(f'N-gram Entropy vs Order: {title}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ngram_orders)
    
    # Add value labels on points
    for i, (x, y) in enumerate(zip(ngram_orders, ngram_entropies)):
        ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10)
    
    plt.tight_layout()
    return fig

def plot_cond_entropy(analyzer, title, figsize=(10, 6)):
    """Plot Markov chain entropy progression"""
    
    if title not in analyzer.results:
        print(f"No results found for '{title}'")
        return None
    
    results = analyzer.results[title]
    
    # Extract Markov chain data up to analyzer's max_markov_order
    markov_orders = []
    markov_entropies = []
    
    for order in range(analyzer.max_markov_order + 1):
        if order in results['markov']:
            markov_orders.append(order)
            markov_entropies.append(results['markov'][order]['entropy'])
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(markov_orders, markov_entropies, 'o-', linewidth=3, markersize=8, color=colors[1])
    ax.set_xlabel('Context Size', fontsize=12)
    ax.set_ylabel('Entropy (bits)', fontsize=12)
    ax.set_title(f'Conditional Entropy vs Context Size: {title}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(markov_orders)
    
    # Add value labels on points
    for i, (x, y) in enumerate(zip(markov_orders, markov_entropies)):
        ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10)
    
    plt.tight_layout()
    return fig

def plot_information_gain(analyzer, title, figsize=(10, 6)):
    """Plot information gain (entropy reduction) from context"""
    
    if title not in analyzer.results:
        print(f"No results found for '{title}'")
        return None
    
    results = analyzer.results[title]
    
    # Calculate entropy reductions (information gain)
    entropy_reductions = []
    reduction_labels = []
    
    if 0 in results['markov']:
        baseline_entropy = results['markov'][0]['entropy']
        for order in range(1, analyzer.max_markov_order + 1):
            if order in results['markov']:
                reduction = baseline_entropy - results['markov'][order]['entropy']
                entropy_reductions.append(reduction)
                reduction_labels.append(f'Order {order}')
    
    if not entropy_reductions:
        print("No entropy reduction data available")
        return None
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    plot_colors = [colors[i % len(colors)] for i in range(2, len(entropy_reductions) + 2)]
    bars = ax.bar(reduction_labels, entropy_reductions, color=plot_colors, alpha=0.8)
    
    ax.set_xlabel('Markov Chain Order', fontsize=12)
    ax.set_ylabel('Information Gain (bits)', fontsize=12)
    ax.set_title(f'Information Gain from Context: {title}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels if there are many bars
    if len(reduction_labels) > 4:
        ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    return fig

def plot_entropy_rate_by_order(analyzer, title, figsize=(10, 6)):
    """Plot entropy rate H_n/n vs n-gram order n, showing convergence to H(X)"""
    
    if title not in analyzer.results:
        print(f"No results found for '{title}'")
        return None
    
    results = analyzer.results[title]
    
    if 'entropy_rate_by_order' not in results:
        print("No entropy rate by order data available")
        return None
    
    entropy_rate_data = results['entropy_rate_by_order']
    orders = entropy_rate_data['orders']
    entropy_rates = entropy_rate_data['entropy_rates']
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(orders, entropy_rates, 'o-', linewidth=3, markersize=8, color=colors[0])
    ax.set_xlabel('N-gram Order (n)', fontsize=12)
    ax.set_ylabel('Entropy Rate H_n/n (bits per symbol)', fontsize=12)
    ax.set_title(f'Entropy Rate H(X) = lim(n→∞) H_n/n: {title}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(orders)
    
    # Add value labels on points
    for x, y in zip(orders, entropy_rates):
        ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10)
    
    # Add horizontal line showing the final entropy rate estimate
    if len(entropy_rates) > 1:
        final_rate = entropy_rates[-1]
        ax.axhline(y=final_rate, color='red', linestyle='--', alpha=0.7, 
                   label=f'H_{len(orders)}/{len(orders)} ≈ {final_rate:.3f}')
        ax.legend()
    
    plt.tight_layout()
    return fig

def plot_entropy_and_entropy_rate(analyzer, title, figsize=(12, 6)):
    """Plot both H_n and H_n/n to show the relationship"""
    
    if title not in analyzer.results:
        print(f"No results found for '{title}'")
        return None
    
    results = analyzer.results[title]
    
    if 'entropy_rate_by_order' not in results:
        print("No entropy rate by order data available")
        return None
    
    entropy_rate_data = results['entropy_rate_by_order']
    orders = entropy_rate_data['orders']
    entropies = entropy_rate_data['entropies']  # H_n
    entropy_rates = entropy_rate_data['entropy_rates']  # H_n/n
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot H_n (raw entropy)
    ax1.plot(orders, entropies, 'o-', linewidth=3, markersize=8, color=colors[1])
    ax1.set_xlabel('N-gram Order (n)', fontsize=12)
    ax1.set_ylabel('Entropy H_n (bits)', fontsize=12)
    ax1.set_title('Raw N-gram Entropy H_n', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(orders)
    
    # Add value labels
    for x, y in zip(orders, entropies):
        ax1.annotate(f'{y:.1f}', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    # Plot H_n/n (entropy rate)
    ax2.plot(orders, entropy_rates, 'o-', linewidth=3, markersize=8, color=colors[0])
    ax2.set_xlabel('N-gram Order (n)', fontsize=12)
    ax2.set_ylabel('Entropy Rate H_n/n (bits per symbol)', fontsize=12)
    ax2.set_title('Entropy Rate H_n/n → H(X)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(orders)
    
    # Add value labels
    for x, y in zip(orders, entropy_rates):
        ax2.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    # Add horizontal line for final entropy rate
    if len(entropy_rates) > 1:
        final_rate = entropy_rates[-1]
        ax2.axhline(y=final_rate, color='red', linestyle='--', alpha=0.7)
    
    plt.suptitle(f'N-gram Entropy Analysis: {title}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_conditional_entropy_by_order(analyzer, title, figsize=(10, 6)):
    """Plot conditional entropy H(X|context) vs Markov order"""
    
    if title not in analyzer.results:
        print(f"No results found for '{title}'")
        return None
    
    results = analyzer.results[title]
    
    if 'conditional_entropy_by_order' not in results:
        print("No conditional entropy by order data available")
        return None
    
    conditional_data = results['conditional_entropy_by_order']
    orders = conditional_data['orders']
    conditional_entropies = conditional_data['conditional_entropies']
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(orders, conditional_entropies, 'o-', linewidth=3, markersize=8, color=colors[1])
    ax.set_xlabel('Context Size (Markov Order)', fontsize=12)
    ax.set_ylabel('Conditional Entropy H(X|context) (bits)', fontsize=12)
    ax.set_title(f'Conditional Entropy vs Context Size: {title}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(orders)
    
    # Add value labels on points
    for x, y in zip(orders, conditional_entropies):
        ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10)
    
    plt.tight_layout()
    return fig

def plot_entropy_comparison(analyzer, title, figsize=(15, 5)):
    """Plot comprehensive entropy analysis: H_n, H_n/n, and H(X|context)"""
    
    if title not in analyzer.results:
        print(f"No results found for '{title}'")
        return None
    
    results = analyzer.results[title]
    
    if 'entropy_rate_by_order' not in results or 'conditional_entropy_by_order' not in results:
        print("Missing entropy rate or conditional entropy data")
        return None
    
    # Create three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    # Plot 1: Raw entropy H_n
    entropy_rate_data = results['entropy_rate_by_order']
    orders1 = entropy_rate_data['orders']
    entropies = entropy_rate_data['entropies']
    
    ax1.plot(orders1, entropies, 'o-', linewidth=3, markersize=8, color=colors[1])
    ax1.set_xlabel('N-gram Order (n)', fontsize=10)
    ax1.set_ylabel('Entropy H_n (bits)', fontsize=10)
    ax1.set_title('Raw N-gram Entropy', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(orders1)
    
    # Plot 2: Entropy rate H_n/n
    entropy_rates = entropy_rate_data['entropy_rates']
    
    ax2.plot(orders1, entropy_rates, 'o-', linewidth=3, markersize=8, color=colors[0])
    ax2.set_xlabel('N-gram Order (n)', fontsize=10)
    ax2.set_ylabel('Entropy Rate H_n/n (bits/symbol)', fontsize=10)
    ax2.set_title('Entropy Rate → H(X)', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(orders1)
    
    # Add horizontal line for convergence
    if len(entropy_rates) > 1:
        final_rate = entropy_rates[-1]
        ax2.axhline(y=final_rate, color='red', linestyle='--', alpha=0.7)
    
    # Plot 3: Conditional entropy H(X|context)
    conditional_data = results['conditional_entropy_by_order']
    orders2 = conditional_data['orders']
    conditional_entropies = conditional_data['conditional_entropies']
    
    ax3.plot(orders2, conditional_entropies, 'o-', linewidth=3, markersize=8, color=colors[2])
    ax3.set_xlabel('Context Size', fontsize=10)
    ax3.set_ylabel('H(X|context) (bits)', fontsize=10)
    ax3.set_title('Conditional Entropy', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(orders2)
    
    plt.suptitle(f'Complete Entropy Analysis: {title}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig



def diagnose_entropy_methods(analyzer, title):
    """Compare the two entropy calculation methods"""
    
    if title not in analyzer.results:
        print(f"No results for {title}")
        return
    
    results = analyzer.results[title]
    text = analyzer.texts[title]
    
    print(f"Text length: {len(text):,} characters")
    print(f"Unique characters: {len(set(text))}")
    print()
    
    # Check high-order values
    entropy_rate_data = results['entropy_rate_by_order']
    conditional_data = results['conditional_entropy_by_order']
    
    max_order = min(len(entropy_rate_data['orders']), len(conditional_data['orders'])) - 1
    
    print("Comparison of final values:")
    print(f"Conditional entropy (order {max_order}): {conditional_data['conditional_entropies'][max_order]:.4f} bits")
    print(f"Entropy rate H_{max_order+1}/{max_order+1}: {entropy_rate_data['entropy_rates'][max_order]:.4f} bits")
    print()
    
    # Check for boundary effects in n-gram calculation
    for n in [10, 12, 15]:
        if n <= len(entropy_rate_data['orders']):
            idx = n - 1  # 0-indexed
            total_ngrams = entropy_rate_data['orders'][idx]
            unique_ngrams = entropy_rate_data['unique_counts'][idx] if 'unique_counts' in entropy_rate_data else "N/A"
            
            # Calculate what percentage are unique (indication of data sparsity)
            if isinstance(unique_ngrams, int):
                sparsity = unique_ngrams / (len(text) - n + 1) * 100
                print(f"N-gram order {n}: {unique_ngrams:,} unique / {len(text) - n + 1:,} total = {sparsity:.1f}% unique")
    
    print()
    print("Expected behavior:")
    print("- If entropy rate H_n/n is too high, likely due to data sparsity in large n-grams")
    print("- Conditional entropy should be the more reliable estimate for high orders")
    print("- Both should converge to the same value in theory")