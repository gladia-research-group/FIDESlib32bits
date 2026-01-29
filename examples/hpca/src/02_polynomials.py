#!/usr/bin/env python3
"""
Plot polynomial evaluation results comparing manual sigmoid and Chebyshev approaches.
"""

import matplotlib.pyplot as plt
import pandas as pd
import sys

def main():
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "polynomial_results.csv"
    
    df = pd.read_csv(csv_file)
    
    plt.figure(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    plt.plot(df['x'], df['expected'], 'k-', linewidth=2, label='Expected sigma(x)')
    plt.plot(df['x'], df['manual'], 'b--', linewidth=1.5, marker='o', markersize=4, label='Manual (degree 3)')
    plt.plot(df['x'], df['chebyshev'], 'r:', linewidth=1.5, marker='s', markersize=4, label='Chebyshev')
    
    plt.xlabel('x', fontsize=12)
    plt.ylabel('sigma(x)', fontsize=12)
    plt.title('Sigmoid Approximation: Manual vs Chebyshev', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('polynomial_plot.png', dpi=150)
    print(f"Plot saved to polynomial_plot.png")
    plt.show()

if __name__ == "__main__":
    main()
