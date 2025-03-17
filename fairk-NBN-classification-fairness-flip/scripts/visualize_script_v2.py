import os.path
import sys

import pandas as pd
from matplotlib import pyplot as plt


current = os.path.dirname((os.path.realpath(__file__)))
parent = os.path.dirname(current)
sys.path.append(parent)

csv_path =os.path.join(parent, "data/law_ida"
                               "/results/most_common_flip_results.csv")
df = pd.read_csv(csv_path)


def visualize_dom_attr_vs_sen_attr(df1, df2, name):
    plt.figure(figsize=(10, 6))

    min_dom = df1['number_dom_attr_predicted_positive'].min()
    plt.axhline(y=min_dom, color='orange', linestyle='--', linewidth=1.5,
                label='Flip Goal of Sensitive Attr Predicted Positive')

    # Plot lines with distinct red and blue shades
    plt.plot(df1['train_val_flipped'], df1['number_sensitive_attr_predicted_positive'],
             color='#FA8072' , label='IDA Red Predicted Positive')
    plt.plot(df2['train_val_flipped'], df2['number_sensitive_attr_predicted_positive'],
             color='#DD0000' , label='EDA Red Predicted Positive')

    plt.plot(df1['train_val_flipped'], df1['number_dom_attr_predicted_positive'],
             color='#72ECFA', label='IDA Blue Predicted Positive')
    plt.plot(df2['train_val_flipped'], df2['number_dom_attr_predicted_positive'],
             color='#00DDDD', label='EDA Blue Predicted Positive ')

    plt.xlabel('Iteration/Train Set Flips')
    plt.ylabel('Predicted Positive Count')
    plt.title('Red vs Blue Positive Predictions Over Iterations')
    plt.legend()
    plt.savefig(f'{name}.png', dpi=1200, bbox_inches='tight', pad_inches=0.04)



def visualize_rprvsbpr(df1, df2, name):
    plt.rcParams.update({
        "text.usetex": True,  # Use LaTeX for text rendering
        "font.family": "serif",  # Use a serif font
    })
    plt.figure(figsize=(10, 6))

    min_dom = df1['bpr'].min()
    plt.axhline(y=min_dom, color='orange', linestyle='--', linewidth=1.5,
                label='Initial BPR')

    # Plot lines with distinct red and blue shades
    plt.plot(df1['train_val_flipped'], df1['rpr'],linestyle='-', linewidth=1.8,
             color='#DD0000' , label='IDA Red Positive Rate')
    plt.plot(df2['train_val_flipped'], df2['rpr'],linestyle='-.', linewidth=2,
             color='#DD0000' , label='EDA Red Positive Rate')

    plt.plot(df1['train_val_flipped'], df1['bpr'],linestyle='-', linewidth=1.8,
             color='#00DDDD', label='IDA Blue Positive Rate')
    plt.plot(df2['train_val_flipped'], df2['bpr'],linestyle='-.', linewidth=2,
             color='#00DDDD', label='EDA Blue Positive Rate')

    plt.xlabel('Iteration/Train Set Flips')
    plt.ylabel('Positive Rate')
    plt.title('Red vs Blue Positive Rate over Iterations')
    plt.legend()
    plt.savefig(f'{name}.png', dpi=1200, bbox_inches='tight', pad_inches=0.04)
    plt.savefig(f'{name}.pdf', dpi=1200, bbox_inches='tight', pad_inches=0.04)

def visualize_accuracy(df1, df2, name):
    plt.rcParams.update({
        "text.usetex": True,  # Use LaTeX for text rendering
        "font.family": "serif",  # Use a serif font
    })
    plt.figure(figsize=(10, 6))


    # Plot lines with distinct red and blue shades
    plt.plot(df1['train_val_flipped'], df1['accuracy'],
             color='#FA8072' , label='IDA Accuracy')
    plt.plot(df2['train_val_flipped'], df2['accuracy'],
             color='#DD0000' , label='EDA Accuracy')


    plt.xlabel('Iteration/Train Set Flips')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Iterations')
    plt.legend()

    #plt.savefig(f'{name}.png', dpi=1200, bbox_inches='tight', pad_inches=0.04)
    plt.savefig(f'{name}.pdf', dpi=1200, bbox_inches='tight', pad_inches=0.04)

def visualize_random(df2, name):
    plt.rcParams.update({
        "text.usetex": True,  # Use LaTeX for text rendering
        "font.family": "serif",  # Use a serif font
    })
    plt.figure(figsize=(10, 6))

    min_dom = df2['bpr'].min()
    plt.axhline(y=min_dom, color='orange', linestyle='--', linewidth=1.5,
                label='Initial BPR')

    # Plot lines with distinct red and blue shades

    plt.plot(df2['train_val_flipped'], df2['rpr'],
             color='#DD0000' , label='Red Positive Rate')


    plt.plot(df2['train_val_flipped'], df2['bpr'],
             color='#00DDDD', label='Blue Positive Rate')

    plt.xlabel('Iteration/Train Set Flips')
    plt.ylabel('Positive Rate')
    plt.title('Red vs Blue Positive Rate over Iterations')
    plt.legend()
    plt.savefig(f'{name}.pdf', dpi=1200, bbox_inches='tight', pad_inches=0.04)

def visualize_compare_rprvsbpr(df1, df2, df3, name):
    plt.rcParams.update({
        "text.usetex": True,  # Use LaTeX for text rendering
        "font.family": "serif",  # Use a serif font
    })
    plt.figure(figsize=(10, 6))

    min_dom = df1['bpr'].min()
    plt.axhline(y=min_dom, color='orange', linestyle='--', linewidth=1.5,
                label='Initial BPR')

    # Plot lines with different styles
    plt.plot(df1['train_val_flipped'], df1['rpr'],color='#DD0000',
              linestyle='--', linewidth=1.8, label='IDA Red Positive Rate')
    plt.plot(df2['train_val_flipped'], df2['rpr'],color='#DD0000',
              linestyle='-', linewidth=1.5, label='EDA Red Positive Rate')
    plt.plot(df3['train_val_flipped'], df3['rpr'],color='#DD0000',
              linestyle='-.', linewidth=2, label='Random Red Positive Rate')

    plt.plot(df1['train_val_flipped'], df1['bpr'],linestyle='--', linewidth=1.8,
             color='#00DDDD', label='IDA Blue Positive Rate')
    plt.plot(df2['train_val_flipped'], df2['bpr'],linestyle='-', linewidth=1.5,
             color='#00DDDD', label='EDA Blue Positive Rate')
    plt.plot(df3['train_val_flipped'], df3['bpr'],linestyle='-.', linewidth=2,
             color='#00DDDD', label='Random Blue Positive Rate')

    plt.xlabel('Iteration/Train Set Flips')
    plt.ylabel('Positive Rate')
    plt.title('Red vs Blue Positive Rate over Iterations')
    plt.legend()
    plt.savefig(f'{name}.pdf', dpi=1200, bbox_inches='tight', pad_inches=0.04)