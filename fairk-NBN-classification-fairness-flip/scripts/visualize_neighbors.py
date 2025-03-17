import logging
import os
import time
import hydra
import numpy as np
import matplotlib.lines as mlines
import imageio
import pandas as pd
from omegaconf import OmegaConf
from scipy.interpolate import griddata  # For interpolation
from dotenv import load_dotenv
from src.config.schema import Config
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from src.jobs.fairness_parity import FairnessParity
from src.utils.preprocess_utils import flip_value

load_dotenv()
hydra_config_path = '../' +os.getenv("HYDRA_CONFIG_PATH")
hydra_config_name = os.getenv("HYDRA_CONFIG_NAME")

def merge_t_dfs(df1, df2):
    df1 =df1.copy()
    df2 = df2.copy()

    df1['color'] = 1
    df2['color'] = 0

    selected_train_df = pd.concat([df1, df2], ignore_index=False, sort=False)
    return selected_train_df

def make_plot(fp,selected_val_df,neighbor_train_df, num=None):

    x_min, x_max = fp.x_train['value'].min()-1, fp.x_train['value'].max() +1
    y_min, y_max = fp.x_train['value1'].min()-1, fp.x_train['value1'].max() +1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2), np.arange(y_min, y_max, 0.2))

    Z = fp.model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(30, 30))
    bright_cmap = mcolors.ListedColormap(['#FFCC00', '#339933'])
    plt.imshow(Z, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap=bright_cmap, alpha=0.6)


    train_colors = ['red' if cls == 1 else 'blue' for cls in neighbor_train_df['color']]
    train_scatter =plt.scatter(neighbor_train_df['value'], neighbor_train_df['value1'],
               c=train_colors, marker='o', s=100, edgecolor='k', label='Train Neighbors', zorder=1)

    val_colors = ['red' if cls == 1 else 'blue' for cls in selected_val_df['color']]
    val_scatter =plt.scatter(selected_val_df['value'], selected_val_df['value1'],
                c=val_colors, marker='x',s=80, label='Validation Points', zorder=2)

    filename = f"frame_{num}.png"

    yellow_patch = mlines.Line2D([], [], color='#FFCC00', lw=4, label='Positive')
    green_patch = mlines.Line2D([], [], color='#339933', lw=4, label='Negative')

    plt.legend(handles=[yellow_patch, green_patch, train_scatter, val_scatter], loc='upper left', fontsize=16)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title('Decision Boundary')
    plt.savefig(filename,  bbox_inches='tight', pad_inches=0.04)
    print(f"Creating Frame{num+1}")

    plt.close()

    return filename

def make_gif(frames):
    with imageio.get_writer("fair_knn.gif", mode="I", duration=4, loop=100) as writer:
        for frame in frames:
            image = imageio.imread(frame)
            writer.append_data(image)




def get_train_representation(fp):
    train_indexes = list(fp.reverse_index_innit.keys())
    train_indexes_ = list(fp.reverse_index.keys())
    filtered_indexes = [index for index in train_indexes if index not in train_indexes_]
    neighbor_train = fp.x_train.loc[filtered_indexes]
    sensitive_val = [fp.y_train_sensitive_attr[i] for i, idx in enumerate(fp.x_train.index) if idx in filtered_indexes]
    neighbor_train['color'] = sensitive_val
    return neighbor_train

@hydra.main(config_path=hydra_config_path, config_name=hydra_config_name, version_base=None)
def main(config: Config):
    frames = []
    fp = FairnessParity(config)
    reslts_df, train_indexer, _ = fp.run_fairness_parity()
    neighbor_train_df = get_train_representation(fp)
    selected_val_df = merge_t_dfs(fp.t0, fp.t1)
    frame = make_plot(fp,selected_val_df,neighbor_train_df,num=0)
    frames.append(frame)
    for i,index in enumerate(train_indexer):
        fp.y_train = flip_value(fp.y_train, index, fp.class_positive_value,
                                  fp.config.data.class_attribute.name)
        fp._train()
        frame =make_plot(fp,selected_val_df,neighbor_train_df,num=i)
        frames.append(frame)

    make_gif(frames)

if __name__ == "__main__":
    main()
