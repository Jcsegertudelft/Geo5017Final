import matplotlib.pyplot as plt
import pandas as pd

def plot_results(results_file):
    df = pd.read_csv(results_file)
    print(df.columns)
    fig, (ax1,ax2) = plt.subplots(1,2, sharex = True)
    fig.set_size_inches(8,5)
    fig.supxlabel('Epoch', fontsize = 15)

    ax1.set_xlim(0,101)

    ax1.plot(df['epoch'],df['train/box_loss'],color='red', marker = 'o',markersize = 4,alpha = 0.7)
    ax1.plot(df['epoch'],df['val/box_loss'],color='blue', marker = 'o',markersize = 4,alpha = 0.7)
    ax1.set_title('Box loss', size = 16, weight = 'bold')

    ax2.plot(df['epoch'],df['train/cls_loss'],color='red',label='training set', marker='o',markersize=4,alpha = 0.7)
    ax2.plot(df['epoch'],df['val/cls_loss'],color='blue',label='validation set', marker='o',markersize=4,alpha = 0.7)
    ax2.legend(fontsize = 13)
    ax2.set_title('Classification loss', size = 16, weight = 'bold')
    plt.tight_layout()
    plt.savefig('loss_plot.png',dpi=300)
    plt.show()



if __name__ == '__main__':
    plot_results('results_run1.csv')