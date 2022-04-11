import matplotlib.pyplot as plt

def plot_model_comparison(results, figsize = (15, 8), y = 'mse'):
    fig, ax = plt.subplots(figsize = figsize)
    sns.boxplot(data = results, y = y, x="model", ax = ax)
    ax.set_xlabel("", size=40)
    ax.set_ylabel("MSE", size=20)
    ax.set_title("Estimators vs MSE", size=30)
    plt.show()