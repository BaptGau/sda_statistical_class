import seaborn as sns

if __name__ == "__main__":
    data = sns.load_dataset(name="mpg")
    print(data.info())

    # todo : Analyse exploratoire des données
