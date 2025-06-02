import numpy as np

from practice.exercices.hypothesis_testing.utils.stat_test_practice_dataclass import (
    ProblemContext,
)

target = np.random.normal(1000, 200, 50)
correlated = target + np.random.normal(0, 50, 50)

mock_problems = {
    "sales_comparison": ProblemContext(
        problem_statement="Une entreprise veut savoir si ses ventes ont augmenté après une campagne publicitaire. Elle compare les ventes d’un échantillon de magasins avant et après la campagne.",
        data=(np.random.gamma(4, 5, 1000) + 100, np.random.gamma(6, 5, 1000) + 100),
        hints=[],
    ),
    # simple cases
    "variance_comparison": ProblemContext(
        problem_statement="On veut comparer les variances de deux échantillons, afin de savoir si ces dernières sont identiques ou non.",
        data=(np.random.uniform(0, 10, 1000), np.random.normal(0, 10, 1000)),
        hints=[
            "On peut utiliser le test de Levene pour comparer les variances de deux échantillons.",
            "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bartlett.html",
        ],
    ),
    "multiple_means_comparisons": ProblemContext(
        problem_statement="On veut comparer les moyennes de trois groupes de données pour savoir si toutes sont identiques ou si au moins une est différente.",
        data=(
            np.random.uniform(0, 10, 100),
            np.random.normal(5, 10, 100),
            np.random.exponential(scale=5.0, size=100),
        ),
        hints=[
            "On peut utiliser une ANOVA pour comparer les moyennes de plusieurs groupes.",
            "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html",
        ],
    ),
    "normality_test": ProblemContext(
        problem_statement="On veut s'assurer que la distribution étudiée suit bien une loi normale.",
        data=np.random.laplace(0, 1, 100),
        hints=[
            "On peut utiliser le test de Shapiro-Wilk pour tester la normalité d'une distribution.",
            "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html",
        ],
    ),
    # intermediate cases
    "binary_variable_comparison": ProblemContext(
        problem_statement="Une entreprise de ressources humaines souhaite savoir si les scores de performance diffèrent entre deux groupes d'employés (ceux ayant suivi une formation avancée et ceux n'en ayant pas suivi), afin d’évaluer l'impact de la formation sur la performance.",
        data=(
            np.random.normal(70, 10, 50),  # Scores de performance (quantitative)
            np.random.choice(
                ["ont suivi une formation", "n'ont pas suivi de formation"], 50
            ),  # Variable binaire indiquant si l'employé a suivi la formation (1) ou non (0)
        ),
        hints=[
            "Pour déterminer si les moyennes des groupes diffèrent de manière significative, un test de comparaison de moyennes peut être envisagé.",
            "Il faut au préalable créer deux sous échantillons à partir de la variable binaire donnéee",
            "Le test t de Student, ou le test de Mann-Whitney si les distributions sont non normales, peut être utile pour cette analyse.",
        ],
    ),
    "normality_large_sample": ProblemContext(
        problem_statement="On veut tester la normalité d'un échantillon de grande taille.",
        data=np.random.chisquare(df=2, size=1000),
        hints=[
            "On peut utiliser le test de Lilliefors ou d'Anderson-Darling pour tester la normalité d'un échantillon de grande taille, en utilisant la fonction de répartition d'une loi normale.",
            "https://www.statsmodels.org/dev/generated/statsmodels.stats.diagnostic.lilliefors.html",
        ],
    ),
    # end to end
    "customer_purchase_category": ProblemContext(
        problem_statement="Un analyste souhaite vérifier si la distribution des catégories de produits achetés diffère entre deux magasins, pour voir si les préférences de produits varient d'un magasin à l'autre.",
        data=(
            np.array(
                [30, 45, 25]
            ),  # Fréquences pour le magasin A dans trois catégories de produits
            np.array(
                [35, 40, 25]
            ),  # Fréquences pour le magasin B dans trois catégories de produits
        ),
        hints=[
            "On peut comparer deux distributions pour voir si elles sont statistiquement similaires.",
            "Un test du chi² permet de déterminer si les distributions observées diffèrent de manière significative.",
        ],
    ),
    "service_ratings_comparison": ProblemContext(
        problem_statement="Un hôtel veut savoir si la satisfaction des clients diffère entre trois types de services (ex. : chambre, restaurant, spa), afin de cibler les améliorations en fonction des retours clients.",
        data=(
            np.random.randint(1, 5, 50),  # Notes des clients pour le service de chambre
            np.random.randint(1, 5, 50),  # Notes pour le restaurant
            np.random.randint(1, 5, 50),  # Notes pour le spa
        ),
        hints=[
            "On peut comparer plusieurs groupes sans faire d'hypothèse de normalité des données pour voir si des différences existent.",
            "Le test de Kruskal-Wallis peut permettre d’évaluer si les évaluations moyennes diffèrent significativement.",
        ],
    ),
    "sales_vs_ads": ProblemContext(
        problem_statement="L'équipe de vente souhaite comprendre si les ventes mensuelles sont corrélées avec le budget publicitaire mensuel pour optimiser l'allocation des ressources.",
        data=(target, correlated),
        hints=[
            "La corrélation permet de déterminer s'il existe une relation linéaire entre deux variables.",
            "Un test de corrélation de Pearson pourrait aider à évaluer cette relation.",
        ],
    ),
}
