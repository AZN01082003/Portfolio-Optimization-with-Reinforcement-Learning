# Visualisations

"""
Module pour les visualisations liées au portefeuille et à la performance.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def plot_portfolio_weights(weights, tickers, save_path=None):
    """
    Tracer l'évolution des poids du portefeuille au fil du temps.
    
    Args:
        weights: Tableau des poids (n_steps, n_stocks)
        tickers: Liste des symboles d'actions
        save_path: Chemin pour sauvegarder le graphique
    """
    weights_array = np.array(weights)
    plt.figure(figsize=(12, 8))
    
    # Heatmap des poids
    plt.subplot(1, 2, 1)
    sns.heatmap(weights_array.T, cmap="viridis", yticklabels=tickers)
    plt.title("Poids du Portefeuille (Heatmap)")
    plt.xlabel("Pas de temps")
    plt.ylabel("Action")
    
    # Graphique empilé
    plt.subplot(1, 2, 2)
    plt.stackplot(range(weights_array.shape[0]), weights_array.T, labels=tickers, alpha=0.8)
    plt.title("Poids du Portefeuille (Empilé)")
    plt.xlabel("Pas de temps")
    plt.ylabel("Poids")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Graphique des poids sauvegardé dans {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_performance_metrics(returns, benchmark_returns=None, save_path=None):
    """
    Tracer les métriques de performance du portefeuille.
    
    Args:
        returns: Tableau des rendements quotidiens
        benchmark_returns: Tableau des rendements du benchmark (optionnel)
        save_path: Chemin pour sauvegarder le graphique
    """
    returns_array = np.array(returns)
    
    # Calculer les rendements cumulés
    cumulative_returns = np.cumprod(1 + returns_array) - 1
    
    plt.figure(figsize=(15, 12))
    
    # Rendements cumulés
    plt.subplot(2, 2, 1)
    plt.plot(cumulative_returns * 100, label="Portefeuille")
    if benchmark_returns is not None:
        benchmark_returns_array = np.array(benchmark_returns)
        benchmark_cumulative = np.cumprod(1 + benchmark_returns_array) - 1
        plt.plot(benchmark_cumulative * 100, label="Benchmark", linestyle="--")
    plt.title("Rendement Cumulé (%)")
    plt.xlabel("Pas de temps")
    plt.ylabel("Rendement (%)")
    plt.legend()
    plt.grid(True)
    
    # Distribution des rendements
    plt.subplot(2, 2, 2)
    sns.histplot(returns_array * 100, kde=True, label="Portefeuille")
    if benchmark_returns is not None:
        sns.histplot(benchmark_returns_array * 100, kde=True, label="Benchmark", color="orange")
    plt.title("Distribution des Rendements Quotidiens")
    plt.xlabel("Rendement (%)")
    plt.ylabel("Fréquence")
    plt.legend()
    plt.grid(True)
    
    # Drawdowns
    plt.subplot(2, 2, 3)
    
    def calculate_drawdowns(returns):
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        return drawdowns
    
    drawdowns = calculate_drawdowns(returns_array)
    plt.plot(drawdowns * 100)
    plt.title("Drawdowns")
    plt.xlabel("Pas de temps")
    plt.ylabel("Drawdown (%)")
    plt.grid(True)
    
    # Rendements roulants annualisés
    plt.subplot(2, 2, 4)
    window = min(252, len(returns_array) // 2)  # 1 an ou moins si pas assez de données
    
    if len(returns_array) > window:
        rolling_returns = pd.Series(returns_array).rolling(window).mean() * 252 * 100  # Annualisé
        plt.plot(rolling_returns, label="Portefeuille")
        
        if benchmark_returns is not None and len(benchmark_returns_array) > window:
            rolling_benchmark = pd.Series(benchmark_returns_array).rolling(window).mean() * 252 * 100  # Annualisé
            plt.plot(rolling_benchmark, label="Benchmark", linestyle="--")
        
        plt.title(f"Rendement Roulant Annualisé ({window} jours)")
        plt.xlabel("Pas de temps")
        plt.ylabel("Rendement (%)")
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Graphique des métriques de performance sauvegardé dans {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_backtest_summary(results_dir, benchmark_comparison=True, save_path=None):
    """
    Crée un résumé visuel des résultats de backtest.
    
    Args:
        results_dir: Répertoire contenant les résultats d'évaluation
        benchmark_comparison: Si True, inclut la comparaison avec le benchmark
        save_path: Chemin pour sauvegarder le graphique
    """
    # Charger les données d'évaluation
    data_path = os.path.join(results_dir, "test_evaluation_data.npy")
    if not os.path.exists(data_path):
        logger.error(f"Fichier de données d'évaluation non trouvé: {data_path}")
        return
    
    data = np.load(data_path, allow_pickle=True).item()
    
    # Extraire les données
    values = data['values']
    returns = data['returns']
    drawdowns = data['drawdowns']
    weights = data['weights']
    
    plt.figure(figsize=(15, 10))
    
    # Valeur du portefeuille
    plt.subplot(2, 2, 1)
    plt.plot(values)
    plt.title("Valeur du Portefeuille")
    plt.xlabel("Pas de temps")
    plt.ylabel("Valeur ($)")
    plt.grid(True)
    
    # Rendement cumulé
    plt.subplot(2, 2, 2)
    cumulative_returns = np.cumprod(1 + np.array(returns)) - 1
    plt.plot(cumulative_returns * 100)
    plt.title("Rendement Cumulé")
    plt.xlabel("Pas de temps")
    plt.ylabel("Rendement (%)")
    plt.grid(True)
    
    # Drawdowns
    plt.subplot(2, 2, 3)
    plt.plot(np.array(drawdowns) * 100)
    plt.title("Drawdowns")
    plt.xlabel("Pas de temps")
    plt.ylabel("Drawdown (%)")
    plt.grid(True)
    
    # Volatilité roulante
    plt.subplot(2, 2, 4)
    window = min(30, len(returns) // 4)  # 1 mois ou moins si pas assez de données
    if len(returns) > window:
        rolling_vol = pd.Series(returns).rolling(window).std() * np.sqrt(252) * 100  # Annualisée
        plt.plot(rolling_vol)
        plt.title(f"Volatilité Roulante ({window} jours)")
        plt.xlabel("Pas de temps")
        plt.ylabel("Volatilité (%)")
        plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Résumé du backtest sauvegardé dans {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Tracer les poids du portefeuille
    weights_save_path = save_path.replace(".png", "_weights.png") if save_path else None
    plot_portfolio_weights(weights, [f"Asset {i+1}" for i in range(len(weights[0]))], weights_save_path)

def generate_performance_report(results_dir, save_path=None):
    """
    Génère un rapport de performance complet.
    
    Args:
        results_dir: Répertoire contenant les résultats d'évaluation
        save_path: Chemin pour sauvegarder le rapport
    """
    # Charger les données d'évaluation
    data_path = os.path.join(results_dir, "test_evaluation_data.npy")
    if not os.path.exists(data_path):
        logger.error(f"Fichier de données d'évaluation non trouvé: {data_path}")
        return
    
    data = np.load(data_path, allow_pickle=True).item()
    
    # Extraire les données
    values = data['values']
    returns = data['returns']
    drawdowns = data['drawdowns']
    
    # Calculer les métriques de performance
    total_return = (values[-1] / values[0] - 1) * 100
    annualized_return = ((1 + np.mean(returns)) ** 252 - 1) * 100
    volatility = np.std(returns) * np.sqrt(252) * 100
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    max_drawdown = np.min(drawdowns) * 100
    
    # Calculer les métriques avancées
    returns_array = np.array(returns)
    
    def calculate_sortino_ratio(returns, risk_free=0):
        downside_returns = returns[returns < risk_free]
        if len(downside_returns) > 0 and np.std(downside_returns) > 0:
            downside_deviation = np.std(downside_returns) * np.sqrt(252)
            return ((1 + np.mean(returns)) ** 252 - 1 - risk_free) / downside_deviation
        return 0
    
    def calculate_calmar_ratio(returns, max_drawdown):
        if max_drawdown != 0:
            annual_return = ((1 + np.mean(returns)) ** 252 - 1)
            return -annual_return / (max_drawdown / 100)
        return 0
    
    sortino_ratio = calculate_sortino_ratio(returns_array)
    calmar_ratio = calculate_calmar_ratio(returns_array, max_drawdown)
    
    # Calculer les statistiques de distribution
    skewness = pd.Series(returns_array).skew()
    kurtosis = pd.Series(returns_array).kurtosis()
    
    # Créer un DataFrame des métriques
    metrics = {
        'Metric': [
            'Rendement Total (%)', 
            'Rendement Annualisé (%)', 
            'Volatilité Annualisée (%)', 
            'Ratio de Sharpe', 
            'Ratio de Sortino',
            'Ratio de Calmar',
            'Drawdown Maximum (%)',
            'Skewness',
            'Kurtosis'
        ],
        'Value': [
            f"{total_return:.2f}",
            f"{annualized_return:.2f}",
            f"{volatility:.2f}",
            f"{sharpe_ratio:.2f}",
            f"{sortino_ratio:.2f}",
            f"{calmar_ratio:.2f}",
            f"{-max_drawdown:.2f}",
            f"{skewness:.2f}",
            f"{kurtosis:.2f}"
        ]
    }
    
    metrics_df = pd.DataFrame(metrics)
    
    # Sauvegarder en CSV
    if save_path:
        csv_path = save_path.replace('.png', '.csv') if save_path.endswith('.png') else save_path + '.csv'
        metrics_df.to_csv(csv_path, index=False)
        logger.info(f"Rapport de performance sauvegardé dans {csv_path}")
    
    # Afficher les métriques
    logger.info("\nRapport de Performance:")
    for index, row in metrics_df.iterrows():
        logger.info(f"{row['Metric']}: {row['Value']}")
    
    return metrics_df