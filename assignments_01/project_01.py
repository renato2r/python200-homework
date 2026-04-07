import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from prefect import task, flow, get_run_logger

# Absolute path setup
CURRENT_DIR = Path(__file__).parent.resolve()
RESOURCES_DIR = CURRENT_DIR / "resources"
OUTPUT_DIR = CURRENT_DIR / "outputs"

@task(retries=3, retry_delay_seconds=2)
def load_and_merge_data():
    """
    Task 1: Load 10 yearly CSVs, standardize columns, and merge.
    """
    logger = get_run_logger()
    data_frames = []
    years = list(range(2015, 2025))

    column_names = [
        'ranking', 'country', 'region', 'happiness_score', 'gdp_per_capita',
        'social_support', 'health', 'freedom', 'generosity', 'corruption'
    ]

    for year in years:
        file_name = f"world_happiness_{year}.csv"
        file_path = RESOURCES_DIR / file_name
        
        if file_path.exists():
            try:
                df = pd.read_csv(
                    file_path, sep=';', decimal=',', encoding='utf-8-sig',
                    header=0, names=column_names
                )
                df['year'] = year
                data_frames.append(df)
            except Exception as e:
                logger.error(f"Error loading {file_name}: {e}")
                raise
    
    if data_frames:
        merged_df = pd.concat(data_frames, ignore_index=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(OUTPUT_DIR / "merged_happiness.csv", index=False)
        return merged_df
    return None

@task
def create_visualizations(df):
    """
    Task 3: Save 4 charts to the outputs folder.
    """
    logger = get_run_logger()
    
    # 1. Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(df['happiness_score'], kde=True)
    plt.savefig(OUTPUT_DIR / "happiness_histogram.png")
    plt.close()

    # 2. Boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='year', y='happiness_score')
    plt.xticks(rotation=45)
    plt.savefig(OUTPUT_DIR / "happiness_by_year.png")
    plt.close()

    # 3. Scatter
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='gdp_per_capita', y='happiness_score', alpha=0.5)
    plt.savefig(OUTPUT_DIR / "gdp_vs_happiness.png")
    plt.close()

    # 4. Heatmap
    plt.figure(figsize=(12, 10))
    num_df = df.select_dtypes(include=['number']).drop(columns=['year', 'ranking'], errors='ignore')
    sns.heatmap(num_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png")
    plt.close()
    
    logger.info("Task 3: Visualizations generated successfully.")

@task
def run_analytics(df):
    """
    Tasks 4 & 5: Run hypothesis tests and correlations.
    """
    # T-Test: 2019 vs 2020
    s19 = df[df['year'] == 2019]['happiness_score'].dropna()
    s20 = df[df['year'] == 2020]['happiness_score'].dropna()
    _, p_val_pandemic = stats.ttest_ind(s19, s20)

    # Correlation with Bonferroni
    exclude = ['happiness_score', 'ranking', 'year']
    vars_to_test = [c for c in df.select_dtypes(include=['number']).columns if c not in exclude]
    adj_alpha = 0.05 / len(vars_to_test)
    
    strongest_var = None
    max_r = -1
    
    for var in vars_to_test:
        valid = df[[var, 'happiness_score']].dropna()
        r, p = stats.pearsonr(valid[var], valid['happiness_score'])
        if p < adj_alpha and abs(r) > max_r:
            max_r = abs(r)
            strongest_var = var

    return {
        "p_val_pandemic": p_val_pandemic,
        "strongest_var": strongest_var,
        "strongest_r": max_r
    }

@task
def generate_summary_report(df, analytics_results):
    """
    Task 6: Human-readable summary logged and saved to a TXT file.
    """
    logger = get_run_logger()
    report_lines = []

    # 1. Dataset Scale
    total_countries = df['country'].nunique()
    total_years = df['year'].nunique()
    report_lines.append(f"REPORT: The dataset covers {total_countries} unique countries over {total_years} years.")

    # 2. Regional Rankings
    reg_mean = df.groupby('region')['happiness_score'].mean().sort_values(ascending=False)
    top_3 = reg_mean.head(3).index.tolist()
    bot_3 = reg_mean.tail(3).index.tolist()
    report_lines.append(f"REPORT: Top 3 Happiest Regions: {', '.join(top_3)}")
    report_lines.append(f"REPORT: Bottom 3 Regions: {', '.join(bot_3)}")

    # 3. Pandemic Impact
    p_val = analytics_results['p_val_pandemic']
    impact_msg = "there was a statistically significant change" if p_val < 0.05 else "global happiness scores remained stable"
    report_lines.append(f"REPORT: Regarding the 2019-2020 transition, {impact_msg} (p={p_val:.4f})")

    # 4. Strongest Driver
    best_var = analytics_results['strongest_var'].replace('_', ' ').title()
    best_r = analytics_results['strongest_r']
    report_lines.append(f"REPORT: '{best_var}' is the strongest predictor of happiness (Correlation: {best_r:.3f}).")

    # Log to Prefect and Terminal
    for line in report_lines:
        logger.info(line)

    # Save to physical file
    report_path = OUTPUT_DIR / "summary_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    logger.info(f"Summary report saved to {report_path}")

@flow(name="World Happiness Final Pipeline")
def happiness_flow():
    """
    Orchestrates the full process from ingestion to reporting.
    """
    df = load_and_merge_data()
    
    if df is not None:
        create_visualizations(df)
        results = run_analytics(df)
        generate_summary_report(df, results)

if __name__ == "__main__":
    happiness_flow()