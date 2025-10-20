from src import (
    week1_setup, week2_etl, week3_features,
    week4_baseline, week5_advanced_models,
    week6_tuning, week7_final_eval
)

def main():
    week1_setup.run()
    week2_etl.run()
    week3_features.run()
    week4_baseline.run()
    week5_advanced_models.run()
    week6_tuning.run()
    week7_final_eval.run()

if __name__ == "__main__":
    main()
