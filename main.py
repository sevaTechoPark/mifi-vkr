from utils.clean_original_df import make_df_clean
from utils.split_original_df import split_original_df

def main():
    make_df_clean(df, text_col="text", label_col="label")
    split_original_df()
    
    merge_augmentations()

if __name__ == "__main__":
    main()