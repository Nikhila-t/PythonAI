from pathlib import Path
import pandas as pd

## Loads data from provided excel, cleans based on specified creteria and turns into data frame
## args: file_path: location where the raw excel is store
## returns: DataFrame of the cleaned data from excel file
def load_customer_tickets(file_path:str) -> pd.DataFrame:
    
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Customer tickets data file not found at {path}")

    df_raw = pd.read_excel(path)
    df_cleaned = clean_customer_tickets_data(df_raw)

    return df_cleaned

## Performs below data cleaning steps
    ## Clear PII fields, ticketIDs
    ## Handle missing values - Text empty field to "", while Categorical empty field to Unknown
    ## Convert text fields data to lowercase
    ## Remove URLs and special characters
    ## Creates combined_text feature from subject and body fields
## args: df: DataFrame of raw excel data
## returns: DataFrame aftering cleaning the raw DF
def clean_customer_tickets_data(df: pd.DataFrame) -> pd.DataFrame:

    PII_ID_cols = [
        "ticket_id",
        "customer_name",
        "customer_email",
        "customer_phone"
    ]

    text_cols = [
        "ticket_subject",
        "ticket_body"
    ]

    categorical_cols = [
        "product_module",
        "customer_tier",
        "priority"
    ]

    ## Clear PII fields, ticketIDs
    df.drop(columns=[c for c in PII_ID_cols if c in df.columns], inplace=True)

    ## Handle missing values - Text empty field to "", while Categorical empty field to "Unknown"
    text_cols = [c for c in text_cols if c in df.columns]
    cat_cols = [c for c in categorical_cols if c in df.columns]

    df[text_cols] = df[text_cols].fillna("")
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    ## Convert text fields data to lowercase
    ## Remove URLs and special characters  
    df[text_cols] = pd.DataFrame(
                            {
                                c: df[c]
                                    .astype(str)
                                    .str.lower()                                             # convert to lower
                                    .str.replace(r"http\S+|www\S+", "", regex=True)          # remove URLs
                                    .str.replace(r"[^a-z0-9\s]", "", regex=True)             # remove special chars
                                    .str.replace(r"\s+", " ", regex=True)                    # remove unnecessary white spaces, tabs, new lines
                                    .str.strip()
                                for c in text_cols
                            },
                            index=df.index
                        )
    
    ## Ensure all categorical columns are strings
    df[cat_cols] = df[cat_cols].astype(str)

    ## Note-
    ## Performing this feature engg step here to be able to act on the file at root level and reuse it without repeating same action when needed
    
    ## Combine "ticket_subject" and "ticket_body" to form new text field "combined_text".
    df["combined_text"] = df["ticket_subject"] + " " + df["ticket_body"]

    return df