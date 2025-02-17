import pandas as pd

# Filvägar
SCB_FILE = "scb_bulkfil.txt"  # Tab-separerad fil med SNI-koder
BOLAGSVERKET_FILE = "bolagsverket_bulkfil.txt"  # Semikolon-separerad fil med verksamhetsbeskrivningar
OUTPUT_FILE = "processed_data.csv"  # Filen vi sparar efter att ha matchat och rensat datan

def load_and_clean_data():
    """Laddar in och förbereder data från SCB och Bolagsverket"""
    
    # Läs in SCB-data (Tab-separerad)
    scb_df = pd.read_csv(SCB_FILE, sep="\t", dtype=str, on_bad_lines='skip', low_memory=False)
    print(scb_df.head())
    
    # Läs in Bolagsverket-data (Semikolon-separerad), hantera citattecken och dåliga rader
    bolagsverket_df = pd.read_csv(BOLAGSVERKET_FILE, sep=";", dtype=str, on_bad_lines='skip', low_memory=False, quotechar='"')
    print(bolagsverket_df.head())
    # Kontrollera att rätt kolumner finns
    if "organisationsidentitet" not in bolagsverket_df.columns or "PeOrgNr" not in scb_df.columns:
        raise ValueError("Kolumnnamn matchar inte för organisationsnummer!")

    # Byt namn på kolumnen i SCB-filen så att vi kan matcha
    scb_df.rename(columns={"PeOrgNr": "organisationsnummer"}, inplace=True)

    # Rensa SCB-filen från 16-prefix i organisationsnumret
    scb_df["organisationsnummer"] = scb_df["organisationsnummer"].str[2:]  # Ta bort de första två siffrorna (16)

    # Rensa Bolagsverket-filen från extra suffix i organisationsnumret
    bolagsverket_df["organisationsnummer"] = bolagsverket_df["organisationsidentitet"].str.extract(r"(\d+)")

    # Slå ihop data på organisationsnummer
    merged_df = pd.merge(bolagsverket_df, scb_df, on="organisationsnummer", how="inner")

    # Behåll endast relevanta kolumner
    final_df = merged_df[["organisationsnummer", "verksamhetsbeskrivning", "Ng1"]].copy()
    
    # Byt namn på SNI-kolumn för tydlighet
    final_df.rename(columns={"Ng1": "sni_kod"}, inplace=True)

    # Rensa verksamhetsbeskrivning: Ta bort onödiga tecken, små bokstäver och trimma whitespace
    final_df["verksamhetsbeskrivning"] = (
        final_df["verksamhetsbeskrivning"]
        .str.replace(r"[^\w\s]", "", regex=True)  # Ta bort specialtecken
        .str.lower()
        .str.strip()
    )

    # Spara resultatet
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Data har förbehandlats och sparats i {OUTPUT_FILE}")

if __name__ == "__main__":
    load_and_clean_data()
