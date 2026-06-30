import json

import pandas as pd

import tidely as td


def tidely_inspection():
    df = pd.read_csv("qa_ds2_temp.csv")

    # Run Tidely Inspection
    profile = td.inspect(df)

    report = {
        "trust_score": profile.trust_score.__dict__,
        "dna_domain": profile.dna.domain,
        "dna_entities": profile.dna.entities,
        "semantic_types": profile.semantic_types,
        "row_count": profile.row_count,
        "col_count": profile.col_count,
    }

    with open("qa_ds2_tidely_inspection.json", "w") as f:
        json.dump(report, f, indent=4)

    print("Saved Tidely inspection report to qa_ds2_tidely_inspection.json")


if __name__ == "__main__":
    tidely_inspection()
