import json
import os

for i in range(1, 7):
    report_file = f"C:\\Users\\Aaryan Rawat\\.gemini\\antigravity\\brain\\84be442f-3418-4bbc-830d-3a5c2fbe0dc2\\qa3_report_ds{i}.md"
    try:
        with open(f"qa3_ds{i}_baseline.json") as f:
            base = json.load(f)
        with open(f"qa3_ds{i}_inspection.json") as f:
            insp = json.load(f)
        with open(f"qa3_ds{i}_cleaning.json") as f:
            clean = json.load(f)
            
        content = f"""# QA Report Dataset {i}
        
## 1. Manual Inspection Baseline
- Rows: {base['num_rows']}
- Columns: {base['num_cols']}
- Duplicates: {base['duplicate_rows']}

## 2. Inspection Accuracy
- Trust Score: {insp['trust_score']}

## 3. Cleaning Accuracy
```text
{clean['summary']}
```
"""
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Generated {report_file}")
    except Exception as e:
        print(f"Skipping DS{i} - data not ready yet: {e}")
