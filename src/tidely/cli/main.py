"""CLI Interface for Tidely."""

import argparse
import sys
import tidely as td

def main():
    parser = argparse.ArgumentParser(description="Tidely: The Intelligence Layer for Tabular Data")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Clean Command
    clean_parser = subparsers.add_parser("clean", help="Clean a dataset automatically")
    clean_parser.add_argument("input_file", help="Path to the raw CSV file")
    clean_parser.add_argument("--out", "-o", default="cleaned_data.csv", help="Output file path")
    
    # Inspect Command
    inspect_parser = subparsers.add_parser("inspect", help="Inspect a dataset and generate a profile")
    inspect_parser.add_argument("input_file", help="Path to the raw CSV file")
    
    # Report Command
    report_parser = subparsers.add_parser("report", help="Generate a rich report for a dataset")
    report_parser.add_argument("input_file", help="Path to the raw CSV file")
    report_parser.add_argument("--format", "-f", default="html", choices=["html", "pdf"], help="Report format")
    
    args = parser.parse_args()
    
    if args.command == "clean":
        print(f"Loading {args.input_file}...")
        try:
            df = td.load(args.input_file)
            result = td.clean(df)
            print(result.summary())
            td.save(result.df, args.out)
            print(f"\nSaved cleaned data to {args.out}")
        except Exception as e:
            print(f"Error cleaning data: {e}")
            sys.exit(1)
            
    elif args.command == "inspect":
        print(f"Inspecting {args.input_file}...")
        try:
            df = td.load(args.input_file)
            profile = td.inspect(df)
            # Simple terminal output for now
            print(f"Rows: {profile.row_count} | Columns: {profile.col_count}")
            print(f"Overall Trust Score: {profile.trust_score.overall}/100")
        except Exception as e:
            print(f"Error inspecting data: {e}")
            sys.exit(1)
            
    elif args.command == "report":
        print(f"Generating {args.format.upper()} report for {args.input_file}...")
        try:
            df = td.load(args.input_file)
            result = td.clean(df)
            if args.format == "html":
                result.export_html("tidely_report.html")
                print("Exported to tidely_report.html")
            elif args.format == "pdf":
                result.export_pdf("tidely_report.pdf")
        except Exception as e:
            print(f"Error generating report: {e}")
            sys.exit(1)
            
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
